import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from utility_study.core import (
    PooledMetrics, Snapshot, ce_loss, choose_coefficient, evaluation_mode,
    isolated_rng, kd_loss, restored_state, rng_state, tree_equal, virtual_outcome,
)
from utility_study.register import monitoring_indices, paired_updates, validate_checkpoint_metadata
from utility_study.workflow import gate_from_values, phase_jobs, validate_gpus, pop_ready_job
from utility_study.runner import selected_seeds, selected_stages, shard_name, wait_for_idle_gpu
from utility_study.migrate import dataset_paths
from utility_study.continuation import amend, accepts_record, completed_job, scientific_contract, validate_continuation
from utility_study.io import read_json, sha256, source_hashes, write_json


@pytest.mark.parametrize("mode,student_t", [("teacher", 1.0), ("shared", 2.0)])
@pytest.mark.parametrize("coefficient", [1.0, 4.0])
def test_masked_kd_has_the_declared_gradient_and_no_teacher_gradient(mode, student_t, coefficient):
    torch.manual_seed(33)
    z = torch.randn(2, 3, 2, 3, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(z, requires_grad=True)
    target = torch.zeros(2, 4, 6, dtype=torch.long)
    target[:, :2, :2] = -1
    loss = coefficient * kd_loss(z, teacher, target, mode)
    loss.backward()
    valid = torch.nn.functional.interpolate((target != -1)[:, None].double(), (2, 3), mode="nearest")[:, 0].bool()
    expected = coefficient * (torch.softmax(z.detach() / student_t, 1) - torch.softmax(teacher.detach() / 2, 1))
    expected = expected * valid[:, None] / valid.sum() / student_t
    torch.testing.assert_close(z.grad, expected, atol=1e-12, rtol=1e-12)
    assert teacher.grad is None


def test_pixel_ce_can_worsen_while_both_class_ce_and_iou_improve():
    target = torch.zeros(1, 10, 10, dtype=torch.long)
    target[:, -1] = 1
    before = torch.full((1, 2, 10, 10), .01, dtype=torch.float64)
    before[:, 0] = .99
    before[:, 0, -1] = .9
    before[:, 1, -1] = .1
    after = torch.full_like(before, .25)
    after[:, 0] = .75
    after[:, 0, -1] = .1
    after[:, 1, -1] = .9
    a, b = PooledMetrics(2), PooledMetrics(2)
    a.add(before.log(), target)
    b.add(after.log(), target)
    a, b = a.result(), b.result()
    assert b["pixel_ce"] > a["pixel_ce"]
    assert b["class_ce"] < a["class_ce"]
    assert b["soft_miou"] > a["soft_miou"]
    assert b["hard_miou"] > a["hard_miou"]
    assert a["gt_counts"] == [90, 10]


def test_pooled_reductions_ignore_padding_and_are_invariant_to_batch_partition():
    torch.manual_seed(71)
    z = torch.randn(3, 3, 2, 3, dtype=torch.float64)
    y = torch.randint(0, 3, (3, 4, 6))
    y[0, :2] = -1
    whole, pieces = PooledMetrics(3), PooledMetrics(3)
    whole.add(z, y)
    for i in range(3):
        pieces.add(z[i:i+1], y[i:i+1])
    a, b = whole.result(), pieces.result()
    for field in ("pixel_ce", "class_ce", "soft_miou", "hard_miou"):
        assert a[field] == pytest.approx(b[field], abs=1e-14)
    assert a["confusion"] == b["confusion"]
    assert a["valid_pixels"] == int((y != -1).sum())
    assert a["pixel_ce"] == pytest.approx(float(ce_loss(z, y)), abs=1e-12)


def test_soft_iou_uses_pooled_unions_not_image_mean():
    y = torch.tensor([[[0, 0]], [[1, 1]]])
    z = torch.tensor([[[[3., 3.]], [[0., 0.]]], [[[3., 3.]], [[0., 0.]]]], dtype=torch.float64)
    result = PooledMetrics(2)
    result.add(z, y)
    r = result.result()
    for k in range(2):
        expected = r["soft_intersection"][k] / (r["gt_counts"][k] + r["soft_prediction"][k] - r["soft_intersection"][k])
        assert r["soft_iou"][k] == pytest.approx(expected)


def test_all_ignore_and_invalid_labels_fail_explicitly():
    z = torch.zeros(1, 2, 2, 2)
    y = torch.full((1, 2, 2), -1)
    with pytest.raises(ValueError):
        ce_loss(z, y)
    with pytest.raises(ValueError):
        kd_loss(z, z, y)
    metric = PooledMetrics(2)
    metric.add(z, y)
    with pytest.raises(ValueError):
        metric.result()
    with pytest.raises(ValueError):
        metric.add(z, torch.full_like(y, 2))


def small_problem():
    torch.manual_seed(14)
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Dropout(.3), nn.Conv2d(4, 2, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.01)
    x, y = torch.randn(4, 3, 3, 3), torch.randint(0, 2, (4, 3, 3))
    ce_loss(model(x), y).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model[2].eval()  # A mixed training flag must also survive restoration.
    return model, optimizer, x, y


def test_virtual_step_restores_momentum_bn_modes_gradients_and_all_rng_on_exception():
    model, optimizer, x, y = small_problem()
    saved = Snapshot.capture(model, optimizer)
    with pytest.raises(RuntimeError, match="injected"):
        with restored_state(model, optimizer):
            model.train()
            ce_loss(model(x), y).backward()
            optimizer.step()
            optimizer.param_groups[0]["lr"] = .9
            random.random(); np.random.rand(); torch.rand(4)
            assert not tree_equal(saved.model, model.state_dict())
            raise RuntimeError("injected")
    saved.verify(model, optimizer)


def test_replaying_the_same_train_branch_is_exact_and_monitoring_does_not_change_bn():
    model, optimizer, x, y = small_problem()
    snapshot = Snapshot.capture(model, optimizer)
    results = []
    for _ in range(2):
        with restored_state(model, optimizer):
            with isolated_rng(99):
                model.train()
                for step in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    ce_loss(model(x), y).backward()
                    optimizer.step()
                state = copy.deepcopy(model.state_dict())
                with evaluation_mode(model), torch.no_grad():
                    _ = model(x)
                assert tree_equal(state, model.state_dict())
                results.append(state)
        snapshot.verify(model, optimizer)
    assert tree_equal(*results)


def test_metric_is_the_only_choice_difference_and_ties_keep_the_previous_coefficient():
    scores = {0.: 1., 1.: 1. + 1e-8, 4.: .5}
    assert choose_coefficient(scores, "pixel_ce") == 4.
    assert choose_coefficient(scores, "soft_miou") == 1.
    assert choose_coefficient({0.: 0., 4.: 0.}, "soft_miou", previous=1.) == 0.
    with pytest.raises(ValueError):
        choose_coefficient({1.: float("nan")}, "pixel_ce")


def test_isolated_data_augmentation_rng_leaves_main_training_stream_unchanged():
    old = rng_state()
    with isolated_rng(20261015):
        random.random(); np.random.rand(); torch.rand(5)
    assert tree_equal(old, rng_state())


def test_monitor_selection_has_class_coverage_is_unique_and_is_reproducible():
    presence = [[0, 1] if i % 2 else [0, 2] for i in range(400)]
    selected = monitoring_indices(presence, 3)
    assert len(selected) == len(set(selected)) == 256
    assert all(sum(k in presence[i] for i in selected) >= 8 for k in range(3))
    assert selected == monitoring_indices(presence, 3)
    assert monitoring_indices(presence, 3, seed=41) != selected
    with pytest.raises(ValueError):
        monitoring_indices([[0]] * 400, 2)


def test_primary_gate_requires_both_datasets_and_does_not_authorize_full_training():
    values = {(dataset, seed, stage): 1. for dataset in ("voc", "cityscapes")
              for seed in (1234, 2025, 3407) for stage in (20000, 40000, 60000)}
    gate = gate_from_values(values)
    assert gate["pass"] and not gate["full_training_authorized_by_this_gate_alone"]
    for seed in (1234, 2025, 3407):
        values["cityscapes", seed, 20000] = -1.
    assert gate_from_values(values)["pass"]
    for seed in (1234, 2025, 3407):
        values["cityscapes", seed, 40000] = -1.
    assert not gate_from_values(values)["pass"]


def test_newly_registered_kd_intermediates_must_match_the_audited_run():
    args = {"seed": 1234, "dataset": "voc", "student_model": "deeplabv3", "student_backbone": "large",
            "lambda_kd": 4., "kd_loss_mode": "masked", "batch_size": 16, "max_iterations": 80000,
            "lr": .02, "teacher_pretrained": "/teacher.pth", "crop_size": [512, 512]}
    state = {"iteration": 20000, "student": {}, "optimizer": {}, "args": copy.deepcopy(args)}
    run = {"args": args}
    validate_checkpoint_metadata(state, run, 20000)
    state["args"]["seed"] = 2025
    with pytest.raises(ValueError, match="seed"):
        validate_checkpoint_metadata(state, run, 20000)
    state["args"]["seed"] = 1234
    with pytest.raises(ValueError, match="iteration"):
        validate_checkpoint_metadata(state, run, 40000)


def test_mode_comparison_uses_identical_images_and_augmentation():
    eligible = list(range(256, 1000))
    updates = paired_updates(eligible, 16, seed=1234, stage=20000)
    assert updates["matched_train"] == updates["matched_eval"]
    assert len(updates["matched_train"]) == 128
    for legacy, full in zip(updates["legacy_eval"], updates["matched_train"]):
        assert legacy["indices"] == full["indices"][:2]
        assert legacy["augmentation_seed"] == full["augmentation_seed"]
        assert not set(full["indices"]) & set(range(256))


def test_numerical_divergence_is_an_outcome_and_still_restores_the_state():
    model, optimizer, x, y = small_problem()
    original = Snapshot.capture(model, optimizer)
    with virtual_outcome(model, optimizer) as result:
        model.train()
        ce_loss(model(x), y).backward()
        optimizer.step()
        raise FloatingPointError("injected divergence")
    assert result["status"] == "DIVERGED"
    assert result["reason"] == "injected divergence"
    original.verify(model, optimizer)
    with pytest.raises(RuntimeError):
        with virtual_outcome(model, optimizer):
            raise RuntimeError("Engineering errors must not be swallowed")


def test_nonfinite_monitor_predictions_are_distinguished_from_a_wrong_class_mapping():
    y = torch.zeros(1, 2, 2, dtype=torch.long)
    with pytest.raises(FloatingPointError):
        PooledMetrics(2).add(torch.full((1, 2, 2, 2), float("nan")), y)
    with pytest.raises(ValueError):
        PooledMetrics(3).add(torch.zeros(1, 2, 2, 2), y)


def test_seed_queue_partitions_cases_without_duplicate_scientific_observations():
    registered = {"seeds": [1234, 2025, 3407], "checkpoints": [20000, 40000, 60000], "allowed_gpus": [2, 3], "max_gpus": 2}
    jobs = phase_jobs("one_step", registered["seeds"])
    assert len(jobs) == len(set(jobs)) == 18
    assert set(jobs) == {(d, s, k) for d in ("voc", "cityscapes") for s in registered["seeds"] for k in registered["checkpoints"]}
    assert phase_jobs("smoke", registered["seeds"]) == [("voc", 1234, 20000), ("cityscapes", 1234, 20000)]
    assert selected_seeds(registered, "2025") == [2025]
    validate_gpus([2, 3], registered)
    for allocation in ([0, 1], [2], [2, 2], [0, 1, 2, 3]):
        with pytest.raises(ValueError):
            validate_gpus(allocation, registered)
    for seeds in ("0", "1234,1234"):
        with pytest.raises(ValueError):
            selected_seeds(registered, seeds)
    with pytest.raises(ValueError):
        selected_seeds(registered, "2025", smoke=True)
    assert selected_stages(registered, "40000") == [40000]
    assert shard_name("1234", "20000") != shard_name("1234", "40000")
    for stages in ("80000", "20000,20000"):
        with pytest.raises(ValueError):
            selected_stages(registered, stages)
    with pytest.raises(ValueError):
        selected_stages(registered, "40000", smoke=True)


def test_pending_copies_do_not_block_other_ready_cases_and_partial_files_are_not_ready(tmp_path):
    a, b = tmp_path / "a.pth", tmp_path / "b.pth"
    a.with_suffix(".pth.partial").touch()
    b.touch()
    runs = [{"seed": 1234, "condition": "ce", "checkpoints": {"20000": str(a), "40000": str(b)}}]
    registration = {"datasets": {"voc": {"runs": runs}}}
    pending = [("voc", 1234, 20000), ("voc", 1234, 40000)]
    assert pop_ready_job(pending, registration, "ce") == ("voc", 1234, 40000)
    assert pop_ready_job(pending, registration, "ce") is None
    a.with_suffix(".pth.partial").replace(a)
    assert pop_ready_job(pending, registration, "ce") == ("voc", 1234, 20000)
    assert pending == []


@pytest.mark.parametrize("utilizations", [[43, 0, 0], [0, 43, 0, 0]])
def test_gpu_admission_waits_out_stale_utilization_and_requires_two_idle_samples(monkeypatch, utilizations):
    import utility_study.runner as runner
    samples = iter(utilizations)
    calls = []
    def query(*args, **kwargs):
        value = next(samples)
        calls.append(value)
        return f"0, 30000, 100\n3, 0, {value}\n"
    monkeypatch.setattr(runner.subprocess, "check_output", query)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.)
    wait_for_idle_gpu(3)
    assert calls == utilizations


def test_gpu_admission_times_out_without_using_an_occupied_card(monkeypatch):
    import utility_study.runner as runner
    ticks = iter([0., .1, 2.])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "3, 2048, 0\n")
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="did not become idle"):
        wait_for_idle_gpu(3, timeout=1)


@pytest.mark.parametrize("known,free,admitted", [(True, 37488, True), (False, 37488, False), (True, 18000, False)])
def test_shared_admission_preserves_known_keepers_but_requires_headroom(monkeypatch, known, free, admitted):
    import utility_study.runner as runner
    policy = {"minimum_free_mib": 28672}
    ticks = iter([0., .1, 2.])
    def query(command, **kwargs):
        return f"1, GPU-one, {free}\n" if "--query-gpu=index,uuid,memory.free" in command else "101, GPU-one\n"
    monkeypatch.setattr(runner.subprocess, "check_output", query)
    monkeypatch.setattr(runner, "authorized_keeper", lambda pid, p: pid == 101 and known)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    if admitted:
        runner.wait_for_shared_gpu(1, policy, timeout=1)
    else:
        with pytest.raises(RuntimeError, match="unknown process"):
            runner.wait_for_shared_gpu(1, policy, timeout=1)


def continuation_fixture(tmp_path):
    parent, out = tmp_path / "parent", tmp_path / "next"
    registered = {"allowed_gpus": [2, 3], "max_gpus": 2, "source_hashes": source_hashes(),
                  "created_at": "original", "modes": ["matched_eval", "matched_train"],
                  "conditions": ["shared_c1", "shared_c4"], "precision": "FP32",
                  "datasets": {"voc": {"runs": [{"seed": 1234, "condition": "ce", "checkpoints": {"20000": "/state"}}],
                                       "input_sha256": {"/state": "checkpoint"}, "matched_batch": 16}}}
    write_json(parent / "registration.json", registered)
    reg_hash = sha256(parent / "registration.json")
    for mode in registered["modes"]:
        for condition in registered["conditions"]:
            write_json(parent / "one_step/voc/ce/1234/20000" / mode / (condition + ".json"),
                       {"dataset": "voc", "seed": 1234, "checkpoint": 20000, "cohort": "ce", "mode": mode,
                        "condition": condition, "horizon": 1, "checkpoint_sha256": "checkpoint",
                        "registration_sha256": reg_hash, "status": "DIVERGED" if mode == "matched_eval" else "PASS",
                        "state_restored": True})
    write_json(parent / "one_step/voc/summary_ce_seed1234_stage20000.json",
               {"registration_sha256": reg_hash, "status": "PASS", "horizon": 1, "branches": 4})
    new = amend(parent, out, [0, 1, 2, 3], {"shared_gpus": {"1": {"minimum_free_mib": 28672}}})
    return parent, out, registered, new


def test_resource_amendment_preserves_observations_and_reuses_complete_cases(tmp_path):
    parent, out, old, new = continuation_fixture(tmp_path)
    assert scientific_contract(old) == scientific_contract(new)
    assert new["allowed_gpus"] == [0, 1, 2, 3] and new["max_gpus"] == 4
    for source in (parent / "one_step").rglob("*.json"):
        assert source.read_bytes() == (out / source.relative_to(parent)).read_bytes()
    assert completed_job(out, "one_step", "voc", 1234, 20000, "ce", new, sha256(out / "registration.json"))
    assert not completed_job(out, "one_step", "voc", 2025, 20000, "ce", new, sha256(out / "registration.json"))
    with pytest.raises(FileExistsError):
        amend(parent, out, [0, 1, 2, 3], {})


def test_continuation_rejects_changed_batch_and_tampered_inherited_record(tmp_path):
    _, out, _, new = continuation_fixture(tmp_path)
    changed = copy.deepcopy(new)
    changed["datasets"]["voc"]["matched_batch"] = 8
    with pytest.raises(RuntimeError, match="scientific contract"):
        validate_continuation(changed, out)
    target = out / "one_step/voc/ce/1234/20000/matched_train/shared_c1.json"
    row = read_json(target)
    row["status"] = "DIVERGED"
    write_json(target, row)
    with pytest.raises(RuntimeError, match="Inherited record changed"):
        validate_continuation(new, out)
    assert not accepts_record(target, row, new, sha256(out / "registration.json"), out)


def test_parent_registration_hash_alone_does_not_accept_an_unattested_observation(tmp_path):
    _, out, _, new = continuation_fixture(tmp_path)
    path = out / "unattested.json"
    row = {"registration_sha256": new["continuation"]["parent_registration_sha256"]}
    write_json(path, row)
    assert not accepts_record(path, row, new, sha256(out / "registration.json"), out)


def test_resource_amendment_rejects_a_changed_numerical_core(tmp_path):
    parent, _, _, _ = continuation_fixture(tmp_path)
    row = read_json(parent / "registration.json")
    row["source_hashes"]["core.py"] = "different"
    write_json(parent / "registration.json", row)
    with pytest.raises(RuntimeError, match="Protected numerical"):
        amend(parent, tmp_path / "disallowed", [0, 1, 2, 3], {})


def test_migration_remaps_paths_but_preserves_checkpoint_args_and_paired_inputs(tmp_path):
    stages = [20000, 40000, 60000]
    original_args = {"teacher_pretrained": "/old/teacher.pth", "data": "/old/data", "seed": 1234}
    paths = {str(k): f"/old/checkpoints/{k}.pth" for k in stages}
    inputs = {"/old/teacher.pth": "teacher", "/old/runtime-large/train_kd_3090.py": "runtime",
              "/old/runtime-large/dataset/list/voc/train_aug.txt": "split", **{p: str(k) for k, p in paths.items()}}
    updates = paired_updates(list(range(256, 600)), 16, 1234, 20000)
    old = {"runtime": "/old/runtime-large", "data": "/old/data", "teacher": "/old/teacher.pth",
           "train_list": "/old/runtime-large/dataset/list/voc/train_aug.txt",
           "val_list": "/old/runtime-large/dataset/list/voc/val.txt",
           "runs": [{"args": original_args, "checkpoints": paths}],
           "monitor": [{"img": "/old/data/JPEGImages/a.jpg", "label": "/old/data/SegmentationClassAug/a.png"}],
           "updates": updates, "input_sha256": inputs,
           "input_hash_provenance": {p: "original" for p in inputs}}
    untouched = copy.deepcopy(old)
    transfers = {p: {"sha256": str(k), "relative": f"checkpoints/voc/ce/1234/{k}.pth"} for k, p in paths.items()}
    mapped = dataset_paths(old, "voc", tmp_path, transfers)
    assert old == untouched
    assert mapped["runs"][0]["args"] == original_args
    assert mapped["updates"] == updates
    assert mapped["runs"][0]["origin_checkpoints"] == paths
    assert all(str(tmp_path) in p for p in mapped["input_sha256"])
    assert set(mapped["input_origins"].values()) == set(inputs)
    assert mapped["monitor"][0]["img"] == str(tmp_path / "shared/dataset/VOCAug/JPEGImages/a.jpg")
    transfers[paths["20000"]]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="manifest"):
        dataset_paths(old, "voc", tmp_path, transfers)
