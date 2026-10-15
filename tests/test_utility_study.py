import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from utility_study.core import (
    PooledMetrics, Snapshot, ce_loss, choose_coefficient, evaluation_mode,
    isolated_rng, kd_loss, restored_state, rng_state, tree_equal,
)
from utility_study.register import monitoring_indices, validate_checkpoint_metadata
from utility_study.workflow import gate_from_values


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
