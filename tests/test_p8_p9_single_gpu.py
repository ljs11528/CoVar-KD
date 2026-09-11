from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from scripts.diagnostics.summarize_p8_p9_experiments import (
    MILESTONES, P7_TEMPERATURES, P9_STAGE1_TEMPERATURES, SEEDS,
    VALIDATION_RE, namespace_checks, p8_variant, summarize_p8, summarize_p9,
)
from scripts.experiments.covar_match.run_p8_p9_single_gpu import command_for


def namespace(**overrides):
    values = dict(
        teacher_model="deeplabv3", teacher_backbone="resnet101",
        student_model="deeplabv3_mobilenet_ssseg", student_backbone="mobilenetv3_small",
        dataset="voc", crop_size=[512, 512], workers=4, ignore_label=-1,
        aux=False, batch_size=16, max_iterations=80000, lr=0.02,
        momentum=0.9, weight_decay=0.0001, kd_loss_mode="teacher_only",
        lambda_kd=0.0, kd_temperature=1.0, teacher_output_temp=1.0,
        use_covar=False, seed=1234, resume=None, skip_val=False,
        save_per_iters=20000, val_per_iters=20000,
        keep_checkpoint_iters=list(MILESTONES), num_gpus=1, distributed=False,
        teacher_pretrained="data/deeplabv3_resnet101_voc_best_model.pth",
        student_pretrained_base="data/mobilenet_v3_small-47085aa1.pth",
    )
    for name in ("adv", "d", "skd", "cwd_fea", "cwd_logit", "ifv", "fitnet", "at", "psd", "csd"):
        values["lambda_" + name] = 0.0
    values.update(overrides)
    return repr(Namespace(**values))


def checks(text):
    return namespace_checks(text, seed=1234, student_backbone="mobilenetv3_small",
                            lambda_kd=0.0, temperature="1.0", execution_protocol="single_gpu")


@pytest.mark.parametrize("field,value", [
    ("num_gpus", 2), ("distributed", True), ("batch_size", 8),
    ("lambda_cwd_logit", 1.0), ("aux", True), ("resume", "checkpoint.pth"),
    ("max_iterations", 40000), ("lr", 0.1), ("workers", 8),
])
def test_report_rejects_wrong_training_protocol(field, value):
    assert all(checks(namespace()).values())
    assert checks(namespace(**{field: value}))[field] is False


def run(value):
    return {
        "final_miou_percent": value,
        "trajectory": {str(step): {"miou_percent": value, "pixacc_percent": 90.0}
                       for step in MILESTONES},
    }


def test_single_gpu_ce_is_not_labeled_as_a_controlled_p7_comparison():
    ce = {seed: run(61.0) for seed in SEEDS}
    rows = {seed: {"runs": {temperature: run(62.0) for temperature in P7_TEMPERATURES}}
            for seed in SEEDS}
    payload = summarize_p8({"selection_summary": {}}, rows, ce, "single_gpu")
    assert payload["protocol"]["p7_locked"] is False
    assert payload["protocol"]["legacy_two_gpu_runs_pooled"] is False
    assert "paired_kd_minus_ce_pp" not in payload
    assert payload["same_seed_cross_protocol_kd_minus_ce_pp"]["1.0"]["mean"] == 1.0
    assert payload["conclusions"]["controlled_kd_effect_identifiable"] is False
    assert p8_variant(2025, "single_gpu") == "ce_only_80k_seed2025"


def test_cross_pair_overlap_does_not_claim_capacity_or_transfer_evidence():
    values = {"0.25": 60.0, "0.5": 60.5, "1.0": 62.0, "1.5": 61.95, "2.0": 60.0}
    grid = {seed: {temperature: run(values[temperature])
                   for temperature in P9_STAGE1_TEMPERATURES} for seed in SEEDS}
    p7 = {"selection_summary": {"best_mean_temperature": "1.5",
                                "delta_optimal_grid_set": ["0.5", "1.5"]}}
    covar = {"rows": {temperature: {"r_c_mean": float(temperature),
                                    "r_v_mean": float(temperature),
                                    "r_mean": 2 * float(temperature)}
                       for temperature in P7_TEMPERATURES}}
    payload = summarize_p9(p7, covar, grid, None, "absent", 0.2, "single_gpu")
    cross = payload["cross_pair_comparison"]
    assert cross["different_winners_with_covar_overlap"] is True
    assert cross["capacity_effect_identifiable_against_p7"] is False
    assert cross["transfer_rule_validated"] is False
    assert payload["protocol"]["only_student_capacity_changed_from_P7"] is False
    assert payload["phase2_status"] == "required_pending"


def test_queue_rejects_unapproved_temperatures_and_keeps_ce_loss_zero():
    with pytest.raises(ValueError):
        command_for(1234, "kd", "3.0")
    command, _, _, _, _, _ = command_for(2025, "ce", "1.0")
    assert command[command.index("--lambda-kd") + 1] == "0.0"
    assert command[command.index("--max-iterations") + 1] == "80000"
    assert "--resume" not in command
    assert "--skip-val" not in command


@pytest.mark.parametrize("world_size", [1, 2])
def test_validation_emits_one_overall_summary_per_world_size(monkeypatch, world_size):
    import train_kd
    from utils.score import SegmentationMetric

    target = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.nn.functional.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float() * 20

    class Model(torch.nn.Module):
        def forward(self, images):
            return [logits]

    model = Model()
    messages = []
    monkeypatch.setattr(train_kd, "logger", SimpleNamespace(info=messages.append), raising=False)
    monkeypatch.setattr(train_kd, "synchronize", lambda: None)
    trainer = SimpleNamespace(
        args=SimpleNamespace(distributed=world_size > 1),
        metric=SegmentationMetric(2), device=torch.device("cpu"),
        s_model=SimpleNamespace(module=model) if world_size > 1 else model,
        val_loader=[(torch.zeros(1, 3, 2, 2), target, ["synthetic"])],
        num_gpus=world_size, best_pred=0.0, reduce_tensor=lambda value: value,
        save_checkpoint=lambda **kwargs: None,
    )
    train_kd.Trainer.validation(trainer, step=20000)
    summaries = [VALIDATION_RE.search(message) for message in messages]
    summaries = [match for match in summaries if match]
    assert len(summaries) == 1
    assert float(summaries[0].group("miou")) == pytest.approx(100.0)
    assert float(summaries[0].group("pixacc")) == pytest.approx(100.0)
