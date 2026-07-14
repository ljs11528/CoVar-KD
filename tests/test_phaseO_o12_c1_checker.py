import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = (
    ROOT
    / "scripts/experiments/kd_baselines_npu/check_phaseO_o12_c1_run.py"
)
SPEC = importlib.util.spec_from_file_location("phaseO_o12_c1_checker", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def validation_log(*, missing_step=None, duplicate_step=None, include_overall=True):
    lines = []
    for step in CHECKER.EXPECTED_VALIDATION_STEPS:
        if step == missing_step:
            continue
        repeats = 2 if step == duplicate_step else 1
        for _ in range(repeats):
            lines.append(f"Iters: {step}/20000")
            lines.append("Start validation, Total sample: 1449")
            final_miou = 0.20 + step / 100_000.0
            for sample in range(1, CHECKER.VALIDATION_SAMPLES + 1):
                miou = final_miou if sample == CHECKER.VALIDATION_SAMPLES else 0.1
                lines.append(
                    f"Sample: {sample}, Validation pixAcc: 0.800000, "
                    f"mIoU: {miou:.6f}"
                )
            if include_overall:
                lines.append("Overall validation pixAcc: 99.0, mIoU: 0.999999")
    return "\n".join(lines)


def make_state(variant="neutral"):
    save_dir = Path("/tmp/c1-save").resolve()
    log_dir = Path("/tmp/c1-log").resolve()
    return {
        "checkpoint_type": "train_kd_training_state",
        "checkpoint_version": 4,
        "student": {},
        "criterion_cwd": {},
        "criterion_fitnet": {},
        "D": {},
        "optimizer": {},
        "D_optimizer": {},
        "iteration": 20_000,
        "best_pred": 0.4,
        "rng_state": {},
        "rng_state_by_rank": [{}],
        "world_size": 1,
        "args": CHECKER.expected_saved_args(variant, save_dir, log_dir),
        "rtc": None,
        "rtc_o12": CHECKER.expected_rtc_o12(variant),
        "rtc_o12_data_order": {
            "contract": copy.deepcopy(CHECKER.EXPECTED_ORDER),
            "completed_iteration": 20_000,
            "next_global_iteration": None,
            "next_canonical_index_offset": 320_000,
        },
    }, save_dir, log_dir


def test_validation_parser_uses_sample_1449_and_fixed_last10():
    errors = []
    summary = CHECKER.parse_validation_blocks(validation_log(), errors)
    assert errors == []
    assert len(summary["blocks"]) == 25
    assert summary["final_step"] == 20_000
    assert summary["final_mIoU"] == 0.4
    assert summary["best_mIoU"] == 0.4
    assert [row["step"] for row in summary["last10"]] == list(
        range(12_800, 20_001, 800)
    )
    assert summary["last10_mean_mIoU"] == pytest.approx(0.364)
    assert summary["overall_records_ignored"] == 25
    assert summary["final_mIoU"] != 0.999999


def test_validation_parser_rejects_missing_and_incomplete_blocks():
    errors = []
    CHECKER.parse_validation_blocks(validation_log(missing_step=8_000), errors)
    assert any("missing, duplicated, or out of order" in error for error in errors)
    assert any("exactly 25" in error for error in errors)

    text = validation_log(include_overall=False)
    text = text.replace(
        "Sample: 1449, Validation pixAcc: 0.800000, mIoU: 0.400000",
        "",
    )
    errors = []
    CHECKER.parse_validation_blocks(text, errors)
    assert any("ended at sample 1448" in error for error in errors)


def test_validation_parser_rejects_duplicate_block():
    errors = []
    CHECKER.parse_validation_blocks(validation_log(duplicate_step=8_000), errors)
    assert any("missing, duplicated, or out of order" in error for error in errors)
    assert any("exactly 25" in error for error in errors)


def test_checkpoint_order_and_metadata_are_fail_closed():
    state, save_dir, log_dir = make_state("neutral")
    validation = {"best_mIoU": 0.4}
    errors = []
    CHECKER.validate_order_and_metadata(
        state, "neutral", save_dir, log_dir, errors, validation
    )
    assert errors == []

    bad = copy.deepcopy(state)
    bad["rtc_o12_data_order"]["contract"]["complete_order_sha256"] = "0" * 64
    bad["rtc_o12"]["variant"] = "unreliable_only"
    bad["iteration"] = 19_999
    errors = []
    CHECKER.validate_order_and_metadata(
        bad, "neutral", save_dir, log_dir, errors, validation
    )
    assert any("iteration must be 20000" in error for error in errors)
    assert any("checkpoint.rtc_o12.variant mismatch" in error for error in errors)
    assert any("complete_order_sha256 mismatch" in error for error in errors)


def test_log_anomalies_and_disabled_loss_are_rejected():
    errors = []
    anomalies = CHECKER.find_log_anomalies(
        "Traceback (most recent call last)\nKD Loss: nan\nvalue=+Inf", errors
    )
    assert anomalies == ["traceback", "nonfinite_tokens"]
    assert len(errors) == 2

    line = (
        "Iters: 20/20000 || Lr: 0.019982 || Task Loss: 1.0 "
        "|| KD Loss: 0.5 || Adv_G Loss: 0.1 || Adv_D Loss: 0.1 "
        "|| skd_loss: 0.1 || cwd_fea_loss: 2.0 || cwd_logit_loss: 2.0 "
        "|| ifv_loss: 0.0 || at_loss: 0.0 || fitnet_loss: 0.0 "
        "|| psd_loss: 0.0 || csd_loss: 0.0 || Cost Time: 0:00:01 "
        "|| Estimated Time: 1:00:00 || O1.2 variant: neutral "
        "|| O1.2 branch KL mean: 0.50000000 "
        "|| O1.2 cross-entropy mean: 1.50000000 "
        "|| O1.2 teacher entropy mean: 1.00000000 "
        "|| O1.2 KD-only student-logit grad L2: 0.00100000 "
        "|| O1.2 valid pixels: 100 || Teacher output T: 3.0000"
    )
    synthetic = "\n".join(
        line.replace("Iters: 20/20000", f"Iters: {step}/20000")
        for step in CHECKER.EXPECTED_TRAINING_ITERATIONS
    )
    errors = []
    CHECKER.parse_training_diagnostics(synthetic, "neutral", errors)
    assert any("disabled skd is nonzero" in error for error in errors)
