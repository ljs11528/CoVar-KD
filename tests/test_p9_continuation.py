import json
from pathlib import Path

import pytest

from scripts.experiments.covar_match import continue_p9_single_gpu as continuation


def test_fresh_training_command_only_changes_output_paths(monkeypatch, tmp_path):
    queue = continuation.queue
    original = queue.command_for(1234, "kd", "0.25")[0]
    for name in ("RUN_ROOT", "P9_ROOT", "REPORT_ROOT"):
        monkeypatch.setattr(queue, name, getattr(queue, name))
        monkeypatch.setattr(continuation, name, tmp_path / name)
    continuation.configure_paths()
    command = queue.command_for(1234, "kd", "0.25")[0]
    for option in ("--save-dir", "--log-dir"):
        index = original.index(option) + 1
        assert command[index] != original[index]
        command[index] = original[index]
    assert command == original
    assert "--resume" not in command


@pytest.mark.parametrize("required,expected_count", [(False, 15), (True, 21)])
def test_grid_preserves_stage1_and_only_adds_required_temperatures(
    monkeypatch, tmp_path, required, expected_count
):
    queue = continuation.queue
    monkeypatch.setattr(continuation, "REPORT_ROOT", tmp_path)
    calls = []
    reports = []

    def execute(seed, kind, temperature, state, state_path):
        calls.append((seed, kind, temperature))

    def generate(stage):
        reports.append(stage)
        payload = {"stage1": {"phase2_gate": {"required": required}}, "generation": len(reports)}
        (tmp_path / "P9_temperature.json").write_text(json.dumps(payload))
        for suffix in (".md", ".csv"):
            (tmp_path / ("P9_temperature" + suffix)).write_text(str(len(reports)))

    monkeypatch.setattr(queue, "execute_run", execute)
    monkeypatch.setattr(queue, "generate_reports", generate)
    state = {"completed_runs": []}
    continuation.run_approved_plan(state, tmp_path / "state.json")
    stage1 = [(seed, "kd", temperature) for seed in queue.SEEDS
              for temperature in queue.P9_STAGE1_TEMPERATURES]
    stage2 = [(seed, "kd", temperature) for seed in queue.SEEDS
              for temperature in queue.P9_STAGE2_TEMPERATURES]
    assert calls == stage1 + (stage2 if required else [])
    assert len(state["completed_runs"]) == expected_count
    assert len(set(state["completed_runs"])) == expected_count
    assert state["stage2_required"] is required
    assert reports == (["p9", "p9"] if required else ["p9"])
    assert json.loads((tmp_path / "P9_stage1.json").read_text())["generation"] == 1
    assert (tmp_path / "P9_stage1.csv").read_text() == "1"


def test_changed_training_source_blocks_continuation(monkeypatch, tmp_path):
    queue = continuation.queue
    reference = tmp_path / "reference"
    reference.mkdir()
    source = tmp_path / "train_kd.py"
    source.write_text("original")
    protocol = {"sha256": {"train_kd.py": queue.digest(source)}}
    (reference / "protocol.json").write_text(json.dumps(protocol))
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    monkeypatch.setattr(continuation, "REFERENCE_REPORT", reference)
    monkeypatch.setattr(queue, "read_run", lambda *args: {"contract_pass": True})
    assert continuation.verify_reference() == protocol
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="locked source or weight differs"):
        continuation.verify_reference()


def test_changed_runtime_blocks_training():
    reference = {key: "same" for key in continuation.RUNTIME_FIELDS}
    continuation.verify_runtime(reference, dict(reference))
    changed = dict(reference, torch="different")
    with pytest.raises(RuntimeError, match="runtime differs"):
        continuation.verify_runtime(reference, changed)


def test_interrupted_output_is_preserved_without_starting_training(monkeypatch, tmp_path):
    queue = continuation.queue
    monkeypatch.setattr(queue, "P9_ROOT", tmp_path)
    command, root, variant, save_dir, log_dir, student = queue.command_for(1234, "kd", "0.25")
    log_dir.mkdir(parents=True)
    evidence = log_dir / "interrupted.txt"
    evidence.write_text("preserve this attempt")

    def incomplete(*args):
        raise RuntimeError("incomplete run")

    monkeypatch.setattr(queue, "read_run", incomplete)
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: pytest.fail("started training"))
    with pytest.raises(RuntimeError, match="partial output exists"):
        queue.execute_run(1234, "kd", "0.25", {"completed_runs": []}, tmp_path / "state.json")
    assert evidence.read_text() == "preserve this attempt"
    assert not save_dir.exists()
