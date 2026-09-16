import pytest

from scripts.experiments.covar_match import run_p9_h20 as h20


def test_seed_gpu_assignment_is_distinct_and_fixed():
    assert h20.assignments([4, 5, 6]) == [(1234, 4), (2025, 5), (3407, 6)]
    assert h20.assignments([6, 7]) == [(1234, 6), (2025, 7), (3407, 6)]
    for invalid in ([4, 4, 6], [], [0, 1, 2, 3], [-1, 5, 6]):
        with pytest.raises(ValueError):
            h20.assignments(invalid)


def test_stage_waits_for_all_seeds_and_surfaces_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(h20, "RUN_ROOT", tmp_path)
    (tmp_path / "runtime").mkdir()
    calls, waited = [], []

    class Child:
        def __init__(self, seed):
            self.seed = seed

        def wait(self):
            assert len(calls) == 3
            waited.append(self.seed)
            return 1 if self.seed == 2025 else 0

    def launch(command, **kwargs):
        seed = int(command[command.index("--worker-seed") + 1])
        calls.append(command)
        return Child(seed)

    monkeypatch.setattr(h20.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="workers failed"):
        h20.run_stage([4, 5, 6], 1, False)
    assert waited == [1234, 2025, 3407]
    assert [c[c.index("--gpu") + 1] for c in calls] == ["4", "5", "6"]
    assert all(c[c.index("--stage") + 1] == "1" for c in calls)


def test_two_gpu_waves_never_overlap_seeds_on_one_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(h20, "RUN_ROOT", tmp_path)
    (tmp_path / "runtime").mkdir()
    active, started = set(), []

    class Child:
        def __init__(self, gpu):
            self.gpu = gpu

        def wait(self):
            active.remove(self.gpu)
            return 0

    def launch(command, **kwargs):
        gpu = int(command[command.index("--gpu") + 1])
        assert gpu not in active
        active.add(gpu)
        started.append((int(command[command.index("--worker-seed") + 1]), gpu))
        return Child(gpu)

    monkeypatch.setattr(h20.subprocess, "Popen", launch)
    h20.run_stage([6, 7], 1, False)
    assert started == [(1234, 6), (2025, 7), (3407, 6)]
    assert not active


def test_worker_preserves_existing_state(monkeypatch, tmp_path):
    monkeypatch.setattr(h20, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(h20, "setup", lambda gpu: None)
    (tmp_path / "runtime").mkdir()
    path = tmp_path / "runtime/seed1234-stage1.json"
    path.write_text("existing evidence")
    with pytest.raises(RuntimeError, match="existing worker state"):
        h20.worker(1234, 4, 1, False)
    assert path.read_text() == "existing evidence"
