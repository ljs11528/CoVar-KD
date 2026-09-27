from collections import Counter

import pytest
import torch
import torch.nn.functional as F

from scripts.experiments.covar_match import run_p10_h20 as queue
from scripts.diagnostics.summarize_p10_h20 import analyze
from utils.rtc_temperature import masked_temperature_kd_loss
from utils.teacher_only_kd import teacher_target_kd_loss


def test_registered_matrix_has_only_authorized_fresh_runs_and_priority():
    specs = queue.experiment_plan()
    assert len(specs) == len({s['variant'] for s in specs}) == 33
    assert Counter(s['phase'] for s in specs) == {1: 21, 2: 3, 3: 9}
    assert set(Counter(s['group'] for s in specs).values()) == {3}
    assert len(queue.smoke_plan()) == 7
    for spec in specs:
        command, root, save, log = queue.command_for(spec)
        value = lambda flag: command[command.index(flag) + 1]
        assert value('--batch-size') == '16' and value('--max-iterations') == '80000'
        assert value('--kd-loss-mode') == spec['mode']
        assert value('--student-backbone') == 'mobilenetv3_' + spec['student']
        assert '--resume' not in command and '--skip-val' not in command
        assert str(save).startswith(str(queue.RUN_ROOT)) and str(log).startswith(str(queue.RUN_ROOT))
        if spec['mode'] == 'masked':
            assert value('--covar-kd-temp-power') == '0.0'
    with pytest.raises(ValueError, match='outside'):
        queue.command_for(dict(specs[0], temperature='8.0'))


@pytest.mark.parametrize('student_temperature,coefficient', [(1., 1.), (1., 4.), (2., 1.), (2., 4.)])
def test_factorial_loss_and_gradient_have_exactly_one_scaling_factor(student_temperature, coefficient):
    generator = torch.Generator().manual_seed(81)
    student = torch.randn(2, 3, 2, 3, dtype=torch.float64, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 3, 2, 3, dtype=torch.float64, generator=generator)
    native_valid = torch.tensor([[[1, 0, 1], [1, 1, 0]], [[0, 1, 1], [1, 0, 1]]], dtype=torch.bool)
    # Match the trainer's nearest-neighbor projection from label to logits grid.
    high_valid = native_valid.repeat_interleave(2, 1).repeat_interleave(2, 2)
    if student_temperature == 1:
        loss = coefficient * teacher_target_kd_loss(student, teacher, 2., high_valid)
    else:
        valid = F.interpolate(high_valid.float().unsqueeze(1), size=student.shape[-2:], mode='nearest').squeeze(1) > .5
        loss = coefficient * masked_temperature_kd_loss(student, teacher, torch.full_like(valid, 2., dtype=torch.float64), valid, temperature_power=0.)
    target = F.softmax(teacher / 2., dim=1)
    log_student = F.log_softmax(student / student_temperature, dim=1)
    expected = coefficient * (target * (target.log() - log_student)).sum(1)[native_valid].mean()
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    gradient = torch.autograd.grad(loss, student)[0]
    analytic = coefficient / student_temperature * (log_student.exp() - target) * native_valid.unsqueeze(1) / native_valid.sum()
    torch.testing.assert_close(gradient, analytic, rtol=1e-12, atol=1e-12)
    if student_temperature == 2 and coefficient == 4:
        standard = masked_temperature_kd_loss(student, teacher, torch.full_like(native_valid, 2., dtype=torch.float64), native_valid, temperature_power=2.)
        torch.testing.assert_close(loss, standard, rtol=1e-12, atol=1e-12)


def test_shared_t1_equals_existing_teacher_only_value_and_gradient():
    student = torch.tensor([[[[.1, .3]], [[.2, -.1]], [[.7, .2]]]], requires_grad=True)
    teacher = student.detach().flip(1)
    valid = torch.tensor([[[True, False]]])
    old = teacher_target_kd_loss(student, teacher, 1., valid)
    shared = masked_temperature_kd_loss(student, teacher, torch.ones_like(valid, dtype=torch.float), valid, 0.)
    torch.testing.assert_close(old, shared)
    torch.testing.assert_close(torch.autograd.grad(old, student, retain_graph=True)[0], torch.autograd.grad(shared, student)[0])


@pytest.mark.parametrize('memory,utilization,expected', [(0, 0, True), (60000, 0, False), (0, 100, False), (1024, 0, False)])
def test_idle_detection_does_not_treat_low_utilization_resident_jobs_as_free(memory, utilization, expected):
    assert queue.idle(dict(memory_mib=memory, utilization_percent=utilization)) is expected


def test_worker_never_starts_on_busy_gpu(monkeypatch):
    monkeypatch.setattr(queue, 'setup_environment', lambda gpu: None)
    monkeypatch.setattr(queue, 'gpu_inventory', lambda: {6: dict(memory_mib=60000, utilization_percent=0)})
    monkeypatch.setattr(queue.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='became busy'):
        queue.worker(queue.experiment_plan()[0], 6, False)


def synthetic_rows():
    rows = [dict(s, final_miou_percent=61. + s['lambda_kd'] * .1) for s in queue.experiment_plan()]
    for t in queue.TEMPERATURES:
        for seed in queue.SEEDS:
            rows.append(dict(group=f"large_teacher_T{t.replace('.', 'p')}_L1", seed=seed,
                             final_miou_percent=62. if t == '2.0' else 61.))
    return rows


def test_boundary_stays_separate_from_common_grid_and_never_extends_automatically():
    rows = synthetic_rows()
    for row in rows:
        if row['group'] == 'large_teacher_T4p0_L1':
            row['final_miou_percent'] = 63.
    result = analyze(rows)
    assert result['common_grid_response']['large']['mean_winner'] == '2.0'
    assert '4.0' not in result['common_grid_response']['large']['means']
    assert result['boundary_extension']['upper_boundary_unresolved'] is True
    assert result['boundary_extension']['automatic_further_runs'] is False
    assert result['kd_minus_ce_pp']['large']['2.0']['mean'] == 1.
    assert result['leave_one_seed_out']['large']['regret_pp']['mean'] == 0.
    assert result['loss_controls'] is not None


def test_partial_groups_do_not_produce_three_seed_claims_and_duplicates_fail():
    rows = synthetic_rows()
    partial = [r for r in rows if not (r['group'] == 'small_ce' and r['seed'] == 3407)]
    result = analyze(partial)
    assert 'small_ce' not in result['group_statistics']
    assert result['kd_minus_ce_pp']['small']['1.0'] is None
    with pytest.raises(ValueError, match='duplicate'):
        analyze(rows + [rows[0]])


def test_existing_partial_output_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(queue, 'RUN_ROOT', tmp_path)
    monkeypatch.setattr(queue, 'setup_environment', lambda gpu: None)
    monkeypatch.setattr(queue, 'gpu_inventory', lambda: {6: dict(memory_mib=0, utilization_percent=0)})
    spec = queue.experiment_plan()[0]
    log_dir = queue.command_for(spec)[3]
    log_dir.mkdir(parents=True)
    (log_dir / 'evidence.txt').write_text('partial evidence')
    monkeypatch.setattr(queue.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(RuntimeError, match='existing output'):
        queue.worker(spec, 6, False)
    assert (log_dir / 'evidence.txt').read_text() == 'partial evidence'
