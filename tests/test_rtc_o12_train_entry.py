from types import SimpleNamespace

import pytest
import torch

import train_kd


def o12_args(max_iterations=20, skip_val=True):
    return SimpleNamespace(
        kd_loss_mode=train_kd.RTC_O12_KD_LOSS_MODE,
        use_covar=False,
        rtc_o12_variant='unreliable_only',
        rtc_o12_cdf_path='cdf.pt',
        rtc_o12_parameters_path='parameters.json',
        rtc_o12_gate_path='gate.json',
        dataset='voc',
        teacher_model='deeplabv3',
        teacher_backbone='resnet101',
        student_model='deeplabv3_mobilenet_ssseg',
        student_backbone='mobilenetv3_small',
        ignore_label=-1,
        batch_size=16,
        workers=8,
        seed=1234,
        log_iter=20,
        save_per_iters=800,
        val_per_iters=800,
        device_type='npu',
        start_epoch=0,
        local_rank=0,
        teacher_pretrained_base='None',
        student_pretrained='None',
        crop_size=[512, 512],
        max_iterations=max_iterations,
        no_cuda=False,
        skip_val=skip_val,
        teacher_pretrained='teacher.pth',
        student_pretrained_base='student.pth',
        teacher_output_temp=3.0,
        kd_temperature=1.0,
        lambda_kd=1.0,
        lambda_adv=0.001,
        lambda_d=0.1,
        lambda_cwd_fea=50.0,
        lambda_cwd_logit=3.0,
        lr=0.02,
        momentum=0.9,
        weight_decay=1e-4,
        lambda_skd=0.0,
        lambda_ifv=0.0,
        lambda_fitnet=0.0,
        lambda_at=0.0,
        lambda_psd=0.0,
        lambda_csd=0.0,
        rtc_enable_reliable=None,
        rtc_enable_unreliable=None,
        rtc_shuffle=None,
        rtc_reverse_routing=None,
    )


def artifact_contract():
    return {
        'b': 0.3,
        'branch_scalar_temperatures': {
            'unreliable_only': {
                'arithmetic': 1.08,
                'harmonic': 1.04,
            },
            'full_budgeted': {
                'arithmetic': 0.995,
                'harmonic': 0.98,
            },
        },
    }


def test_legacy_modes_do_not_require_o12_attributes():
    train_kd.validate_rtc_o12_cli_contract(
        SimpleNamespace(kd_loss_mode='legacy')
    )
    train_kd.validate_rtc_o12_cli_contract(
        SimpleNamespace(kd_loss_mode='masked')
    )


def test_cli_contract_accepts_only_registered_smoke_and_20k_validation():
    train_kd.validate_rtc_o12_cli_contract(o12_args(), world_size=1)
    train_kd.validate_rtc_o12_cli_contract(
        o12_args(max_iterations=20000, skip_val=False), world_size=1
    )
    with pytest.raises(ValueError, match='skip-val'):
        train_kd.validate_rtc_o12_cli_contract(
            o12_args(max_iterations=20, skip_val=False), world_size=1
        )
    with pytest.raises(ValueError, match='skip-val'):
        train_kd.validate_rtc_o12_cli_contract(
            o12_args(max_iterations=20000, skip_val=True), world_size=1
        )


@pytest.mark.parametrize(
    'field,value,fragment',
    [
        ('use_covar', True, 'forbids --use-covar'),
        ('teacher_output_temp', 1.0, 'teacher-output-temp'),
        ('kd_temperature', 2.0, 'kd-temperature'),
        ('device_type', 'cpu', 'device-type'),
        ('max_iterations', 21, 'max-iterations'),
    ],
)
def test_cli_contract_rejects_recipe_drift(field, value, fragment):
    args = o12_args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match=fragment):
        train_kd.validate_rtc_o12_cli_contract(args, world_size=1)


def test_exact_logit_shape_contract_forbids_interpolation():
    student = torch.randn(2, 21, 8, 8)
    teacher = torch.randn(2, 21, 8, 8)
    train_kd.validate_rtc_o12_logit_shapes(student, teacher)
    with pytest.raises(ValueError, match='forbids implicit'):
        train_kd.validate_rtc_o12_logit_shapes(
            student, torch.randn(2, 21, 7, 8)
        )
    with pytest.raises(ValueError, match='shape'):
        train_kd.validate_rtc_o12_logit_shapes(
            student, torch.randn(2, 20, 8, 8)
        )


def test_canonical_dataset_index_ignores_blank_lines_and_rejects_duplicates(tmp_path):
    path = tmp_path / 'train.txt'
    path.write_text('a\n\n b \n', encoding='utf-8')
    index = train_kd.build_rtc_o12_dataset_index(path)
    assert index == {'a': 0, 'b': 1}
    assert train_kd.resolve_rtc_o12_dataset_indices(['b', 'a'], index) == [1, 0]
    with pytest.raises(ValueError, match='cannot be mapped'):
        train_kd.resolve_rtc_o12_dataset_indices(['missing'], index)

    path.write_text('a\na\n', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        train_kd.build_rtc_o12_dataset_index(path)


def test_all_spatial_branches_obey_registered_direction_and_bounds():
    config = train_kd.O12CalibrationConfig()
    quantile = torch.tensor(
        [[[0.0, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0]]],
        dtype=torch.float32,
    )
    valid = torch.ones_like(quantile, dtype=torch.bool)
    contract = artifact_contract()
    maps = {}
    for variant in ('neutral', 'reliable_only', 'unreliable_only', 'full_budgeted'):
        maps[variant], _ = train_kd.build_rtc_o12_temperature_for_variant(
            quantile, valid, variant, contract, config
        )

    assert torch.equal(maps['neutral'], torch.ones_like(quantile))
    assert bool((maps['reliable_only'] <= 1.0).all())
    assert bool((maps['unreliable_only'] >= 1.0).all())
    assert maps['full_budgeted'][0, 0, 0] == pytest.approx(0.9)
    assert maps['full_budgeted'][0, 0, -1] == pytest.approx(
        torch.exp(torch.tensor(0.3)).item()
    )
    assert torch.equal(
        maps['full_budgeted'][0, 0, 2:5], torch.ones(3)
    )


def test_scalar_and_shuffle_controls_use_frozen_branch_specific_values():
    config = train_kd.O12CalibrationConfig()
    quantile = torch.tensor(
        [[[0.1, 0.5, 0.85, 1.0], [0.2, 0.7, 0.9, 0.95]]],
        dtype=torch.float32,
    )
    valid = torch.tensor(
        [[[True, True, True, False], [True, True, True, True]]]
    )
    contract = artifact_contract()
    scalar, spec = train_kd.build_rtc_o12_temperature_for_variant(
        quantile,
        valid,
        'unreliable_arithmetic_scalar',
        contract,
        config,
    )
    assert spec['branch'] == 'unreliable_only'
    assert torch.equal(
        scalar[valid], torch.full_like(scalar[valid], 1.08)
    )
    assert scalar[~valid].item() == 1.0

    spatial, _ = train_kd.build_rtc_o12_temperature_for_variant(
        quantile, valid, 'unreliable_only', contract, config
    )
    shuffled_a, _ = train_kd.build_rtc_o12_temperature_for_variant(
        quantile,
        valid,
        'unreliable_shuffled',
        contract,
        config,
        dataset_indices=[7],
        global_iteration=11,
    )
    shuffled_b, _ = train_kd.build_rtc_o12_temperature_for_variant(
        quantile,
        valid,
        'unreliable_shuffled',
        contract,
        config,
        dataset_indices=[7],
        global_iteration=11,
    )
    assert torch.equal(shuffled_a, shuffled_b)
    assert torch.equal(
        torch.sort(spatial[valid]).values,
        torch.sort(shuffled_a[valid]).values,
    )
    assert shuffled_a[~valid].item() == 1.0


def test_teacher_target_is_single_temperature_detached_and_student_is_unscaled():
    raw_teacher = torch.tensor([[[[2.0]], [[0.0]]]])
    pixel_temperature = torch.tensor([[[1.5]]])
    target = train_kd.build_o12_teacher_target(
        raw_teacher, pixel_temperature, teacher_output_temperature=3.0
    )
    expected = torch.softmax(raw_teacher / (3.0 * 1.5), dim=1)
    double_applied = torch.softmax(raw_teacher / (3.0 * 3.0 * 1.5), dim=1)
    assert torch.allclose(target, expected)
    assert not torch.allclose(target, double_applied)
    assert not target.requires_grad

    student = torch.tensor([[[[0.4]], [[-0.2]]]], requires_grad=True)
    terms = train_kd.compute_o12_masked_kl_terms(
        student, target, torch.ones(1, 1, 1, dtype=torch.bool)
    )
    loss = train_kd.normalize_o12_ddp_loss(
        terms.kl_sum, terms.valid_count, world_size=1
    )
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_frozen_cdf_source_contract_accepts_exact_and_rejects_drift():
    cdf = train_kd.load_frozen_reliability_cdf(
        train_kd.rtc_o12_canonical_paths()['cdf']
    )
    train_kd.validate_rtc_o12_cdf_contract(cdf, num_classes=21)

    metadata = dict(cdf.metadata)
    metadata['source_sha256'] = dict(metadata['source_sha256'])
    metadata['source_sha256']['rtc_temperature'] = '0' * 64
    drifted = SimpleNamespace(
        checksum_sha256=cdf.checksum_sha256,
        metadata=metadata,
    )
    with pytest.raises(ValueError, match='source mapping'):
        train_kd.validate_rtc_o12_cdf_contract(drifted, num_classes=21)


def test_artifact_loader_rejects_noncanonical_paths_before_training():
    args = o12_args()
    with pytest.raises(ValueError, match='path is not canonical'):
        train_kd.load_rtc_o12_artifact_contract(args)


def test_canonical_sampler_resume_matches_uninterrupted_sample_names():
    full_sampler = train_kd.RTCO12CanonicalBatchSampler(
        canonical_population=13,
        batch_size=4,
        num_iterations=12,
        start_iteration=0,
        seed=1234,
    )
    full_batches = list(full_sampler)
    resumed_sampler = train_kd.RTCO12CanonicalBatchSampler(
        canonical_population=13,
        batch_size=4,
        num_iterations=12,
        start_iteration=5,
        seed=1234,
    )
    resumed_batches = list(resumed_sampler)
    assert resumed_batches == full_batches[5:]
    assert resumed_sampler.order_sha256 == full_sampler.order_sha256

    canonical_names = ['sample_{:02d}'.format(i) for i in range(13)]
    full_names = [
        [canonical_names[index] for index in batch]
        for batch in full_batches[5:]
    ]
    resumed_names = [
        [canonical_names[index] for index in batch]
        for batch in resumed_batches
    ]
    assert resumed_names == full_names
    assert resumed_sampler.contract()['complete_order_sha256_scope'] == (
        'full canonical-index sequence for current max_iterations'
    )


def test_20_step_order_is_exact_prefix_of_20k_order():
    smoke = train_kd.RTCO12CanonicalBatchSampler(
        canonical_population=train_kd.RTC_O12_TRAIN_DATASET_SIZE,
        batch_size=16,
        num_iterations=20,
        seed=1234,
    )
    formal = train_kd.RTCO12CanonicalBatchSampler(
        canonical_population=train_kd.RTC_O12_TRAIN_DATASET_SIZE,
        batch_size=16,
        num_iterations=20000,
        seed=1234,
    )
    assert torch.equal(
        smoke._order,
        formal._order[:20 * 16],
    )
    assert smoke.contract()['total_canonical_indices'] == 20 * 16
    assert formal.contract()['total_canonical_indices'] == 20000 * 16


def test_resume_next_batch_uses_next_one_based_shuffle_iteration():
    completed_iteration = 5
    full_sampler = train_kd.RTCO12CanonicalBatchSampler(
        canonical_population=17,
        batch_size=4,
        num_iterations=10,
        seed=1234,
    )
    resumed_sampler = train_kd.RTCO12CanonicalBatchSampler(
        canonical_population=17,
        batch_size=4,
        num_iterations=10,
        start_iteration=completed_iteration,
        seed=1234,
    )
    uninterrupted_batch = list(full_sampler)[completed_iteration]
    resumed_batch = next(iter(resumed_sampler))
    assert resumed_batch == uninterrupted_batch
    next_global_iteration = completed_iteration + 1

    temperature = torch.arange(
        1, 13, dtype=torch.float32
    ).reshape(1, 3, 4)
    valid = torch.ones_like(temperature, dtype=torch.bool)
    resumed_shuffle = train_kd.shuffle_o12_temperature_within_images(
        temperature,
        valid,
        dataset_indices=[resumed_batch[0]],
        global_iteration=next_global_iteration,
    )
    uninterrupted_shuffle = train_kd.shuffle_o12_temperature_within_images(
        temperature,
        valid,
        dataset_indices=[uninterrupted_batch[0]],
        global_iteration=6,
    )
    wrong_iteration_shuffle = train_kd.shuffle_o12_temperature_within_images(
        temperature,
        valid,
        dataset_indices=[resumed_batch[0]],
        global_iteration=5,
    )
    assert torch.equal(resumed_shuffle, uninterrupted_shuffle)
    assert not torch.equal(resumed_shuffle, wrong_iteration_shuffle)


def checkpoint_trainer(o12_enabled):
    trainer = train_kd.Trainer.__new__(train_kd.Trainer)
    trainer.s_model = torch.nn.Linear(2, 2)
    trainer.criterion_cwd = torch.nn.Linear(2, 2)
    trainer.criterion_fitnet = torch.nn.Linear(2, 2)
    trainer.D_model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.s_model.parameters(), lr=0.1)
    trainer.D_optimizer = torch.optim.Adam(
        trainer.D_model.parameters(), lr=0.1
    )
    trainer.best_pred = 0.25
    trainer.rtc_config = None
    trainer.args = SimpleNamespace(
        rtc_o12_variant='unreliable_only',
        max_iterations=20,
        skip_val=True,
        marker='checkpoint-test',
    )
    if o12_enabled:
        trainer.rtc_o12_config = train_kd.O12CalibrationConfig()
        trainer.rtc_o12_contract = {
            **artifact_contract(),
            'cdf_sha256': train_kd.RTC_O12_CDF_SHA256,
            'parameters_sha256': '1' * 64,
            'gate_sha256': '2' * 64,
            'source_sha256': {
                'rtc_o12_calibration': '3' * 64,
                'diagnose_rtc_o12_budget': '4' * 64,
                'check_rtc_o12_gate': '5' * 64,
                'train_entry': '6' * 64,
            },
        }
        trainer.rtc_o12_data_order_contract = (
            train_kd.RTCO12CanonicalBatchSampler(
                canonical_population=13,
                batch_size=4,
                num_iterations=20,
                seed=1234,
            ).contract()
        )
    else:
        trainer.rtc_o12_config = None
        trainer.rtc_o12_contract = None
        trainer.rtc_o12_data_order_contract = None
    return trainer


def test_checkpoint_version_is_v4_only_for_o12(monkeypatch):
    monkeypatch.setattr(train_kd, 'get_world_size', lambda: 1)
    rng_states = [{'python': 'frozen'}]
    o12_trainer = checkpoint_trainer(o12_enabled=True)
    o12_state = o12_trainer._training_state_dict(20, rng_states)
    assert o12_state['checkpoint_version'] == 4
    assert o12_state['iteration'] == 20
    assert o12_state['rtc_o12']['phase'] == 'O1.2'
    assert o12_state['rtc_o12']['variant'] == 'unreliable_only'
    assert (
        o12_state['rtc_o12']['teacher_target_contract'][
            'student_temperature'
        ]
        == 1.0
    )
    assert (
        o12_state['rtc_o12']['teacher_target_contract'][
            'temperature_loss_power'
        ]
        is None
    )
    assert (
        o12_state['rtc_o12']['data_order_contract']
        == o12_trainer.rtc_o12_data_order_contract
    )
    order_state = o12_state['rtc_o12_data_order']
    assert order_state['completed_iteration'] == 20
    assert order_state['next_global_iteration'] is None
    assert order_state['next_canonical_index_offset'] == 20 * 4
    assert (
        order_state['contract']['complete_order_sha256']
        == o12_trainer.rtc_o12_data_order_contract['complete_order_sha256']
    )

    legacy_trainer = checkpoint_trainer(o12_enabled=False)
    legacy_state = legacy_trainer._training_state_dict(20, rng_states)
    assert legacy_state['checkpoint_version'] == 3
    assert 'rtc_o12' not in legacy_state


def test_o12_resume_rejects_incomplete_or_legacy_state(monkeypatch):
    monkeypatch.setattr(train_kd, 'get_world_size', lambda: 1)
    trainer = checkpoint_trainer(o12_enabled=True)
    with pytest.raises(ValueError, match='checkpoint v4 complete state'):
        trainer._load_resume_checkpoint(
            {
                'checkpoint_type': 'train_kd_training_state',
                'checkpoint_version': 3,
            },
            'legacy.pth',
        )
