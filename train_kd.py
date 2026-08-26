import argparse
import time
import datetime
import os
import shutil
import math
import json
import hashlib
import sys
import random
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = True
except Exception:
    HAS_TORCH_NPU = False

from PCOS import get_max_confidence_and_residual_variance

from losses import *
from models.model_zoo import get_segmentation_model

from utils.sagan import Discriminator
from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from utils.flops import cal_multi_adds, cal_param_size
from utils.covar_temperature import (
    NewtonCoVarConfig,
    covar_temperature_kd_loss,
    newton_covar_temperature_map,
)
from utils.teacher_only_kd import teacher_target_kd_loss
from utils.rtc_temperature import (
    RTCConfig,
    build_rtc_temperature_map,
    collect_rtc_diagnostics,
    compute_reference_reliability,
    file_sha256,
    load_frozen_reliability_cdf,
    masked_temperature_kd_loss,
)
from utils.rtc_o12_calibration import (
    O12CalibrationConfig,
    build_o12_teacher_target,
    build_o12_temperature_map,
    compute_o12_masked_kl_terms,
    normalize_o12_ddp_loss,
    shuffle_o12_temperature_within_images,
)


# Lazy import datasets inside Trainer to avoid unnecessary deps


def npu_is_available():
    return HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device_type(requested):
    requested = str(requested).lower()
    if requested != 'auto':
        if requested == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested --device-type cuda, but CUDA is not available.')
        if requested == 'npu' and not npu_is_available():
            raise RuntimeError('Requested --device-type npu, but torch_npu/NPU is not available.')
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if npu_is_available():
        return 'npu'
    return 'cpu'


def set_accelerator_device(device_type, local_rank):
    if device_type == 'cuda':
        torch.cuda.set_device(local_rank)
    elif device_type == 'npu':
        torch.npu.set_device(local_rank)


def empty_accelerator_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'npu':
        torch.npu.empty_cache()


def seed_everything(seed, rank=0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if npu_is_available():
        torch.npu.manual_seed_all(seed)


def unwrap_module(module):
    return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module


def load_state_dict_compatible(module, state_dict, strict=True):
    """Load state dicts saved with or without a DistributedDataParallel prefix."""
    if state_dict is None:
        return
    cleaned = {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }
    unwrap_module(module).load_state_dict(cleaned, strict=strict)


def _move_value_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    return value


def move_optimizer_state_to_device(optimizer, device):
    if optimizer is None:
        return
    for state_id, state in optimizer.state.items():
        optimizer.state[state_id] = _move_value_to_device(state, device)


def capture_rng_state():
    """Capture RNGs that can affect training after a checkpoint boundary."""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
    if npu_is_available() and hasattr(torch.npu, 'get_rng_state'):
        state['npu'] = torch.npu.get_rng_state()
    return state


def restore_rng_state(state):
    """Restore a state produced by capture_rng_state."""
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])
    if (
        'npu' in state
        and npu_is_available()
        and hasattr(torch.npu, 'set_rng_state')
    ):
        torch.npu.set_rng_state(state['npu'])


def load_checkpoint_file(path):
    """Load trusted local checkpoints across torch versions."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
        return torch.load(path, map_location='cpu')


def is_training_state_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return False
    return (
        checkpoint.get('checkpoint_type') == 'train_kd_training_state'
        or (
            ('student' in checkpoint or 'state_dict' in checkpoint)
            and any(key in checkpoint for key in ('optimizer', 'iteration', 'rng_state'))
        )
    )


def extract_student_state_dict(checkpoint):
    if is_training_state_checkpoint(checkpoint):
        return checkpoint.get('student', checkpoint.get('state_dict'))
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    if isinstance(checkpoint, dict) and 'student' in checkpoint:
        return checkpoint['student']
    return checkpoint


def format_sample_validation_log(sample, pix_acc, miou):
    return "Sample: {:d}, Validation pixAcc: {:.6f}, mIoU: {:.6f}".format(
        int(sample), float(pix_acc), float(miou)
    )


def format_overall_validation_log(pix_acc, miou):
    return "Overall validation pixAcc: {:.6f}, mIoU: {:.6f}".format(
        float(pix_acc), float(miou)
    )

RTC_O12_KD_LOSS_MODE = 'rtc_o12_teacher_target'
RTC_O12_VARIANTS = (
    'neutral',
    'reliable_only',
    'unreliable_only',
    'full_budgeted',
    'unreliable_arithmetic_scalar',
    'unreliable_harmonic_scalar',
    'unreliable_shuffled',
    'full_arithmetic_scalar',
    'full_harmonic_scalar',
    'full_shuffled',
)

RTC_O12_CDF_SHA256 = '8ba8376a03c2f93835339d98670fa357b381b23a22cbf2a7fc48b0c2441d7e69'
RTC_O12_TEACHER_SHA256 = 'ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58'
RTC_O12_STUDENT_INIT_SHA256 = '47085aa164b2977003221458a2a5fdf5f46f434f5539b164dc71969b4dd4cd75'
RTC_O12_TRAIN_LIST_SHA256 = 'd1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24'
RTC_O12_TRAIN_DATASET_SIZE = 10582
RTC_O12_TRAIN_NATIVE_VALID = 32246990
RTC_O12_GATE_PROFILE = 'o12_budgeted_teacher_target_non_synonymous_v1'
RTC_O12_FROZEN_O11_SOURCES = {
    'rtc_temperature': '01b7b6e6aa0d513561332510347b52ea9411330dfb0f2da54abdc36f2375fe59',
    'build_rtc_cdf': 'c88bdfcb885cde01cbf437e2e7751c8eab510067aacf530b469df808ae6604dd',
    'diagnose_rtc_routing': 'f838f25b70b8eadfc982873057e5fb68c54c81089a32e9121fd664e02916f9ef',
    'check_rtc_o11_gate': '55553ec523ac2c2a979470b8542f31878462bbe5a919e0c52132edfbba4eb256',
}


def rtc_o12_canonical_paths():
    diagnostics = os.path.join(cur_path, 'runs', 'diagnostics')
    return {
        'cdf': os.path.realpath(os.path.join(
            diagnostics, 'phaseO_o11', 'voc_train_rtc_confidence_cdf.pt'
        )),
        'o11_gate': os.path.realpath(os.path.join(
            diagnostics, 'phaseO_o11', 'o11_confidence_gate.json'
        )),
        'parameters': os.path.realpath(os.path.join(
            diagnostics, 'phaseO_o12', 'o12_budget_parameters.json'
        )),
        'gate': os.path.realpath(os.path.join(
            diagnostics, 'phaseO_o12', 'o12_joint_gate.json'
        )),
        'train_list': os.path.realpath(os.path.join(
            cur_path, 'dataset', 'list', 'voc', 'train_aug.txt'
        )),
    }


def rtc_o12_expected_configuration():
    config = O12CalibrationConfig()
    config.validate()
    return {
        'phase': 'O1.2',
        'reliability_mode': 'confidence',
        'reliability_definition_id': 'neg_log_top1_confidence_v1',
        'assess_temperature': 1.0,
        'epsilon': 1e-8,
        'q_reliable': config.q_reliable,
        'q_unreliable': config.q_unreliable,
        'p_reliable': config.p_reliable,
        'p_unreliable': config.p_unreliable,
        'a': config.a_star,
        'b_min': 0.0,
        'b_max': config.b_max,
        'target_arithmetic_mean': config.target_mean,
        'minimum_harmonic_mean': config.min_harmonic_mean,
        'bisection_iterations': config.bisection_iterations,
        'teacher_output_temperature': config.teacher_output_temperature,
        'temperature_map_dtype': 'float32',
        'budget_accumulator_dtype': 'float64',
        'temperature_minimum': config.temperature_min,
        'temperature_maximum': config.temperature_max,
        'tau_temperature': 1e-6,
        'tau_probability': 1e-6,
        'tau_entropy': 1e-6,
        'tau_student': 1e-7,
        'tau_formula_monotonic': 1e-12,
    }


def _rtc_o12_resolve(path):
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def _rtc_o12_require(condition, message):
    if not condition:
        raise ValueError('O1.2 contract violation: {}'.format(message))


def _rtc_o12_read_json(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError('O1.2 {} not found: {}'.format(label, path))
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    _rtc_o12_require(isinstance(payload, dict), '{} root is not an object'.format(label))
    return payload


def _rtc_o12_all_true(payload):
    return (
        isinstance(payload, dict)
        and bool(payload)
        and all(value is True for value in payload.values())
    )


def rtc_o12_current_source_sha256():
    paths = {
        'rtc_o12_calibration': os.path.join(cur_path, 'utils', 'rtc_o12_calibration.py'),
        'diagnose_rtc_o12_budget': os.path.join(
            cur_path, 'scripts', 'diagnostics', 'diagnose_rtc_o12_budget.py'
        ),
        'check_rtc_o12_gate': os.path.join(
            cur_path, 'scripts', 'diagnostics', 'check_rtc_o12_gate.py'
        ),
        'train_entry': os.path.join(cur_path, 'train_kd.py'),
    }
    for path in paths.values():
        if not os.path.isfile(path):
            raise FileNotFoundError('O1.2 source not found: {}'.format(path))
    return {name: file_sha256(path) for name, path in paths.items()}


def validate_rtc_o12_cdf_contract(cdf, num_classes):
    _rtc_o12_require(cdf.checksum_sha256 == RTC_O12_CDF_SHA256, 'CDF SHA mismatch')
    metadata = dict(cdf.metadata)
    expected = {
        'phase': 'O1.1',
        'dataset': 'voc',
        'split': 'train_aug',
        'num_classes': 21,
        'processed_images': RTC_O12_TRAIN_DATASET_SIZE,
        'dataset_size': RTC_O12_TRAIN_DATASET_SIZE,
        'full_dataset_scan': True,
        'batch_size': 4,
        'workers': 0,
        'max_images': 0,
        'max_pixels_per_image': 4096,
        'num_quantiles': 4097,
        'crop_size': [512, 512],
        'scale': True,
        'mirror': True,
        'seed': 1234,
        'teacher_output_grid': 'native',
        'valid_mask_resize': 'nearest',
        'assess_temperature': 1.0,
        'coefficient_a': 0.0,
        'coefficient_a_active': False,
        'reliability_mode': 'confidence',
        'reliability_definition_id': 'neg_log_top1_confidence_v1',
        'reliability_epsilon': 1e-8,
        'active_terms': ['confidence'],
        'teacher_sha256': RTC_O12_TEACHER_SHA256,
        'train_list_sha256': RTC_O12_TRAIN_LIST_SHA256,
        'nonfinite_valid_pixels': 0,
    }
    for key, value in expected.items():
        _rtc_o12_require(metadata.get(key) == value, 'CDF metadata {} drift'.format(key))
    _rtc_o12_require(int(num_classes) == 21, 'dataset class count must be 21')
    _rtc_o12_require(
        metadata.get('source_sha256') == RTC_O12_FROZEN_O11_SOURCES,
        'CDF O1.1 source mapping mismatch',
    )
    _rtc_o12_require(
        int(metadata.get('valid_native_pixels', -1))
        == int(metadata.get('finite_valid_pixels', -2)) > 0,
        'CDF finite population mismatch',
    )


def load_rtc_o12_artifact_contract(args):
    """Hard-fail unless all frozen O1.2 inputs and sources agree."""
    paths = rtc_o12_canonical_paths()
    requested = {
        'cdf': _rtc_o12_resolve(args.rtc_o12_cdf_path),
        'parameters': _rtc_o12_resolve(args.rtc_o12_parameters_path),
        'gate': _rtc_o12_resolve(args.rtc_o12_gate_path),
    }
    for name, path in requested.items():
        _rtc_o12_require(path == paths[name], '{} path is not canonical'.format(name))
        if not os.path.isfile(path):
            raise FileNotFoundError('O1.2 {} not found: {}'.format(name, path))
    _rtc_o12_require(
        file_sha256(paths['cdf']) == RTC_O12_CDF_SHA256, 'CDF SHA mismatch'
    )

    o11_source_paths = {
        'rtc_temperature': os.path.join(cur_path, 'utils', 'rtc_temperature.py'),
        'build_rtc_cdf': os.path.join(
            cur_path, 'scripts', 'diagnostics', 'build_rtc_cdf.py'
        ),
        'diagnose_rtc_routing': os.path.join(
            cur_path, 'scripts', 'diagnostics', 'diagnose_rtc_routing.py'
        ),
        'check_rtc_o11_gate': os.path.join(
            cur_path, 'scripts', 'diagnostics', 'check_rtc_o11_gate.py'
        ),
    }
    for name, path in o11_source_paths.items():
        _rtc_o12_require(
            file_sha256(path) == RTC_O12_FROZEN_O11_SOURCES[name],
            'frozen O1.1 source changed: {}'.format(name),
        )
    for label, path, expected_sha in (
        ('teacher', args.teacher_pretrained, RTC_O12_TEACHER_SHA256),
        ('student init', args.student_pretrained_base, RTC_O12_STUDENT_INIT_SHA256),
        ('train list', paths['train_list'], RTC_O12_TRAIN_LIST_SHA256),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError('O1.2 {} not found: {}'.format(label, path))
        _rtc_o12_require(file_sha256(path) == expected_sha, '{} SHA mismatch'.format(label))

    parameters = _rtc_o12_read_json(paths['parameters'], 'parameters')
    gate = _rtc_o12_read_json(paths['gate'], 'joint gate')
    o11_gate = _rtc_o12_read_json(paths['o11_gate'], 'O1.1 gate')
    sources = rtc_o12_current_source_sha256()
    configuration = rtc_o12_expected_configuration()
    parameters_sha = file_sha256(paths['parameters'])
    _rtc_o12_require(o11_gate.get('joint_gate_pass') is True, 'O1.1 gate is false')
    _rtc_o12_require(
        o11_gate.get('actual_cdf_sha256') == RTC_O12_CDF_SHA256,
        'O1.1 gate CDF mismatch',
    )

    parameter_expected = {
        'schema_version': 1,
        'phase': 'O1.2',
        'artifact_kind': 'o12_budget_parameters',
        'split': 'train',
        'stage': 'solve',
        'formal_full_run': True,
        'configuration': configuration,
        'source_sha256': sources,
        'cdf_sha256': RTC_O12_CDF_SHA256,
        'teacher_sha256': RTC_O12_TEACHER_SHA256,
        'list_sha256': RTC_O12_TRAIN_LIST_SHA256,
        'dataset_size': RTC_O12_TRAIN_DATASET_SIZE,
        'processed_images': RTC_O12_TRAIN_DATASET_SIZE,
        'valid_native_pixels': RTC_O12_TRAIN_NATIVE_VALID,
        'finite_native_pixels': RTC_O12_TRAIN_NATIVE_VALID,
        'nonfinite_native_pixels': 0,
        'o11_joint_gate_pass': True,
        'all_checks_pass': True,
    }
    for key, value in parameter_expected.items():
        _rtc_o12_require(
            parameters.get(key) == value, 'parameters {} mismatch'.format(key)
        )
    _rtc_o12_require(_rtc_o12_all_true(parameters.get('checks')), 'parameter checks failed')
    _rtc_o12_require(
        _rtc_o12_resolve(parameters.get('cdf_path', '')) == paths['cdf'],
        'parameter CDF path mismatch',
    )

    gate_expected = {
        'schema_version': 1,
        'phase': 'O1.2',
        'gate_profile': RTC_O12_GATE_PROFILE,
        'joint_gate_pass': True,
        'configuration': configuration,
        'source_sha256': sources,
        'parameters_sha256': parameters_sha,
        'cdf_sha256': RTC_O12_CDF_SHA256,
    }
    for key, value in gate_expected.items():
        _rtc_o12_require(gate.get(key) == value, 'joint gate {} mismatch'.format(key))
    _rtc_o12_require(_rtc_o12_all_true(gate.get('joint_checks')), 'joint checks failed')
    _rtc_o12_require(
        _rtc_o12_resolve(gate.get('parameters_path', '')) == paths['parameters'],
        'joint gate parameter path mismatch',
    )
    _rtc_o12_require(
        _rtc_o12_resolve(gate.get('cdf_path', '')) == paths['cdf'],
        'joint gate CDF path mismatch',
    )
    for key in ('parameters_evaluation', 'train_evaluation', 'val_evaluation'):
        row = gate.get(key)
        _rtc_o12_require(isinstance(row, dict) and row.get('pass') is True,
                         '{} did not pass'.format(key))
        _rtc_o12_require(_rtc_o12_all_true(row.get('checks')),
                         '{} checks failed'.format(key))

    solution = parameters.get('solution')
    _rtc_o12_require(isinstance(solution, dict), 'solution missing')
    b_value = solution.get('b')
    _rtc_o12_require(
        solution.get('feasible') is True
        and isinstance(b_value, (int, float))
        and not isinstance(b_value, bool)
        and math.isfinite(float(b_value))
        and 0.0 <= float(b_value) <= configuration['b_max'],
        'solution b invalid',
    )
    _rtc_o12_require(solution.get('iterations') == 64, 'solution iterations mismatch')
    _rtc_o12_require(
        solution.get('population') == RTC_O12_TRAIN_NATIVE_VALID,
        'solution population mismatch',
    )
    _rtc_o12_require(
        _rtc_o12_float_matches(solution.get('a'), configuration['a']),
        'solution a mismatch',
    )
    _rtc_o12_require(abs(float(solution.get('mean')) - .995) <= 1e-4,
                     'solution mean misses budget')
    _rtc_o12_require(float(solution.get('harmonic_mean')) >= .98,
                     'solution harmonic mean misses budget')
    _rtc_o12_require(float(solution.get('theoretical_high_risk_endpoint')) >= 1.25,
                     'solution high-risk endpoint misses gate')

    scalar_table = parameters.get('branch_scalar_temperatures')
    _rtc_o12_require(isinstance(scalar_table, dict), 'branch scalar table missing')
    scalars = {}
    for branch in ('unreliable_only', 'full_budgeted'):
        row = scalar_table.get(branch)
        _rtc_o12_require(isinstance(row, dict), '{} scalar row missing'.format(branch))
        scalars[branch] = {}
        for moment in ('arithmetic', 'harmonic'):
            value = row.get(moment)
            _rtc_o12_require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and .9 <= float(value) <= 1.5,
                '{} {} scalar invalid'.format(branch, moment),
            )
            scalars[branch][moment] = float(value)
    return {
        'configuration': configuration,
        'b': float(b_value),
        'branch_scalar_temperatures': scalars,
        'cdf_path': paths['cdf'],
        'cdf_sha256': RTC_O12_CDF_SHA256,
        'parameters_path': paths['parameters'],
        'parameters_sha256': parameters_sha,
        'gate_path': paths['gate'],
        'gate_sha256': file_sha256(paths['gate']),
        'o11_gate_path': paths['o11_gate'],
        'o11_gate_sha256': file_sha256(paths['o11_gate']),
        'train_list_path': paths['train_list'],
        'train_list_sha256': RTC_O12_TRAIN_LIST_SHA256,
        'teacher_sha256': RTC_O12_TEACHER_SHA256,
        'student_init_sha256': RTC_O12_STUDENT_INIT_SHA256,
        'source_sha256': sources,
    }


def rtc_o12_is_enabled(args):
    return getattr(args, 'kd_loss_mode', None) == RTC_O12_KD_LOSS_MODE


def _rtc_o12_float_matches(observed, expected, tolerance=1e-12):
    try:
        return math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def validate_rtc_o12_cli_contract(args, world_size=None):
    """Reject ambiguous O1.2 launches before datasets or models are created."""
    if not rtc_o12_is_enabled(args):
        return

    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(not args.use_covar, 'O1.2 forbids --use-covar')
    require(args.rtc_o12_variant in RTC_O12_VARIANTS,
            'O1.2 requires an explicit supported --rtc-o12-variant')
    for option, value in (
        ('--rtc-o12-cdf-path', args.rtc_o12_cdf_path),
        ('--rtc-o12-parameters-path', args.rtc_o12_parameters_path),
        ('--rtc-o12-gate-path', args.rtc_o12_gate_path),
    ):
        require(isinstance(value, str) and bool(value.strip()),
                'O1.2 requires {}'.format(option))

    exact_values = (
        ('--dataset', args.dataset, 'voc'),
        ('--teacher-model', args.teacher_model, 'deeplabv3'),
        ('--teacher-backbone', args.teacher_backbone, 'resnet101'),
        ('--student-model', args.student_model, 'deeplabv3_mobilenet_ssseg'),
        ('--student-backbone', args.student_backbone, 'mobilenetv3_small'),
        ('--ignore-label', args.ignore_label, -1),
        ('--batch-size', args.batch_size, 16),
        ('--workers', args.workers, 8),
        ('--seed', args.seed, 1234),
        ('--log-iter', args.log_iter, 20),
        ('--save-per-iters', args.save_per_iters, 800),
        ('--val-per-iters', args.val_per_iters, 800),
        ('--device-type', args.device_type, 'npu'),
        ('--start_epoch', args.start_epoch, 0),
        ('--local-rank', args.local_rank, 0),
        ('--teacher-pretrained-base', args.teacher_pretrained_base, 'None'),
        ('--student-pretrained', args.student_pretrained, 'None'),
    )
    for option, observed, expected in exact_values:
        require(observed == expected,
                'O1.2 requires {}={} (observed {})'.format(
                    option, expected, observed
                ))
    require(list(args.crop_size) == [512, 512],
            'O1.2 requires --crop-size 512 512')
    require(int(world_size) == 1,
            'O1.2 requires exactly one process/NPU')
    require(args.max_iterations in (20, 20000),
            'O1.2 --max-iterations must be 20 or 20000')
    require(not args.no_cuda, 'O1.2 forbids --no-cuda')
    require(
        args.skip_val is (args.max_iterations == 20),
        'O1.2 requires --skip-val for 20-step smoke and validation for 20k',
    )
    require(
        isinstance(args.teacher_pretrained, str)
        and args.teacher_pretrained != 'None',
        'O1.2 requires an explicit teacher checkpoint',
    )
    require(
        isinstance(args.student_pretrained_base, str)
        and args.student_pretrained_base != 'None',
        'O1.2 requires an explicit student ImageNet initialization',
    )

    float_values = (
        ('--teacher-output-temp', args.teacher_output_temp, 3.0),
        ('--kd-temperature', args.kd_temperature, 1.0),
        ('--lambda-kd', args.lambda_kd, 1.0),
        ('--lambda-adv', args.lambda_adv, 0.001),
        ('--lambda-d', args.lambda_d, 0.1),
        ('--lambda-cwd-fea', args.lambda_cwd_fea, 50.0),
        ('--lambda-cwd-logit', args.lambda_cwd_logit, 3.0),
        ('--lr', args.lr, 0.02),
        ('--momentum', args.momentum, 0.9),
        ('--weight-decay', args.weight_decay, 1e-4),
        ('--lambda-skd', args.lambda_skd, 0.0),
        ('--lambda-ifv', args.lambda_ifv, 0.0),
        ('--lambda-fitnet', args.lambda_fitnet, 0.0),
        ('--lambda-at', args.lambda_at, 0.0),
        ('--lambda-psd', args.lambda_psd, 0.0),
        ('--lambda-csd', args.lambda_csd, 0.0),
    )
    for option, observed, expected in float_values:
        require(
            _rtc_o12_float_matches(observed, expected),
            'O1.2 requires {}={} (observed {})'.format(
                option, expected, observed
            ),
        )

    legacy_rtc_values = (
        args.rtc_enable_reliable,
        args.rtc_enable_unreliable,
        args.rtc_shuffle,
        args.rtc_reverse_routing,
    )
    require(all(value is None for value in legacy_rtc_values),
            'O1.2 forbids legacy RTC branch/shuffle flags')
    if errors:
        raise ValueError('; '.join(errors))


def validate_rtc_o12_logit_shapes(student_logits, teacher_logits):
    if not torch.is_tensor(student_logits) or not torch.is_tensor(teacher_logits):
        raise TypeError('O1.2 student and teacher logits must be tensors')
    if student_logits.ndim != 4 or teacher_logits.ndim != 4:
        raise ValueError('O1.2 logits must both have shape [B, C, H, W]')
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            'O1.2 forbids implicit logit interpolation: student shape {} != '
            'teacher shape {}'.format(
                tuple(student_logits.shape), tuple(teacher_logits.shape)
            )
        )


def build_rtc_o12_dataset_index(list_path):
    with open(list_path, 'r', encoding='utf-8') as handle:
        names = [line.strip() for line in handle if line.strip()]
    if not names:
        raise ValueError('O1.2 canonical train list is empty')
    if len(set(names)) != len(names):
        raise ValueError('O1.2 canonical train list contains duplicate sample names')
    return {name: index for index, name in enumerate(names)}


def resolve_rtc_o12_dataset_indices(sample_names, index_by_name):
    if isinstance(sample_names, str):
        sample_names = [sample_names]
    indices = []
    for name in sample_names:
        if not isinstance(name, str) or name not in index_by_name:
            raise ValueError(
                'O1.2 sample name cannot be mapped uniquely to canonical train '
                'list: {!r}'.format(name)
            )
        indices.append(index_by_name[name])
    return indices


RTC_O12_SAMPLE_ORDER_ALGORITHM = 'rtc_o12_canonical_order_v1'
RTC_O12_SAMPLE_ORDER_SEED = 1234


def rtc_o12_sample_epoch_seed(epoch, seed=RTC_O12_SAMPLE_ORDER_SEED):
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError('O1.2 sampling epoch must be a non-negative integer')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('O1.2 sample-order seed must be non-negative')
    payload = '{}|{}|{}'.format(
        RTC_O12_SAMPLE_ORDER_ALGORITHM, seed, epoch
    ).encode('ascii')
    seed64 = int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        byteorder='big',
        signed=False,
    )
    return seed64 % ((1 << 63) - 1)


class RTCO12CanonicalBatchSampler(data.Sampler):
    """Stateless canonical-ID order with exact completed-iteration slicing."""

    def __init__(
        self,
        canonical_population,
        batch_size,
        num_iterations,
        start_iteration=0,
        seed=RTC_O12_SAMPLE_ORDER_SEED,
    ):
        integers = {
            'canonical_population': canonical_population,
            'batch_size': batch_size,
            'num_iterations': num_iterations,
            'start_iteration': start_iteration,
            'seed': seed,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError('O1.2 {} must be an integer'.format(name))
        if canonical_population <= 0 or batch_size <= 0 or num_iterations <= 0:
            raise ValueError('O1.2 sampler sizes must be positive')
        if not 0 <= start_iteration <= num_iterations:
            raise ValueError('O1.2 start_iteration is outside the run')
        if seed < 0:
            raise ValueError('O1.2 sampler seed must be non-negative')
        self.canonical_population = canonical_population
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.start_iteration = start_iteration
        self.seed = seed
        self._order = self._build_complete_order()
        order_bytes = (
            self._order.numpy()
            .astype('<i8', copy=False)
            .tobytes(order='C')
        )
        self.order_sha256 = hashlib.sha256(order_bytes).hexdigest()

    def _build_complete_order(self):
        required = self.num_iterations * self.batch_size
        chunks = []
        collected = 0
        sampling_epoch = 0
        while collected < required:
            generator = torch.Generator()
            generator.manual_seed(
                rtc_o12_sample_epoch_seed(sampling_epoch, self.seed)
            )
            permutation = torch.randperm(
                self.canonical_population,
                generator=generator,
                dtype=torch.int64,
                device='cpu',
            )
            chunks.append(permutation)
            collected += self.canonical_population
            sampling_epoch += 1
        return torch.cat(chunks, dim=0)[:required].contiguous()

    def __iter__(self):
        for global_iteration in range(
            self.start_iteration + 1, self.num_iterations + 1
        ):
            start = (global_iteration - 1) * self.batch_size
            end = start + self.batch_size
            yield self._order[start:end].tolist()

    def __len__(self):
        return self.num_iterations - self.start_iteration

    def contract(self):
        return {
            'schema_version': 1,
            'algorithm': RTC_O12_SAMPLE_ORDER_ALGORITHM,
            'seed': self.seed,
            'seed_derivation': (
                'seed63=SHA256(ASCII(algorithm|seed|sampling_epoch))[:8] '
                'big-endian mod (2^63-1)'
            ),
            'permutation': 'torch.randperm(canonical_population, CPU generator)',
            'canonical_population': self.canonical_population,
            'batch_size': self.batch_size,
            'max_iterations': self.num_iterations,
            'total_canonical_indices': int(self._order.numel()),
            'complete_order_sha256': self.order_sha256,
            'complete_order_sha256_scope': (
                'full canonical-index sequence for current max_iterations'
            ),
            'resume_offset': 'completed_iteration * batch_size',
        }


def rtc_o12_variant_spec(variant):
    specs = {
        'neutral': ('neutral', None, False),
        'reliable_only': ('reliable_only', None, False),
        'unreliable_only': ('unreliable_only', None, False),
        'full_budgeted': ('full_budgeted', None, False),
        'unreliable_arithmetic_scalar': ('unreliable_only', 'arithmetic', False),
        'unreliable_harmonic_scalar': ('unreliable_only', 'harmonic', False),
        'unreliable_shuffled': ('unreliable_only', None, True),
        'full_arithmetic_scalar': ('full_budgeted', 'arithmetic', False),
        'full_harmonic_scalar': ('full_budgeted', 'harmonic', False),
        'full_shuffled': ('full_budgeted', None, True),
    }
    if variant not in specs:
        raise ValueError('unsupported O1.2 variant: {!r}'.format(variant))
    branch, scalar_moment, shuffled = specs[variant]
    return {
        'variant': variant,
        'branch': branch,
        'scalar_moment': scalar_moment,
        'shuffled': shuffled,
    }


def validate_rtc_o12_temperature_contract(
    temperature,
    reliability_quantile,
    valid_mask,
    variant_spec,
    config,
):
    if temperature.shape != reliability_quantile.shape:
        raise ValueError('O1.2 temperature and quantile shapes must match')
    if valid_mask.dtype != torch.bool or valid_mask.shape != temperature.shape:
        raise ValueError('O1.2 native valid mask must be Boolean [B, H, W]')
    if temperature.dtype != torch.float32:
        raise TypeError('O1.2 temperature map must be float32')
    selected = temperature[valid_mask]
    if selected.numel() and not bool(torch.isfinite(selected).all().item()):
        raise FloatingPointError('O1.2 temperature contains non-finite valid values')
    tolerance = 1e-6
    if selected.numel() and bool(
        (
            (selected < config.temperature_min - tolerance)
            | (selected > config.temperature_max + tolerance)
        ).any().item()
    ):
        raise AssertionError('O1.2 temperature violates frozen bounds')
    invalid = ~valid_mask
    if bool(invalid.any().item()) and not torch.equal(
        temperature[invalid], torch.ones_like(temperature[invalid])
    ):
        raise AssertionError('O1.2 invalid pixels must have exactly T=1')
    if variant_spec['shuffled'] or variant_spec['scalar_moment'] is not None:
        return

    branch = variant_spec['branch']
    quantile = reliability_quantile.to(device=temperature.device)
    if branch == 'neutral':
        if not torch.equal(selected, torch.ones_like(selected)):
            raise AssertionError('O1.2 neutral branch must have exactly T=1')
        return
    if branch == 'reliable_only':
        active = valid_mask & (quantile < config.q_reliable)
        inactive = valid_mask & ~active
        if bool(active.any().item()) and bool(
            (temperature[active] > 1.0 + tolerance).any().item()
        ):
            raise AssertionError('O1.2 reliable branch produced T>1')
    elif branch == 'unreliable_only':
        active = valid_mask & (quantile > config.q_unreliable)
        inactive = valid_mask & ~active
        if bool(active.any().item()) and bool(
            (temperature[active] < 1.0 - tolerance).any().item()
        ):
            raise AssertionError('O1.2 unreliable branch produced T<1')
    else:
        active = None
        inactive = (
            valid_mask
            & (quantile >= config.q_reliable)
            & (quantile <= config.q_unreliable)
        )
        reliable = valid_mask & (quantile < config.q_reliable)
        unreliable = valid_mask & (quantile > config.q_unreliable)
        if bool(reliable.any().item()) and bool(
            (temperature[reliable] > 1.0 + tolerance).any().item()
        ):
            raise AssertionError('O1.2 full reliable pixels produced T>1')
        if bool(unreliable.any().item()) and bool(
            (temperature[unreliable] < 1.0 - tolerance).any().item()
        ):
            raise AssertionError('O1.2 full unreliable pixels produced T<1')
    if bool(inactive.any().item()) and not torch.equal(
        temperature[inactive], torch.ones_like(temperature[inactive])
    ):
        raise AssertionError('O1.2 inactive/neutral pixels must have exactly T=1')


def build_rtc_o12_temperature_for_variant(
    reliability_quantile,
    valid_mask,
    variant,
    artifact_contract,
    config,
    dataset_indices=None,
    global_iteration=None,
):
    spec = rtc_o12_variant_spec(variant)
    if spec['scalar_moment'] is not None:
        scalar = artifact_contract['branch_scalar_temperatures'][
            spec['branch']
        ][spec['scalar_moment']]
        base = torch.full_like(
            reliability_quantile, float(scalar), dtype=torch.float32
        )
        temperature = torch.where(valid_mask, base, torch.ones_like(base))
    else:
        coefficient_b = (
            artifact_contract['b']
            if spec['branch'] in ('unreliable_only', 'full_budgeted')
            else None
        )
        temperature = build_o12_temperature_map(
            reliability_quantile,
            valid_mask,
            b=coefficient_b,
            branch=spec['branch'],
            config=config,
        )
        if spec['shuffled']:
            if dataset_indices is None or global_iteration is None:
                raise ValueError('O1.2 shuffled variant requires index and iteration')
            temperature = shuffle_o12_temperature_within_images(
                temperature,
                valid_mask,
                dataset_indices,
                int(global_iteration),
            )
    validate_rtc_o12_temperature_contract(
        temperature, reliability_quantile, valid_mask, spec, config
    )
    return temperature, spec


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='deeplabv3',
                        help='model name')  
    parser.add_argument('--student-model', type=str, default='deeplabv3',
                        help='model name')                      
    parser.add_argument('--student-backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='resnet101',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/VOCAug/',  
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='ignore label')
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')


    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    parser.add_argument('--kd-loss-mode', type=str, default='legacy',
                        choices=['legacy', 'masked', 'teacher_only', RTC_O12_KD_LOSS_MODE],
                        help='legacy, fair masked, teacher-target-only, or strict O1.2 pixel KD')
    parser.add_argument("--lambda-kd", type=float, default=0., help="lambda_kd")
    parser.add_argument("--lambda-adv", type=float, default=0., help="lambda adversarial loss")
    parser.add_argument("--lambda-d", type=float, default=0., help="lambda discriminator loss")
    parser.add_argument("--lambda-skd", type=float, default=0., help="lambda skd")
    parser.add_argument("--lambda-cwd-fea", type=float, default=0., help="lambda cwd feature")
    parser.add_argument("--lambda-cwd-logit", type=float, default=0., help="lambda cwd logit")
    parser.add_argument("--lambda-ifv", type=float, default=0., help="lambda ifvd")
    parser.add_argument("--lambda-fitnet", type=float, default=0., help="lambda fitnet")
    parser.add_argument("--lambda-at", type=float, default=0., help="lambda attention transfer")
    parser.add_argument("--lambda-psd", type=float, default=0., help="lambda pixel similarity KD")
    parser.add_argument("--lambda-csd", type=float, default=0., help="lambda category similarity KD")
    parser.add_argument('--use-covar', action='store_true', default=False,
                        help='enable CoVar in the logit KD path')
    parser.add_argument('--covar-temp-mode', type=str, default='newton',
                        choices=['newton', 'legacy_weight', 'rtc'],
                        help='paper Newton temperature or the earlier Gaussian weighting implementation')
    parser.add_argument('--covar-alpha', type=float, default=2.0,
                        help='alpha used only by legacy_weight mode')
    parser.add_argument('--teacher-output-temp', type=float, default=1.0,
                        help='soften teacher logits in the logit KD path only')
    parser.add_argument('--covar-temp-base', type=float, default=1.0,
                        help='initial temperature for Newton CoVar')
    parser.add_argument('--covar-temp-min', type=float, default=0.5,
                        help='minimum pixel temperature for Newton CoVar')
    parser.add_argument('--covar-temp-max', type=float, default=8.0,
                        help='maximum pixel temperature for Newton CoVar')
    parser.add_argument('--covar-kd-temp-power', type=float, default=2.0,
                        help='gamma in KL(student/T, teacher/T) * T^gamma')
    parser.add_argument('--covar-grad-eta', type=float, default=0.6,
                        help='damping factor for Newton temperature updates')
    parser.add_argument('--covar-grad-max-iter', type=int, default=8,
                        help='number of Newton temperature updates')
    parser.add_argument('--covar-a', type=float, default=None,
                        help='reliability variance coefficient; default is (K-1)^2/2')
    parser.add_argument('--covar-reliability-mode', type=str, default='full',
                        choices=['full', 'confidence', 'variance'],
                        help='reliability terms used by Newton CoVar')
    parser.add_argument('--covar-newton-hessian-eps', type=float, default=1e-5,
                        help='minimum positive Hessian magnitude for a Newton step')
    parser.add_argument('--covar-newton-max-step', type=float, default=0.25,
                        help='maximum absolute Newton step; <=0 disables clipping')
    parser.add_argument('--rtc-cdf-path', type=str, default=None,
                        help='frozen training-set reliability CDF artifact')
    parser.add_argument('--rtc-assess-temperature', type=float, default=1.0,
                        help='fixed temperature used only for raw-teacher reliability')
    parser.add_argument('--rtc-route-quantile', type=float, default=0.80,
                        help='high-risk route starts above this frozen-CDF quantile')
    parser.add_argument('--rtc-route-width', type=float, default=0.05,
                        help='continuous route transition width')
    parser.add_argument('--rtc-temp-reliable', type=float, default=0.5,
                        help='strong-sharpening endpoint temperature')
    parser.add_argument('--rtc-temp-neutral', type=float, default=1.0,
                        help='neutral pixel temperature')
    parser.add_argument('--rtc-temp-unreliable', type=float, default=2.0,
                        help='smoothing endpoint temperature')
    parser.add_argument('--rtc-alpha-reliable', type=float, default=1.0,
                        help='fraction of the reliable endpoint target')
    parser.add_argument('--rtc-alpha-unreliable', type=float, default=1.0,
                        help='fraction of the unreliable endpoint target')
    parser.add_argument(
        '--rtc-enable-reliable',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='explicitly enable/disable the low-risk sharpening branch',
    )
    parser.add_argument(
        '--rtc-enable-unreliable',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='explicitly enable/disable the high-risk smoothing branch',
    )
    parser.add_argument('--rtc-bisection-iters', type=int, default=16,
                        help='fixed target-temperature bisection iterations')
    parser.add_argument('--rtc-shuffle', action=argparse.BooleanOptionalAction, default=None,
                        help='explicitly enable/disable within-image temperature shuffling')
    parser.add_argument('--rtc-reverse-routing', action=argparse.BooleanOptionalAction, default=None,
                        help='explicitly enable/disable swapping reliable and unreliable gates')
               

    parser.add_argument('--rtc-o12-variant', type=str, default=None,
                        choices=RTC_O12_VARIANTS,
                        help='explicit O1.2 teacher-target calibration/control variant')
    parser.add_argument('--rtc-o12-cdf-path', type=str, default=None,
                        help='frozen O1.1 confidence CDF required by O1.2')
    parser.add_argument('--rtc-o12-parameters-path', type=str, default=None,
                        help='frozen O1.2 train budget parameter artifact')
    parser.add_argument('--rtc-o12-gate-path', type=str, default=None,
                        help='passing O1.2-A joint gate artifact')
    # accelerator setting
    parser.add_argument('--device-type', type=str, default='auto', choices=['auto', 'cuda', 'npu', 'cpu'],
                        help='accelerator backend; auto prefers CUDA, then Ascend NPU, then CPU')
    parser.add_argument('--seed', type=int, default=1234,
                        help='base random seed for reproducible training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local-rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--keep-checkpoint-iters', type=int, nargs='*', default=[],
                        help='iterations whose student and training states are kept separately')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
    parser.add_argument('--teacher-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str, default='None',
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='None',
                    help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')

                        
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()
    if args.teacher_output_temp <= 0:
        parser.error('--teacher-output-temp must be positive')
    if args.kd_temperature <= 0:
        parser.error('--kd-temperature must be positive')
    if any(step <= 0 or step > args.max_iterations
           for step in args.keep_checkpoint_iters):
        parser.error('--keep-checkpoint-iters must lie in [1, max-iterations]')
    if len(set(args.keep_checkpoint_iters)) != len(args.keep_checkpoint_iters):
        parser.error('--keep-checkpoint-iters must not contain duplicates')
    if args.kd_loss_mode == 'teacher_only':
        if args.teacher_output_temp != 1.0:
            parser.error('teacher_only requires --teacher-output-temp 1.0')
        if args.use_covar:
            parser.error('teacher_only is a scalar target-temperature control; omit --use-covar')
    try:
        validate_rtc_o12_cli_contract(args)
    except ValueError as error:
        parser.error(str(error))
    if args.use_covar and args.covar_temp_mode == 'rtc':
        explicit_rtc_flags = {
            '--rtc-enable-reliable': args.rtc_enable_reliable,
            '--rtc-enable-unreliable': args.rtc_enable_unreliable,
            '--rtc-shuffle': args.rtc_shuffle,
            '--rtc-reverse-routing': args.rtc_reverse_routing,
        }
        missing_flags = [
            name for name, value in explicit_rtc_flags.items() if value is None
        ]
        if missing_flags:
            parser.error(
                'RTC requires explicit Boolean choices: {}'.format(
                    ', '.join(missing_flags)
                )
            )
        if args.kd_loss_mode != 'masked':
            parser.error('RTC requires --kd-loss-mode masked')
        if not args.rtc_cdf_path:
            parser.error('RTC requires --rtc-cdf-path')
        if args.dataset != 'voc':
            parser.error('Phase O RTC currently supports only VOC with a matching frozen CDF')
        if args.resume:
            parser.error('Phase O RTC formal runs forbid --resume; start a fresh run directory')
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.student_backbone.startswith('resnet'):
        args.aux = True
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        if args.distributed and args.device in ("cuda", "npu"):
            self.device = torch.device(f"{args.device}:{args.local_rank}")
        else:
            self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.rank = get_rank()
        if rtc_o12_is_enabled(args):
            validate_rtc_o12_cli_contract(args, world_size=self.num_gpus)
        self.resume_checkpoint = None
        self.resume_is_full_state = False
        self.start_iteration = 0
        if args.resume:
            if not os.path.isfile(args.resume):
                raise FileNotFoundError('Resume checkpoint not found: {}'.format(args.resume))
            extension = os.path.splitext(args.resume)[1].lower()
            if extension not in ('.pth', '.pkl'):
                raise ValueError('Only .pth and .pkl checkpoints are supported.')
            print('Resuming training, loading {}...'.format(args.resume))
            self.resume_checkpoint = load_checkpoint_file(args.resume)
            self.resume_is_full_state = is_training_state_checkpoint(self.resume_checkpoint)
            if self.resume_is_full_state:
                self.start_iteration = int(self.resume_checkpoint.get('iteration', 0))
                if self.start_iteration > args.max_iterations:
                    raise ValueError(
                        'Checkpoint iteration {} exceeds --max-iterations {}'.format(
                            self.start_iteration, args.max_iterations
                        )
                    )

        self.rtc_o12_config = None
        self.rtc_o12_contract = None
        self.rtc_o12_cdf = None
        self.rtc_o12_dataset_index_by_name = None
        self.rtc_o12_batch_sampler = None
        self.rtc_o12_data_order_contract = None
        if rtc_o12_is_enabled(args):
            if self.resume_checkpoint is not None and not self.resume_is_full_state:
                raise ValueError(
                    'O1.2 resume requires a complete O1.2 training-state checkpoint'
                )
            self.rtc_o12_config = O12CalibrationConfig()
            self.rtc_o12_config.validate()
            self.rtc_o12_contract = load_rtc_o12_artifact_contract(args)

        self.covar_newton_config = None
        if args.use_covar and args.covar_temp_mode == 'newton':
            self.covar_newton_config = NewtonCoVarConfig(
                base_temperature=args.covar_temp_base,
                min_temperature=args.covar_temp_min,
                max_temperature=args.covar_temp_max,
                kd_temperature_power=args.covar_kd_temp_power,
                eta=args.covar_grad_eta,
                max_iterations=args.covar_grad_max_iter,
                hessian_epsilon=args.covar_newton_hessian_eps,
                max_step=args.covar_newton_max_step,
                coefficient_a=args.covar_a,
                reliability_mode=args.covar_reliability_mode,
            )
            self.covar_newton_config.validate()

        self.rtc_config = None
        self.rtc_cdf = None
        if args.use_covar and args.covar_temp_mode == 'rtc':
            self.rtc_config = RTCConfig(
                assess_temperature=args.rtc_assess_temperature,
                route_quantile=args.rtc_route_quantile,
                route_width=args.rtc_route_width,
                reliable_temperature=args.rtc_temp_reliable,
                neutral_temperature=args.rtc_temp_neutral,
                unreliable_temperature=args.rtc_temp_unreliable,
                alpha_reliable=args.rtc_alpha_reliable,
                alpha_unreliable=args.rtc_alpha_unreliable,
                enable_reliable=args.rtc_enable_reliable,
                enable_unreliable=args.rtc_enable_unreliable,
                bisection_iterations=args.rtc_bisection_iters,
                kd_temperature_power=args.covar_kd_temp_power,
                coefficient_a=args.covar_a,
                reliability_mode=args.covar_reliability_mode,
            )
            self.rtc_config.validate()


        # Lazy dataset imports to avoid importing unused backends
        if args.dataset == 'citys':
            from dataset.cityscapes import CSTrainValSet
            train_dataset = CSTrainValSet(args.data, 
                                            list_path='./dataset/list/cityscapes/train.lst', 
                                            max_iters=args.max_iterations*args.batch_size, 
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(args.data, 
                                        list_path='./dataset/list/cityscapes/val.lst', 
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            from dataset.voc import VOCDataTrainSet, VOCDataValSet
            self._assert_voc_aug_paths(args.data)
            if not args.skip_val:
                self._assert_voc_val_paths(args.data)
            train_dataset = VOCDataTrainSet(args.data, './dataset/list/voc/train_aug.txt', max_iters=args.max_iterations*args.batch_size, 
                                          crop_size=args.crop_size, scale=True, mirror=True, ignore_label=args.ignore_label)
            val_dataset = VOCDataValSet(args.data, './dataset/list/voc/val.txt', ignore_label=args.ignore_label)
        elif args.dataset == 'ade20k':
            from dataset.ade20k import ADETrainSet, ADEDataValSet
            train_dataset = ADETrainSet(args.data, max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = ADEDataValSet(args.data)
        elif args.dataset == 'camvid':
            from dataset.camvid import CamvidTrainSet, CamvidValSet
            train_dataset = CamvidTrainSet(args.data, './dataset/list/CamVid/camvid_train_list.txt', max_iters=args.max_iterations*args.batch_size,
                            ignore_label=args.ignore_label, crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CamvidValSet(args.data, './dataset/list/CamVid/camvid_val_list.txt')
        elif args.dataset == 'coco_stuff_164k':
            from dataset.coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet
            train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt', max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CocoStuff164kValSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
        else:
            raise ValueError('dataset unfind')

        if self.rtc_o12_config is not None:
            self.rtc_o12_cdf = load_frozen_reliability_cdf(
                self.rtc_o12_contract['cdf_path'],
                device=self.device,
            )
            validate_rtc_o12_cdf_contract(
                self.rtc_o12_cdf, train_dataset.num_class
            )
            self.rtc_o12_dataset_index_by_name = build_rtc_o12_dataset_index(
                self.rtc_o12_contract['train_list_path']
            )
            if len(self.rtc_o12_dataset_index_by_name) != RTC_O12_TRAIN_DATASET_SIZE:
                raise ValueError('O1.2 canonical train list population mismatch')
            canonical_names = list(
                self.rtc_o12_dataset_index_by_name.keys()
            )
            dataset_names = list(
                getattr(train_dataset, 'img_ids', [])
            )[:RTC_O12_TRAIN_DATASET_SIZE]
            if dataset_names != canonical_names:
                raise ValueError(
                    'O1.2 sampler indices do not match canonical sample names'
                )
            args.rtc_o12_cdf_path = self.rtc_o12_cdf.path
            args.rtc_o12_cdf_sha256 = self.rtc_o12_cdf.checksum_sha256
            args.rtc_o12_parameters_path = self.rtc_o12_contract['parameters_path']
            args.rtc_o12_parameters_sha256 = self.rtc_o12_contract[
                'parameters_sha256'
            ]
            args.rtc_o12_gate_path = self.rtc_o12_contract['gate_path']
            args.rtc_o12_gate_sha256 = self.rtc_o12_contract['gate_sha256']
            logger.info(
                'Loaded O1.2 frozen inputs: variant={} b={:.12g} cdf={} '
                'parameters={} gate={}'.format(
                    args.rtc_o12_variant,
                    self.rtc_o12_contract['b'],
                    self.rtc_o12_contract['cdf_sha256'][:12],
                    self.rtc_o12_contract['parameters_sha256'][:12],
                    self.rtc_o12_contract['gate_sha256'][:12],
                )
            )

        if self.rtc_config is not None:
            self.rtc_cdf = load_frozen_reliability_cdf(
                args.rtc_cdf_path,
                device=self.device,
            )
            self._validate_rtc_cdf_metadata(self.rtc_cdf, train_dataset.num_class)
            cdf_checksums = all_gather(self.rtc_cdf.checksum_sha256)
            if len(set(cdf_checksums)) != 1:
                raise RuntimeError(
                    'RTC CDF checksum differs across ranks: {}'.format(cdf_checksums)
                )
            args.rtc_cdf_path = self.rtc_cdf.path
            args.rtc_cdf_sha256 = self.rtc_cdf.checksum_sha256
            logger.info(
                'Loaded frozen RTC CDF: path={} sha256={} metadata={}'.format(
                    self.rtc_cdf.path, self.rtc_cdf.checksum_sha256,
                    dict(self.rtc_cdf.metadata),
                )
            )

    
        args.batch_size = args.batch_size // self.num_gpus
        if self.rtc_o12_config is not None:
            self.rtc_o12_batch_sampler = RTCO12CanonicalBatchSampler(
                canonical_population=RTC_O12_TRAIN_DATASET_SIZE,
                batch_size=args.batch_size,
                num_iterations=args.max_iterations,
                start_iteration=self.start_iteration,
                seed=args.seed,
            )
            self.rtc_o12_data_order_contract = (
                self.rtc_o12_batch_sampler.contract()
            )
            args.rtc_o12_complete_order_sha256 = (
                self.rtc_o12_data_order_contract[
                    'complete_order_sha256'
                ]
            )
            train_batch_sampler = self.rtc_o12_batch_sampler
            logger.info(
                'O1.2 canonical sample order: algorithm={} '
                'completed_iteration={} full_order_sha256={}'.format(
                    self.rtc_o12_data_order_contract['algorithm'],
                    self.start_iteration,
                    args.rtc_o12_complete_order_sha256,
                )
            )
        else:
            train_sampler = make_data_sampler(
                train_dataset, shuffle=True, distributed=args.distributed
            )
            train_batch_sampler = make_batch_data_sampler(
                train_sampler,
                args.batch_size,
                args.max_iterations,
                start_iter=self.start_iteration,
            )
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        self.t_model = get_segmentation_model(model=args.teacher_model, 
                                            backbone=args.teacher_backbone,
                                            local_rank=args.local_rank,
                                            pretrained_base='None',
                                            pretrained=args.teacher_pretrained,
                                            aux=True, 
                                            norm_layer=nn.BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device)

        self.s_model = get_segmentation_model(model=args.student_model, 
                                            backbone=args.student_backbone,
                                            local_rank=args.local_rank,
                                            pretrained_base=args.student_pretrained_base,
                                            pretrained='None',
                                            aux=args.aux, 
                                            norm_layer=BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device)
        
        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()

        self.use_adv = args.lambda_adv != 0. or args.lambda_d != 0.
        self.D_model = None
        if self.use_adv:
            self.D_model = Discriminator(
                preprocess_GAN_mode=1,
                input_channel=train_dataset.num_class,
                distributed=args.distributed,
            ).to(self.device)

        # create criterion
        x = torch.randn(1, 3, args.crop_size[0], args.crop_size[1]).to(self.device)
        t_y = self.t_model(x)
        s_y = self.s_model(x)
        t_channels = t_y[-1].size(1)
        s_channels = s_y[-1].size(1)

        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = CriterionKD(temperature=args.kd_temperature).to(self.device)
        self.criterion_adv = CriterionAdv('hinge').to(self.device)
        self.criterion_adv_for_G = CriterionAdvForG('hinge').to(self.device)
        self.criterion_skd = CriterionStructuralKD().to(self.device)
        self.criterion_ifv = CriterionIFV(train_dataset.num_class).to(self.device)
        self.criterion_cwd = CriterionCWD(s_channels, t_channels, norm_type='channel',divergence='kl', temperature=4.).to(self.device)
        self.criterion_fitnet = CriterionFitNet(s_channels, t_channels).to(self.device)
        self.criterion_at = CriterionAT().to(self.device)
        self.criterion_dsd = CriterionDoubleSimKD().to(self.device)

    
        params_list = nn.ModuleList([])
        params_list.append(self.s_model)
        params_list.append(self.criterion_cwd)
        params_list.append(self.criterion_fitnet)


        self.optimizer = torch.optim.SGD(params_list.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        self.D_optimizer = None
        if self.use_adv:
            self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                self.D_model.parameters()),
                                                4e-4, [0.9, 0.99])
        
        if args.distributed:
            ddp_kwargs = {}
            if self.device.type in ('cuda', 'npu'):
                ddp_kwargs = {'device_ids': [args.local_rank], 'output_device': args.local_rank}
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model, **ddp_kwargs)
            if self.use_adv:
                self.D_model = nn.parallel.DistributedDataParallel(self.D_model, **ddp_kwargs)
            self.criterion_cwd = nn.parallel.DistributedDataParallel(self.criterion_cwd, **ddp_kwargs)
            self.criterion_fitnet = nn.parallel.DistributedDataParallel(self.criterion_fitnet, **ddp_kwargs)
            
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0
        self.current_iteration = self.start_iteration
        if self.resume_checkpoint is not None:
            self._load_resume_checkpoint(self.resume_checkpoint, args.resume)

    def _validate_rtc_cdf_metadata(self, cdf, num_classes):
        metadata = dict(cdf.metadata)
        required = (
            'num_classes',
            'assess_temperature',
            'coefficient_a',
            'reliability_mode',
            'teacher_sha256',
            'train_list_sha256',
            'dataset',
            'split',
            'processed_images',
            'dataset_size',
            'max_images',
            'full_dataset_scan',
            'batch_size',
            'workers',
            'valid_native_pixels',
            'finite_valid_pixels',
            'nonfinite_valid_pixels',
            'sample_count',
            'num_quantiles',
            'max_pixels_per_image',
            'crop_size',
            'scale',
            'mirror',
            'seed',
            'teacher_output_grid',
            'valid_mask_resize',
            'source_sha256',
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError('RTC CDF metadata missing fields: {}'.format(missing))
        if metadata['dataset'] != 'voc':
            raise ValueError('Phase O RTC requires a VOC training-set CDF')
        if int(metadata['processed_images']) != int(metadata['dataset_size']):
            raise ValueError('RTC CDF was not built from a complete dataset scan')
        if int(metadata['num_classes']) != int(num_classes):
            raise ValueError('RTC CDF num_classes does not match the dataset')
        pre_registered_metadata = {
            'split': 'train_aug',
            'max_images': 0,
            'full_dataset_scan': True,
            'batch_size': 4,
            'workers': 0,
            'max_pixels_per_image': 4096,
            'num_quantiles': 4097,
            'crop_size': [512, 512],
            'scale': True,
            'mirror': True,
            'seed': 1234,
            'teacher_output_grid': 'native',
            'valid_mask_resize': 'nearest',
        }
        for key, expected_value in pre_registered_metadata.items():
            if metadata[key] != expected_value:
                raise ValueError(
                    'RTC CDF metadata drift for {}: expected={} observed={}'.format(
                        key, expected_value, metadata[key]
                    )
                )
        for boolean_key in ('full_dataset_scan', 'scale', 'mirror'):
            if metadata[boolean_key] is not True:
                raise ValueError(
                    'RTC CDF metadata {} must be the Boolean true'.format(
                        boolean_key
                    )
                )
        processed_images = int(metadata['processed_images'])
        dataset_size = int(metadata['dataset_size'])
        valid_native_pixels = int(metadata['valid_native_pixels'])
        finite_valid_pixels = int(metadata['finite_valid_pixels'])
        nonfinite_valid_pixels = int(metadata['nonfinite_valid_pixels'])
        sampled_pixels = int(metadata['sample_count'])
        if processed_images <= 0 or dataset_size <= 0:
            raise ValueError('RTC CDF dataset counters must be positive')
        if valid_native_pixels != finite_valid_pixels + nonfinite_valid_pixels:
            raise ValueError('RTC CDF valid-pixel counters are inconsistent')
        if nonfinite_valid_pixels != 0:
            raise ValueError('RTC CDF formal scan contains non-finite valid pixels')
        if sampled_pixels <= 0 or sampled_pixels > finite_valid_pixels:
            raise ValueError('RTC CDF sampled-pixel count is invalid')
        source_sha256 = metadata['source_sha256']
        if not isinstance(source_sha256, dict):
            raise ValueError('RTC CDF source_sha256 must be a dictionary')
        expected_rtc_sha256 = file_sha256(
            os.path.join(cur_path, 'utils', 'rtc_temperature.py')
        )
        if source_sha256.get('rtc_temperature') != expected_rtc_sha256:
            raise ValueError(
                'RTC CDF was built with a different RTC reliability implementation'
            )
        if not math.isclose(
            float(metadata['assess_temperature']),
            float(self.rtc_config.assess_temperature),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('RTC CDF assess_temperature does not match the run')
        expected_a = self.rtc_config.coefficient_a
        if expected_a is None:
            expected_a = float((int(num_classes) - 1) ** 2) / 2.0
        if not math.isclose(
            float(metadata['coefficient_a']), float(expected_a),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError('RTC CDF coefficient_a does not match the run')
        if metadata['reliability_mode'] != self.rtc_config.reliability_mode:
            raise ValueError('RTC CDF reliability_mode does not match the run')
        if file_sha256(self.args.teacher_pretrained) != metadata['teacher_sha256']:
            raise ValueError('RTC CDF teacher checksum does not match the run')
        train_list_path = './dataset/list/voc/train_aug.txt'
        if self.args.dataset == 'voc' and file_sha256(train_list_path) != metadata['train_list_sha256']:
            raise ValueError('RTC CDF train-list checksum does not match the run')

    def _load_resume_checkpoint(self, checkpoint, path):
        if self.rtc_o12_config is not None:
            required = (
                'student',
                'criterion_cwd',
                'criterion_fitnet',
                'D',
                'optimizer',
                'D_optimizer',
                'iteration',
                'rng_state_by_rank',
                'rtc_o12_data_order',
            )
            missing = [
                key for key in required
                if key not in checkpoint or checkpoint.get(key) is None
            ]
            if (
                checkpoint.get('checkpoint_type') != 'train_kd_training_state'
                or checkpoint.get('checkpoint_version') != 4
                or missing
            ):
                raise ValueError(
                    'O1.2 resume requires checkpoint v4 complete state; '
                    'missing={}'.format(missing)
                )
            saved_o12 = checkpoint.get('rtc_o12')
            expected_o12 = self._rtc_o12_checkpoint_metadata()
            if saved_o12 != expected_o12:
                raise ValueError('O1.2 resume metadata/source/configuration drift')
            expected_order_state = self._rtc_o12_data_order_state(
                int(checkpoint['iteration'])
            )
            if checkpoint.get('rtc_o12_data_order') != expected_order_state:
                raise ValueError(
                    'O1.2 resume canonical sample-order state drift'
                )
        if self.rtc_config is not None:
            saved_rtc = checkpoint.get('rtc') if isinstance(checkpoint, dict) else None
            if not saved_rtc:
                raise ValueError('RTC resume checkpoint has no frozen RTC metadata')
            expected = self._rtc_checkpoint_metadata()
            for key in (
                'config', 'cdf_sha256', 'kd_loss_mode',
                'teacher_output_temp', 'shuffle', 'reverse_routing', 'world_size',
            ):
                if saved_rtc.get(key) != expected.get(key):
                    raise ValueError(
                        'RTC resume configuration drift for {}: saved={} current={}'.format(
                            key, saved_rtc.get(key), expected.get(key)
                        )
                    )
        student_state = extract_student_state_dict(checkpoint)
        load_state_dict_compatible(self.s_model, student_state, strict=True)
        if not self.resume_is_full_state:
            logger.info(
                'Loaded legacy student-only checkpoint from {}; training starts at iteration 0'.format(path)
            )
            return

        module_states = (
            ('criterion_cwd', self.criterion_cwd),
            ('criterion_fitnet', self.criterion_fitnet),
            ('D', self.D_model),
        )
        for key, module in module_states:
            if module is not None and checkpoint.get(key) is not None:
                load_state_dict_compatible(module, checkpoint[key], strict=True)

        if checkpoint.get('optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            move_optimizer_state_to_device(self.optimizer, self.device)
        if self.D_optimizer is not None and checkpoint.get('D_optimizer') is not None:
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
            move_optimizer_state_to_device(self.D_optimizer, self.device)

        self.start_iteration = int(checkpoint.get('iteration', self.start_iteration))
        self.current_iteration = self.start_iteration
        self.best_pred = float(checkpoint.get('best_pred', self.best_pred))
        rng_state = checkpoint.get('rng_state')
        rank_states = checkpoint.get('rng_state_by_rank')
        if isinstance(rank_states, (list, tuple)) and self.rank < len(rank_states):
            rng_state = rank_states[self.rank]
        elif isinstance(rank_states, dict):
            rng_state = rank_states.get(str(self.rank), rank_states.get(self.rank, rng_state))
        restore_rng_state(rng_state)
        logger.info(
            'Resumed full training state from {}: iteration={}, best_mIoU={:.6f}'.format(
                path, self.start_iteration, self.best_pred
            )
        )

    def _rtc_checkpoint_metadata(self):
        if self.rtc_config is None:
            return None
        return {
            'config': self.rtc_config.to_dict(),
            'cdf_path': self.rtc_cdf.path,
            'cdf_sha256': self.rtc_cdf.checksum_sha256,
            'cdf_metadata': dict(self.rtc_cdf.metadata),
            'kd_loss_mode': self.args.kd_loss_mode,
            'teacher_output_temp': float(self.args.teacher_output_temp),
            'shuffle': bool(self.args.rtc_shuffle),
            'reverse_routing': bool(self.args.rtc_reverse_routing),
            'world_size': int(get_world_size()),
        }

    def _rtc_o12_checkpoint_metadata(self):
        if self.rtc_o12_config is None:
            return None
        spec = rtc_o12_variant_spec(self.args.rtc_o12_variant)
        scalar_temperature = None
        if spec['scalar_moment'] is not None:
            scalar_temperature = self.rtc_o12_contract[
                'branch_scalar_temperatures'
            ][spec['branch']][spec['scalar_moment']]
        return {
            'phase': 'O1.2',
            'variant': self.args.rtc_o12_variant,
            'variant_spec': spec,
            'calibration_config': self.rtc_o12_config.to_dict(),
            'artifact_contract': dict(self.rtc_o12_contract),
            'teacher_target_contract': {
                'formula': 'softmax(raw_teacher/(teacher_output_temperature*T_pixel))',
                'teacher_output_temperature': 3.0,
                'student_temperature': 1.0,
                'teacher_target_detached': True,
                'temperature_loss_power': None,
            },
            'scalar_temperature': scalar_temperature,
            'data_order_contract': dict(
                self.rtc_o12_data_order_contract
            ),
            'world_size': int(get_world_size()),
            'max_iterations': int(self.args.max_iterations),
            'skip_val': bool(self.args.skip_val),
        }

    def _rtc_o12_data_order_state(self, completed_iteration):
        completed_iteration = int(completed_iteration)
        if not 0 <= completed_iteration <= self.args.max_iterations:
            raise ValueError('O1.2 completed iteration is outside the run')
        next_iteration = (
            completed_iteration + 1
            if completed_iteration < self.args.max_iterations
            else None
        )
        return {
            'contract': dict(self.rtc_o12_data_order_contract),
            'completed_iteration': completed_iteration,
            'next_global_iteration': next_iteration,
            'next_canonical_index_offset': (
                completed_iteration
                * self.rtc_o12_data_order_contract['batch_size']
            ),
        }

    def _training_state_dict(self, iteration, rng_states):
        state = {
            'checkpoint_type': 'train_kd_training_state',
            'checkpoint_version': 3,
            'student': unwrap_module(self.s_model).state_dict(),
            'criterion_cwd': unwrap_module(self.criterion_cwd).state_dict(),
            'criterion_fitnet': unwrap_module(self.criterion_fitnet).state_dict(),
            'D': None if self.D_model is None else unwrap_module(self.D_model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'D_optimizer': None if self.D_optimizer is None else self.D_optimizer.state_dict(),
            'iteration': int(iteration),
            'best_pred': float(self.best_pred),
            'rng_state': rng_states[0],
            'rng_state_by_rank': rng_states,
            'world_size': int(get_world_size()),
            'args': dict(vars(self.args)),
            'rtc': self._rtc_checkpoint_metadata(),
        }
        if self.rtc_o12_config is not None:
            state['checkpoint_version'] = 4
            state['rtc_o12'] = self._rtc_o12_checkpoint_metadata()
            state['rtc_o12_data_order'] = (
                self._rtc_o12_data_order_state(iteration)
            )
        return state

    def save_checkpoint(self, is_best=False, iteration=None, save_latest=True):
        iteration = self.current_iteration if iteration is None else int(iteration)
        local_rng_state = capture_rng_state()
        rng_states = all_gather(local_rng_state)
        if get_rank() != 0:
            return
        training_state = self._training_state_dict(iteration, rng_states)
        save_checkpoint(
            self.s_model,
            self.args,
            is_best=is_best,
            training_state=training_state,
            save_latest=save_latest,
        )

    @staticmethod
    def _assert_voc_aug_paths(root):
        jpeg_dir = os.path.join(root, 'JPEGImages')
        aug_dir = os.path.join(root, 'SegmentationClassAug')
        missing = []
        if not os.path.isdir(jpeg_dir):
            missing.append(jpeg_dir)
        if not os.path.isdir(aug_dir):
            missing.append(aug_dir)
        if missing:
            raise ValueError(f"VOC data root invalid: missing directories: {', '.join(missing)}. "
                             f"Please set --data to VOCAug root with JPEGImages/ and SegmentationClassAug/.")

    @staticmethod
    def _assert_voc_val_paths(root):
        jpeg_dir = os.path.join(root, 'JPEGImages')
        val_dir = os.path.join(root, 'SegmentationClass')
        missing = []
        if not os.path.isdir(jpeg_dir):
            missing.append(jpeg_dir)
        if not os.path.isdir(val_dir):
            missing.append(val_dir)
        if missing:
            raise ValueError(f"VOC validation requires: {', '.join(missing)}. "
                             f"If using VOCAug root, pass --skip-val or set --data to VOC2012 root for validation.")

    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr*((1-float(iter)/max_iter)**(power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def reduce_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    def aggregate_rtc_diagnostics(self, maps):
        """Compute exact global batch diagnostics instead of averaging rank quantiles."""
        valid = maps.valid_mask
        active = (
            ((maps.gate_reliable > 0) & (self.rtc_config.alpha_reliable > 0))
            | ((maps.gate_unreliable > 0) & (self.rtc_config.alpha_unreliable > 0))
        )
        active = (
            valid
            & active
            & ~maps.fallback_mask
            & ~maps.tie_mask
            & maps.finite_mask
        )
        local_payload = {
            'temperature': maps.temperature[valid].detach().float().cpu(),
            'reliability': maps.reliability[valid].detach().float().cpu(),
            'quantile': maps.reliability_quantile[valid].detach().float().cpu(),
            'gate_reliable': maps.gate_reliable[valid].detach().float().cpu(),
            'gate_unreliable': maps.gate_unreliable[valid].detach().float().cpu(),
            'residual_active': maps.target_residual[active].detach().float().cpu(),
            'fallback': maps.fallback_mask[valid].detach().float().cpu(),
            'tie': maps.tie_mask[valid].detach().float().cpu(),
            'finite': maps.finite_mask[valid].detach().float().cpu(),
            'shuffled': bool(maps.shuffled),
        }
        gathered = all_gather(local_payload)

        def concatenate(key):
            values = [item[key] for item in gathered if item[key].numel() > 0]
            return torch.cat(values) if values else torch.empty(0, dtype=torch.float32)

        temperature = concatenate('temperature')
        if temperature.numel() == 0:
            diagnostics = collect_rtc_diagnostics(
                maps,
                self.rtc_config,
                teacher_output_temperature=self.args.teacher_output_temp,
            )
            diagnostics['target_residual_active_count'] = 0.0
            return diagnostics
        reliability = concatenate('reliability')
        quantile = concatenate('quantile')
        gate_reliable = concatenate('gate_reliable')
        gate_unreliable = concatenate('gate_unreliable')
        residual = concatenate('residual_active')
        fallback = concatenate('fallback')
        tie = concatenate('tie')
        finite = concatenate('finite')
        shuffled = any(item['shuffled'] for item in gathered)
        temperature_quantiles = torch.quantile(
            temperature, torch.tensor([0.10, 0.50, 0.90, 0.95])
        )
        bisection_scale = math.ldexp(
            1.0, -(self.rtc_config.bisection_iterations + 1)
        )
        floating_tolerance = (
            4.0
            * torch.finfo(temperature.dtype).eps
            * max(1.0, self.rtc_config.unreliable_temperature)
        )
        reliable_tolerance = (
            (self.rtc_config.neutral_temperature
             - self.rtc_config.reliable_temperature)
            * bisection_scale
            + floating_tolerance
        )
        unreliable_tolerance = (
            (self.rtc_config.unreliable_temperature
             - self.rtc_config.neutral_temperature)
            * bisection_scale
            + floating_tolerance
        )
        neutral_tolerance = max(
            reliable_tolerance, unreliable_tolerance
        )
        residual_is_applicable = not shuffled
        if residual.numel() > 0 and residual_is_applicable:
            residual_p95 = float(torch.quantile(residual, 0.95))
            residual_mean = float(residual.mean())
            residual_max = float(residual.max())
        else:
            residual_p95 = 0.0
            residual_mean = 0.0
            residual_max = 0.0
        return {
            'valid_count': float(temperature.numel()),
            'reliability_mean': float(reliability.mean()),
            'quantile_mean': float(quantile.mean()),
            'reliable_coverage': float((gate_reliable > 0).float().mean()),
            'unreliable_coverage': float((gate_unreliable > 0).float().mean()),
            'reliable_gate_mean': float(gate_reliable.mean()),
            'unreliable_gate_mean': float(gate_unreliable.mean()),
            'temperature_mean': float(temperature.mean()),
            'temperature_harmonic_mean': float(1.0 / (1.0 / temperature).mean()),
            'temperature_q10': float(temperature_quantiles[0]),
            'temperature_q50': float(temperature_quantiles[1]),
            'temperature_q90': float(temperature_quantiles[2]),
            'temperature_p95': float(temperature_quantiles[3]),
            'temperature_reliable_endpoint_rate': float(
                (torch.abs(temperature - self.rtc_config.reliable_temperature)
                 <= reliable_tolerance).float().mean()
            ),
            'temperature_neutral_rate': float(
                (torch.abs(temperature - self.rtc_config.neutral_temperature)
                 <= neutral_tolerance).float().mean()
            ),
            'temperature_unreliable_endpoint_rate': float(
                (torch.abs(temperature - self.rtc_config.unreliable_temperature)
                 <= unreliable_tolerance).float().mean()
            ),
            'effective_teacher_temperature_mean': float(
                temperature.mean() * self.args.teacher_output_temp
            ),
            'target_residual_active_count': float(residual.numel()),
            'target_residual_mean': residual_mean,
            'target_residual_p95': residual_p95,
            'target_residual_max': residual_max,
            'residual_is_applicable': float(residual_is_applicable),
            'fallback_rate': float(fallback.mean()),
            'tie_rate': float(tie.mean()),
            'finite_rate': float(finite.mean()),
            'shuffled': float(shuffled),
        }

    @staticmethod
    def batch_class_stats(max_confidence, residual_variance, valid_mask, epsilon=1e-8):
        valid = valid_mask.float()
        denom = valid.sum(dim=(1, 2)).clamp_min(1.0)

        conf_mean = (max_confidence * valid).sum(dim=(1, 2)) / denom
        res_mean = (residual_variance * valid).sum(dim=(1, 2)) / denom

        conf_var = ((max_confidence - conf_mean.view(-1, 1, 1)) ** 2 * valid).sum(dim=(1, 2)) / denom
        res_var = ((residual_variance - res_mean.view(-1, 1, 1)) ** 2 * valid).sum(dim=(1, 2)) / denom

        means = torch.stack((conf_mean, res_mean), dim=1)
        vars = torch.stack((conf_var, res_var), dim=1)
        return means, vars

    @torch.no_grad()
    def get_covar_weight(self, pred_prob, valid_mask, epsilon=1e-8):
        num_classes = pred_prob.size(1)
        max_confidence, residual_variance = get_max_confidence_and_residual_variance(
            pred_prob, valid_mask, num_classes, epsilon=epsilon)

        means, vars = self.batch_class_stats(max_confidence, residual_variance, valid_mask, epsilon)

        conf_mean = means[:, 0].view(-1, 1, 1)
        res_mean = means[:, 1].view(-1, 1, 1)
        conf_var = vars[:, 0].view(-1, 1, 1)
        res_var = vars[:, 1].view(-1, 1, 1)

        conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + epsilon)
        res_z = (res_mean - residual_variance) / torch.sqrt(res_var + epsilon)

        weight_conf = torch.exp(- (conf_z ** 2) / self.args.covar_alpha)
        weight_res = torch.exp(- (res_z ** 2) / self.args.covar_alpha)

        weight = weight_conf * weight_res
        confident_mask = (conf_z > 0) | (res_z > 0)
        weight = torch.where(confident_mask, torch.ones_like(weight), weight)

        weight_mask = torch.where(valid_mask, weight, torch.zeros_like(weight))
        return weight_mask

    def covar_weighted_kd_loss(self, student_logits, teacher_logits, weight, epsilon=1e-8):
        temperature = self.args.kd_temperature
        s_log_prob = F.log_softmax(student_logits / temperature, dim=1)
        t_prob = F.softmax(teacher_logits / temperature, dim=1)

        kd_map = F.kl_div(s_log_prob, t_prob, reduction='none').sum(dim=1)
        weighted_loss = (kd_map * weight).sum() / weight.sum().clamp_min(epsilon)
        return weighted_loss * (temperature ** 2)

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(self.args.max_iterations))
        if self.start_iteration:
            logger.info('Continuing training from iteration {:d}'.format(self.start_iteration))

        self.s_model.train()
        for iteration, (images, targets, sample_names) in enumerate(
            self.train_loader, start=self.start_iteration + 1
        ):
            self.current_iteration = iteration

            images = images.to(self.device)
            targets = targets.long().to(self.device)
            
            with torch.no_grad():
                t_outputs = self.t_model(images)

            s_outputs = self.s_model(images)

            raw_teacher_logits = t_outputs[0]
            teacher_kd_logits = raw_teacher_logits
            if (
                self.rtc_o12_config is None
                and self.args.teacher_output_temp != 1.0
            ):
                teacher_kd_logits = teacher_kd_logits / self.args.teacher_output_temp

            covar_weight = None
            temperature_map = None
            reliability_map = None
            covar_valid_mask = None
            rtc_maps = None
            o12_temperature_map = None
            o12_valid_mask = None
            o12_reliability_quantile = None
            o12_teacher_target = None
            o12_terms = None
            o12_global_valid_count = None
            if self.rtc_o12_config is not None and self.args.lambda_kd != 0.:
                validate_rtc_o12_logit_shapes(
                    s_outputs[0], raw_teacher_logits
                )
                with torch.no_grad():
                    reference = compute_reference_reliability(
                        raw_teacher_logits,
                        targets != self.args.ignore_label,
                        assess_temperature=1.0,
                        coefficient_a=0.0,
                        reliability_mode='confidence',
                        epsilon=1e-8,
                    )
                    o12_valid_mask = reference.valid_mask
                    nonfinite_valid = o12_valid_mask & ~reference.finite_mask
                    if bool(nonfinite_valid.any().item()):
                        raise FloatingPointError(
                            'O1.2 teacher has non-finite native-valid pixels'
                        )
                    o12_reliability_quantile = self.rtc_o12_cdf.query(
                        reference.reliability
                    )
                    if not bool(
                        torch.isfinite(
                            o12_reliability_quantile[o12_valid_mask]
                        ).all().item()
                    ):
                        raise FloatingPointError(
                            'O1.2 CDF query produced non-finite quantiles'
                        )
                    dataset_indices = resolve_rtc_o12_dataset_indices(
                        sample_names, self.rtc_o12_dataset_index_by_name
                    )
                    o12_temperature_map, _ = (
                        build_rtc_o12_temperature_for_variant(
                            o12_reliability_quantile,
                            o12_valid_mask,
                            self.args.rtc_o12_variant,
                            self.rtc_o12_contract,
                            self.rtc_o12_config,
                            dataset_indices=dataset_indices,
                            global_iteration=iteration,
                        )
                    )
                    o12_teacher_target = build_o12_teacher_target(
                        raw_teacher_logits,
                        o12_temperature_map,
                        teacher_output_temperature=3.0,
                    )
            needs_masked_kd = self.args.kd_loss_mode == 'masked' or self.args.use_covar
            if needs_masked_kd and self.args.lambda_kd != 0.:
                with torch.no_grad():
                    valid_mask = (targets != self.args.ignore_label)
                    if self.args.use_covar and self.args.covar_temp_mode == 'rtc':
                        rtc_maps = build_rtc_temperature_map(
                            raw_teacher_logits,
                            teacher_kd_logits,
                            valid_mask,
                            self.rtc_cdf,
                            self.rtc_config,
                            shuffle=self.args.rtc_shuffle,
                            reverse_routing=self.args.rtc_reverse_routing,
                        )
                        temperature_map = rtc_maps.temperature
                        reliability_map = rtc_maps.reliability
                        covar_valid_mask = rtc_maps.valid_mask
                    elif self.args.use_covar and self.args.covar_temp_mode == 'newton':
                        temperature_map, reliability_map, covar_valid_mask, _, _ = \
                            newton_covar_temperature_map(
                                teacher_kd_logits,
                                valid_mask,
                                self.covar_newton_config,
                            )
                    elif self.args.use_covar:
                        covar_weight = self.get_covar_weight(
                            F.softmax(teacher_kd_logits, dim=1),
                            valid_mask,
                        )
                    else:
                        covar_valid_mask = F.interpolate(
                            valid_mask.float().unsqueeze(1),
                            size=s_outputs[0].shape[-2:],
                            mode='nearest',
                        ).squeeze(1) > 0.5
                        temperature_map = torch.full(
                            covar_valid_mask.shape,
                            float(self.args.kd_temperature),
                            device=s_outputs[0].device,
                            dtype=s_outputs[0].dtype,
                        )
            
            if self.args.aux:
                task_loss = self.criterion(s_outputs[0], targets) + 0.4 * self.criterion(s_outputs[1], targets)
            else:
                task_loss = self.criterion(s_outputs[0], targets)
            
            kd_loss = torch.tensor(0.).to(self.device)
            adv_G_loss = torch.tensor(0.).to(self.device)
            adv_D_loss = torch.tensor(0.).to(self.device)
            skd_loss = torch.tensor(0.).to(self.device)
            cwd_fea_loss = torch.tensor(0.).to(self.device)
            cwd_logit_loss = torch.tensor(0.).to(self.device)
            ifv_loss = torch.tensor(0.).to(self.device)
            fitnet_loss = torch.tensor(0.).to(self.device)
            at_loss = torch.tensor(0.).to(self.device)
            psd_loss = torch.tensor(0.).to(self.device)
            csd_loss = torch.tensor(0.).to(self.device)
            

            if self.args.lambda_adv != 0.:
                adv_G_loss = self.args.lambda_adv * self.criterion_adv_for_G(self.D_model(s_outputs[0]))

            if self.args.lambda_d != 0.:
                adv_D_loss = self.args.lambda_d * (self.criterion_adv(
                    self.D_model(s_outputs[0].detach()),
                    self.D_model(t_outputs[0].detach())))
            
            if self.args.lambda_kd != 0.:
                if self.rtc_o12_config is not None:
                    o12_terms = compute_o12_masked_kl_terms(
                        s_outputs[0],
                        o12_teacher_target,
                        o12_valid_mask,
                    )
                    o12_global_valid_count = o12_terms.valid_count.clone()
                    if get_world_size() > 1:
                        dist.all_reduce(
                            o12_global_valid_count, op=dist.ReduceOp.SUM
                        )
                    kd_loss = self.args.lambda_kd * normalize_o12_ddp_loss(
                        o12_terms.kl_sum,
                        o12_global_valid_count,
                        get_world_size(),
                    )
                elif self.args.kd_loss_mode == 'teacher_only':
                    kd_loss = self.args.lambda_kd * teacher_target_kd_loss(
                        s_outputs[0],
                        raw_teacher_logits,
                        self.args.kd_temperature,
                        targets != self.args.ignore_label,
                    )
                elif rtc_maps is not None:
                    kd_loss = self.args.lambda_kd * masked_temperature_kd_loss(
                        s_outputs[0],
                        teacher_kd_logits,
                        temperature_map,
                        covar_valid_mask,
                        temperature_power=self.rtc_config.kd_temperature_power,
                    )
                elif self.args.kd_loss_mode == 'masked' and not self.args.use_covar:
                    kd_loss = self.args.lambda_kd * masked_temperature_kd_loss(
                        s_outputs[0],
                        teacher_kd_logits,
                        temperature_map,
                        covar_valid_mask,
                        temperature_power=self.args.covar_kd_temp_power,
                    )
                elif self.args.use_covar and temperature_map is not None:
                    kd_loss = self.args.lambda_kd * covar_temperature_kd_loss(
                        s_outputs[0],
                        teacher_kd_logits,
                        temperature_map,
                        covar_valid_mask,
                        temperature_power=self.covar_newton_config.kd_temperature_power,
                    )
                elif self.args.use_covar and covar_weight is not None:
                    kd_loss = self.args.lambda_kd * self.covar_weighted_kd_loss(
                        s_outputs[0], teacher_kd_logits, covar_weight)
                else:
                    kd_loss = self.args.lambda_kd * self.criterion_kd(s_outputs[0], teacher_kd_logits)
            if self.args.lambda_skd != 0:
                skd_loss = self.args.lambda_skd * self.criterion_skd(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_cwd_fea != 0:
                cwd_fea_loss = self.args.lambda_cwd_fea * self.criterion_cwd(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_cwd_logit != 0:
                cwd_logit_loss = self.args.lambda_cwd_logit * self.criterion_cwd(s_outputs[0], t_outputs[0])
            if self.args.lambda_ifv != 0:
                ifv_loss = self.args.lambda_ifv * self.criterion_ifv(s_outputs[-1], t_outputs[-1], targets)
            if self.args.lambda_fitnet != 0:
                fitnet_loss = self.args.lambda_fitnet * self.criterion_fitnet(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_at != 0:
                at_loss = self.args.lambda_at * self.criterion_at(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_psd != 0. and self.args.lambda_csd != 0.:  
                feat_s_list = [s_outputs[-2], s_outputs[-1], s_outputs[0]]
                feat_t_list = [t_outputs[-2], t_outputs[-1], t_outputs[0]]
                psd_loss, csd_loss = self.criterion_dsd(feat_s_list, feat_t_list)
                psd_loss = self.args.lambda_psd * psd_loss
                csd_loss = self.args.lambda_csd * csd_loss

            losses = task_loss + kd_loss + adv_G_loss + \
                        skd_loss + cwd_fea_loss + cwd_logit_loss +\
                        ifv_loss + at_loss + fitnet_loss +\
                        psd_loss + csd_loss 
            D_losses = adv_D_loss
            if (
                self.rtc_o12_config is not None
                or rtc_maps is not None
                or self.args.kd_loss_mode == 'masked'
                or self.args.kd_loss_mode == 'teacher_only'
            ):
                local_nonfinite = not bool(torch.isfinite(losses).item())
                if self.use_adv:
                    local_nonfinite = (
                        local_nonfinite
                        or not bool(torch.isfinite(D_losses).item())
                    )
                failure_flags = torch.zeros(3, device=self.device)
                failure_flags[0] = float(local_nonfinite)
                if rtc_maps is not None and not rtc_maps.shuffled:
                    reliable_active = (
                        rtc_maps.valid_mask & (rtc_maps.gate_reliable > 0)
                    )
                    unreliable_active = (
                        rtc_maps.valid_mask & (rtc_maps.gate_unreliable > 0)
                    )
                    failure_flags[1] = float(
                        bool(reliable_active.any().item())
                        and bool(
                            (rtc_maps.temperature[reliable_active]
                             > self.rtc_config.neutral_temperature + 1e-5)
                            .any()
                            .item()
                        )
                    )
                    failure_flags[2] = float(
                        bool(unreliable_active.any().item())
                        and bool(
                            (rtc_maps.temperature[unreliable_active]
                             < self.rtc_config.neutral_temperature - 1e-5)
                            .any()
                            .item()
                        )
                    )
                if get_world_size() > 1:
                    dist.all_reduce(failure_flags, op=dist.ReduceOp.MAX)
                if bool((failure_flags[0] > 0).item()):
                    raise FloatingPointError(
                        'Non-finite generator or discriminator loss in '
                        'RTC/masked KD path on at least one rank'
                    )
                if bool((failure_flags[1] > 0).item()):
                    raise AssertionError(
                        'RTC reliable branch produced T > T0 on at least one rank'
                    )
                if bool((failure_flags[2] > 0).item()):
                    raise AssertionError(
                        'RTC unreliable branch produced T < T0 on at least one rank'
                    )

            o12_diagnostics = None
            if (
                self.rtc_o12_config is not None
                and iteration % log_per_iters == 0
            ):
                valid_count = int(o12_global_valid_count.item())
                denominator = float(max(valid_count, 1))
                kl_mean = float(o12_terms.kl_sum.detach().item()) / denominator
                cross_entropy_mean = (
                    float(o12_terms.cross_entropy_sum.detach().item())
                    / denominator
                )
                teacher_entropy_mean = (
                    float(o12_terms.teacher_entropy_sum.detach().item())
                    / denominator
                )
                if abs(
                    kl_mean
                    - (cross_entropy_mean - teacher_entropy_mean)
                ) > 1e-5:
                    raise AssertionError(
                        'O1.2 KL != cross_entropy - teacher_entropy'
                    )
                o12_kd_gradient = torch.autograd.grad(
                    kd_loss,
                    s_outputs[0],
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                if not bool(torch.isfinite(o12_kd_gradient).all().item()):
                    raise FloatingPointError(
                        'O1.2 KD-only student-logit gradient is non-finite'
                    )
                o12_diagnostics = {
                    'valid_count': valid_count,
                    'kl_mean': kl_mean,
                    'cross_entropy_mean': cross_entropy_mean,
                    'teacher_entropy_mean': teacher_entropy_mean,
                    'student_logit_grad_l2': float(
                        torch.linalg.vector_norm(
                            o12_kd_gradient.detach().float()
                        ).item()
                    ),
                }

            lr = self.adjust_lr(
                base_lr=self.args.lr,
                iter=iteration - 1,
                max_iter=self.args.max_iterations,
                power=0.9,
            )
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if self.use_adv:
                self.D_optimizer.zero_grad()
                D_losses.backward()
                self.D_optimizer.step()

            task_loss_reduced = self.reduce_mean_tensor(task_loss)
            kd_loss_reduced = self.reduce_mean_tensor(kd_loss)
            adv_G_loss_reduced = self.reduce_mean_tensor(adv_G_loss)
            skd_loss_reduced = self.reduce_mean_tensor(skd_loss)
            cwd_fea_loss_reduced = self.reduce_mean_tensor(cwd_fea_loss)
            cwd_logit_loss_reduced = self.reduce_mean_tensor(cwd_logit_loss)
            ifv_loss_reduced = self.reduce_mean_tensor(ifv_loss)
            at_loss_reduced = self.reduce_mean_tensor(at_loss)
            fitnet_loss_reduced = self.reduce_mean_tensor(fitnet_loss)
            psd_loss_reduced = self.reduce_mean_tensor(psd_loss)
            csd_loss_reduced = self.reduce_mean_tensor(csd_loss)
            
            
            D_losses_reduced = self.reduce_mean_tensor(D_losses)
            elapsed_iterations = max(iteration - self.start_iteration, 1)
            eta_seconds = (
                (time.time() - start_time) / elapsed_iterations
            ) * (self.args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            rtc_diagnostics = None
            if iteration % log_per_iters == 0 and rtc_maps is not None:
                rtc_diagnostics = self.aggregate_rtc_diagnostics(rtc_maps)

            if iteration % log_per_iters == 0 and save_to_disk:
                log_message = (
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || KD Loss: {:.4f}" \
                    "|| Adv_G Loss: {:.4f} || Adv_D Loss: {:.4f}" \
                    "|| skd_loss: {:.4f} || cwd_fea_loss: {:.4f} || cwd_logit_loss: {:.4f} " \
                        "|| ifv_loss: {:.4f} || at_loss: {:.4f} || fitnet_loss: {:.4f} " \
                        "|| psd_loss: {:.4f} || csd_loss: {:.4f} " \
                        "|| Cost Time: {} || Estimated Time: {}".format(
                        iteration, self.args.max_iterations, self.optimizer.param_groups[0]['lr'],
                        task_loss_reduced.item(),
                        kd_loss_reduced.item(), 
                        adv_G_loss_reduced.item(),
                        D_losses_reduced.item(), 
                        skd_loss_reduced.item(),
                        cwd_fea_loss_reduced.item(),
                        cwd_logit_loss_reduced.item(),
                        ifv_loss_reduced.item(),
                        at_loss_reduced.item(),
                        fitnet_loss_reduced.item(),
                        psd_loss_reduced.item(),
                        csd_loss_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), 
                        eta_string))
                if o12_diagnostics is not None:
                    log_message += (
                        ' || O1.2 variant: {variant}'
                        ' || O1.2 branch KL mean: {kl_mean:.8f}'
                        ' || O1.2 cross-entropy mean: {cross_entropy_mean:.8f}'
                        ' || O1.2 teacher entropy mean: {teacher_entropy_mean:.8f}'
                        ' || O1.2 KD-only student-logit grad L2: '
                        '{student_logit_grad_l2:.8f}'
                        ' || O1.2 valid pixels: {valid_count:d}'.format(
                            variant=self.args.rtc_o12_variant,
                            **o12_diagnostics,
                        )
                    )
                if rtc_diagnostics is not None:
                    log_message += (
                        " || RTC T mean/hmean/q10/q50/q90/p95: "
                        "{temperature_mean:.4f}/{temperature_harmonic_mean:.4f}/"
                        "{temperature_q10:.4f}/{temperature_q50:.4f}/"
                        "{temperature_q90:.4f}/{temperature_p95:.4f}"
                        " || RTC endpoints R/T0/U: "
                        "{temperature_reliable_endpoint_rate:.4f}/"
                        "{temperature_neutral_rate:.4f}/"
                        "{temperature_unreliable_endpoint_rate:.4f}"
                        " || RTC R/U cov: {reliable_coverage:.4f}/{unreliable_coverage:.4f}"
                        " || RTC gate R/U: {reliable_gate_mean:.4f}/{unreliable_gate_mean:.4f}"
                        " || RTC residual applicable/shuffled: "
                        "{residual_is_applicable:.0f}/{shuffled:.0f}"
                        " || RTC finite/fallback/tie: "
                        "{finite_rate:.6f}/{fallback_rate:.6f}/{tie_rate:.6f}"
                        " || RTC effective teacher T mean: "
                        "{effective_teacher_temperature_mean:.4f}"
                        " || RTC valid pixels: {valid_count:.0f}"
                        " || CDF sha256: {cdf}".format(
                            cdf=self.rtc_cdf.checksum_sha256[:12],
                            **rtc_diagnostics,
                        )
                    )
                    if rtc_diagnostics['residual_is_applicable'] >= 0.5:
                        log_message += (
                            " || RTC residual active/mean/p95/max: "
                            "{target_residual_active_count:.0f}/"
                            "{target_residual_mean:.6f}/{target_residual_p95:.6f}/"
                            "{target_residual_max:.6f}".format(**rtc_diagnostics)
                        )
                    else:
                        log_message += " || RTC residual: N/A (shuffled map)"
                elif temperature_map is not None and reliability_map is not None:
                    valid_temperatures = temperature_map[covar_valid_mask]
                    valid_reliability = reliability_map[covar_valid_mask]
                    if valid_temperatures.numel() > 0:
                        log_message += (
                            " || CoVar T mean/min/max: {:.4f}/{:.4f}/{:.4f} || r_mean: {:.4f}".format(
                                valid_temperatures.mean().item(),
                                valid_temperatures.min().item(),
                                valid_temperatures.max().item(),
                                valid_reliability.mean().item(),
                            )
                        )
                elif temperature_map is not None:
                    valid_temperatures = temperature_map[covar_valid_mask]
                    if valid_temperatures.numel() > 0:
                        log_message += (
                            " || Masked KD T mean/min/max: {:.4f}/{:.4f}/{:.4f} || gamma: {:.1f}".format(
                                valid_temperatures.mean().item(),
                                valid_temperatures.min().item(),
                                valid_temperatures.max().item(),
                                self.args.covar_kd_temp_power,
                            )
                        )
                elif self.args.kd_loss_mode == 'teacher_only':
                    log_message += (
                        " || Teacher-only target T: {:.4f}"
                        " || Student T: 1.0000 || T^2 compensation: off"
                    ).format(self.args.kd_temperature)
                elif self.args.teacher_output_temp != 1.0:
                    log_message += " || Teacher output T: {:.4f}".format(self.args.teacher_output_temp)
                logger.info(log_message)

            if iteration % save_per_iters == 0:
                self.save_checkpoint(is_best=False, iteration=iteration)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(step=iteration)
                self.s_model.train()

        self.save_checkpoint(is_best=False, iteration=self.current_iteration)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        completed_iterations = max(self.current_iteration - self.start_iteration, 1)
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / completed_iterations))


    def validation(self, step=None):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        empty_accelerator_cache(self.device)  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(outputs[0], (H, W), mode='bilinear', align_corners=True)

            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info(format_sample_validation_log(i + 1, pixAcc, mIoU))
        
        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).to(self.device)
            sum_total_label = torch.tensor(self.metric.total_label).to(self.device)
            sum_total_inter = torch.tensor(self.metric.total_inter).to(self.device)
            sum_total_union = torch.tensor(self.metric.total_union).to(self.device)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label) 
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            logger.info(format_overall_validation_log(
                pixAcc.item() * 100, mIoU * 100
            ))

        new_pred = float(mIoU)
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if is_best:
            self.save_checkpoint(is_best=True, iteration=step, save_latest=False)
        synchronize()


def save_npy(array, name):
    """Save Checkpoint"""
    if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
        directory = os.path.expanduser(args.save_dir)
        np.save(os.path.join(directory, name), array)


def save_checkpoint(
    model,
    args,
    is_best=False,
    training_state=None,
    save_latest=True,
):
    """Save legacy model weights plus an optional complete training state."""
    directory = os.path.expanduser(args.save_dir)
    os.makedirs(directory, exist_ok=True)
    filename = 'kd_{}_{}_{}.pth'.format(
        args.student_model, args.student_backbone, args.dataset
    )
    filename = os.path.join(directory, filename)
    model_state = unwrap_module(model).state_dict()

    if save_latest:
        torch.save(model_state, filename)
        if training_state is not None:
            torch.save(training_state, os.path.join(directory, 'training_state_latest.pth'))
            iteration = int(training_state.get('iteration', 0))
            if iteration in set(getattr(args, 'keep_checkpoint_iters', [])):
                stem = 'kd_{}_{}_{}_iter{:06d}'.format(
                    args.student_model, args.student_backbone, args.dataset,
                    iteration,
                )
                torch.save(model_state, os.path.join(directory, stem + '.pth'))
                torch.save(
                    training_state,
                    os.path.join(directory, 'training_state_iter{:06d}.pth'.format(iteration)),
                )

    if is_best:
        best_filename = 'kd_{}_{}_{}_best_model.pth'.format(
            args.student_model, args.student_backbone, args.dataset
        )
        best_filename = os.path.join(directory, best_filename)
        torch.save(model_state, best_filename)
        if training_state is not None:
            torch.save(training_state, os.path.join(directory, 'training_state_best.pth'))


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if args.no_cuda:
        args.device_type = 'cpu'
    args.device = resolve_device_type(args.device_type)
    seed_everything(args.seed, rank=int(os.environ.get("RANK", 0)))
    if args.device == "cuda":
        cudnn.benchmark = False
        set_accelerator_device(args.device, args.local_rank)
    elif args.device == "npu":
        set_accelerator_device(args.device, args.local_rank)
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        backend = "hccl" if args.device == "npu" else "nccl"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        synchronize()

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.student_model, args.teacher_backbone, args.student_backbone, args.dataset))
    logger.info("Using {} process(es) on device {}".format(num_gpus, args.device))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    empty_accelerator_cache(torch.device(args.device))
