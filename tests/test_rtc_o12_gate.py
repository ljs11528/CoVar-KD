import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from utils.rtc_o12_calibration import (
    build_o12_temperature_map,
    solve_o12_budget_parameter,
)


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load_module(
    'check_rtc_o12_gate',
    'scripts/diagnostics/check_rtc_o12_gate.py',
)
DIAG = load_module(
    'diagnose_rtc_o12_budget',
    'scripts/diagnostics/diagnose_rtc_o12_budget.py',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_info(path):
    values = np.load(path, allow_pickle=False)
    return {
        'path': str(Path(path).resolve()),
        'sha256': sha256(path),
        'dtype': 'float32',
        'count': int(values.size),
        'semantics': 'test exact u',
    }


def runtime_fields(
    process_id=1234, requested_npu_index=0, current_npu_index=None
):
    if current_npu_index is None:
        current_npu_index = requested_npu_index
    return {
        'started_at_utc': '2026-07-13T00:00:00+00:00',
        'ended_at_utc': '2026-07-13T00:00:02+00:00',
        'elapsed_seconds': 2.0,
        'runtime_identity': {
            'process_id': process_id,
            'world_size': 1,
            'rank': 0,
            'local_rank': 0,
            'requested_device': f'npu:{requested_npu_index}',
            'requested_npu_index': requested_npu_index,
            'resolved_device': f'npu:{requested_npu_index}',
            'device_type': 'npu',
            'npu_current_device': current_npu_index,
            'npu_device_name': 'Ascend synthetic test NPU',
            'ascend_rt_visible_devices': None,
        },
    }

def compact(values):
    unique, counts = np.unique(values.astype(np.float32), return_counts=True)
    return unique, counts.astype(np.int64)


def solution_payload(values):
    solved = solve_o12_budget_parameter(torch.from_numpy(values.copy()))
    payload = solved.to_dict()
    payload.update({
        'feasible': True,
        'iterations': solved.bisection_iterations,
        'population': solved.valid_count,
        'mean': solved.arithmetic_mean,
        'harmonic_mean': solved.harmonic_mean,
        'mean_residual': solved.residual,
        'theoretical_high_risk_endpoint': float(np.exp(solved.b)),
        'solver_input': 'exact float32 risk cache; no histogram approximation',
    })
    return payload


def branch_scalars(values, b):
    unique, counts = compact(values)
    full = GATE.temperature_statistics(
        unique, counts, GATE.EXPECTED_CONFIG['a'], b
    )
    unreliable = GATE.temperature_statistics(unique, counts, 0.0, b)
    return {
        'unreliable_only': {
            'arithmetic': unreliable['mean'],
            'harmonic': unreliable['harmonic_mean'],
        },
        'full_budgeted': {
            'arithmetic': full['mean'],
            'harmonic': full['harmonic_mean'],
        },
    }


def make_parameters(path, values):
    solution = solution_payload(values)
    return {
        'schema_version': 1,
        'phase': 'O1.2',
        'artifact_kind': 'o12_budget_parameters',
        'split': 'train',
        **runtime_fields(1001),
        'formal_full_run': True,
        'configuration': copy.deepcopy(GATE.EXPECTED_CONFIG),
        'configuration_fingerprint': GATE.canonical_fingerprint(
            GATE.EXPECTED_CONFIG
        ),
        'source_sha256': GATE.source_sha256(),
        'scan_protocol': GATE.expected_scan_protocol('solve', 'train'),
        'scan_protocol_fingerprint': GATE.canonical_fingerprint(
            GATE.expected_scan_protocol('solve', 'train')
        ),
        'cdf_path': str(GATE.EXPECTED_CDF_PATH),
        'cdf_sha256': GATE.EXPECTED_CDF_SHA256,
        'o11_joint_gate_pass': True,
        'teacher_sha256': GATE.EXPECTED_TEACHER_SHA256,
        'list_sha256': GATE.EXPECTED_LIST_SHA256['train'],
        'dataset_size': 1,
        'valid_native_pixels': int(values.size),
        'finite_native_pixels': int(values.size),
        'nonfinite_native_pixels': 0,
        'risk_cache': cache_info(path),
        'solution': solution,
        'branch_scalar_temperatures': branch_scalars(values, solution['b']),
        'arithmetic_matched_scalar_temperature': solution['mean'],
        'harmonic_matched_scalar_temperature': solution['harmonic_mean'],
    }


def diagnostic_row(mask, wrong, temperature):
    count = int(mask.sum())
    return {
        'count': count,
        'teacher_wrong_count': int((mask & wrong).sum()),
        'temperature_mean': (
            float(temperature[mask].astype(np.float64).mean()) if count else None
        ),
        'c_base_mean': .5 if count else None,
        'c_target_mean': .5 if count else None,
        'entropy_base_mean': 1.0 if count else None,
        'entropy_target_mean': 1.0 if count else None,
    }


def make_split_report(split, path, values, parameters, parameters_sha):
    unique, counts = compact(values)
    solution = parameters['solution']
    stats = GATE.temperature_statistics(
        unique, counts, solution['a'], solution['b']
    )
    effective = GATE.temperature_statistics(
        unique, counts, solution['a'], solution['b'], scale=3.0
    )
    effective['semantics'] = 'T_effective=T_out*T with T_out=3.0'
    wrong = (values >= np.float32(.9)) | (values == np.float32(.8))
    temperature_values = build_o12_temperature_map(
        torch.from_numpy(values.copy()), b=solution['b'],
        branch='full_budgeted',
    ).numpy()
    regions = {}
    for name, mask in (
        ('reliable', values < .6),
        ('neutral', (values >= .6) & (values <= .8)),
        ('unreliable', values > .8),
    ):
        regions[name] = diagnostic_row(mask, wrong, temperature_values)
    bins = []
    indices = np.minimum(np.floor(values * 10).astype(np.int64), 9)
    for index in range(10):
        mask = indices == index
        bins.append({
            'bin': index, **diagnostic_row(mask, wrong, temperature_values)
        })
    total = int(values.size)
    total_wrong = int(wrong.sum())
    class_rows = [
        {'class_id': index, **diagnostic_row(
            np.ones(total, dtype=bool)
            if index == 0 else np.zeros(total, dtype=bool),
            wrong, temperature_values,
        )}
        for index in range(21)
    ]
    binary_sections = {
        'foreground_background': {
            'background': diagnostic_row(
                np.ones(total, dtype=bool), wrong, temperature_values
            ),
            'foreground': diagnostic_row(
                np.zeros(total, dtype=bool), wrong, temperature_values
            ),
        },
        'boundary_interior': {
            'boundary': diagnostic_row(
                np.zeros(total, dtype=bool), wrong, temperature_values
            ),
            'interior': diagnostic_row(
                np.ones(total, dtype=bool), wrong, temperature_values
            ),
        },
        'small_object': {
            'small_object': diagnostic_row(
                np.zeros(total, dtype=bool), wrong, temperature_values
            ),
            'not_small_object': diagnostic_row(
                np.ones(total, dtype=bool), wrong, temperature_values
            ),
        },
    }
    high = regions['unreliable']
    global_rate = total_wrong / total
    high_rate = high['teacher_wrong_count'] / high['count']
    rates = [
        row['teacher_wrong_count'] / row['count'] if row['count'] else None
        for row in bins
    ]
    pairs = [
        (rates[i], rates[j])
        for i in range(10) for j in range(i + 1, 10)
        if rates[i] is not None and rates[j] is not None
    ]
    evidence = {
        'global_teacher_wrong_rate': global_rate,
        'high_risk_coverage': high['count'] / total,
        'high_risk_teacher_wrong_precision': high_rate,
        'high_risk_teacher_wrong_recall': high['teacher_wrong_count'] / total_wrong,
        'high_risk_enrichment': high_rate / global_rate,
        'risk_quantile_pairwise_monotonic_agreement': (
            sum(a <= b for a, b in pairs) / len(pairs)
        ),
    }
    return {
        'schema_version': 1,
        'phase': 'O1.2',
        'artifact_kind': 'o12_budget_evaluation',
        'split': split,
        'formal_full_run': True,
        'processed_images': 1,
        **runtime_fields(1002),
        'dataset_size': 1,
        'valid_native_pixels': total,
        'finite_native_pixels': total,
        'nonfinite_native_pixels': 0,
        'teacher_wrong_pixels': total_wrong,
        'configuration': copy.deepcopy(GATE.EXPECTED_CONFIG),
        'configuration_fingerprint': GATE.canonical_fingerprint(
            GATE.EXPECTED_CONFIG
        ),
        'source_sha256': GATE.source_sha256(),
        'scan_protocol': GATE.expected_scan_protocol('evaluate', split),
        'scan_protocol_fingerprint': GATE.canonical_fingerprint(
            GATE.expected_scan_protocol('evaluate', split)
        ),
        'cdf_path': str(GATE.EXPECTED_CDF_PATH),
        'cdf_sha256': GATE.EXPECTED_CDF_SHA256,
        'teacher_sha256': GATE.EXPECTED_TEACHER_SHA256,
        'list_sha256': GATE.EXPECTED_LIST_SHA256[split],
        'parameters_sha256': parameters_sha,
        'parameters_refit': False,
        'a': solution['a'],
        'b': solution['b'],
        'risk_cache': cache_info(path),
        'temperature': stats,
        'effective_temperature': effective,
        'risk_regions': regions,
        'boundary_counts': {
            'u_eq_0p6': int((values == np.float32(.6)).sum()),
            'u_eq_0p8': int((values == np.float32(.8)).sum()),
            'u_eq_0p6_teacher_wrong': int(
                ((values == np.float32(.6)) & wrong).sum()
            ),
            'u_eq_0p8_teacher_wrong': int(
                ((values == np.float32(.8)) & wrong).sum()
            ),
        },
        'risk_quantile_bins': bins,
        'target_by_risk_decile': copy.deepcopy(bins),
        'risk_evidence': evidence,
        'stratified_diagnostics': {'class': class_rows, **binary_sections},
        'violations': {key: 0 for key in GATE.VIOLATION_KEYS},
        'target_numeric_maxima': {
            'neutral_target_max_abs_error': 0.0,
            'student_softmax_max_abs_error': 0.0,
            'teacher_target_probe_max_abs_difference': .01,
        },
        'closure_checks': {'all_exact_partitions': True},
        'teacher_target_contract': {
            'spatial_temperature_applies_to': 'teacher_target_only',
            'student_temperature': 1.0,
            'teacher_target_detached': True,
            'spatial_temperature_loss_power': 'not_applicable',
        },
    }


def make_o11_reference(report):
    bins = report['risk_quantile_bins']
    boundary = report['boundary_counts']
    high = report['risk_regions']['unreliable']
    total_count = report['valid_native_pixels']
    total_wrong = report['teacher_wrong_pixels']
    low_count = total_count - high['count'] - boundary['u_eq_0p8']
    low_wrong = (
        total_wrong - high['teacher_wrong_count']
        - boundary['u_eq_0p8_teacher_wrong']
    )
    return {
        'phase': 'O1.1',
        'split': report['split'],
        'risk_quantile_bins': [
            {
                'count': row['count'],
                'teacher_wrong_count': row['teacher_wrong_count'],
            }
            for row in bins
        ],
        'risk_routing_counts': {
            'semantics': 'exact native-valid micro counts; high u>q, low u<q, boundary u==q',
            'valid_native_pixels': total_count,
            'teacher_wrong_pixels': total_wrong,
            'high_risk_pixels': high['count'],
            'high_risk_wrong_pixels': high['teacher_wrong_count'],
            'low_risk_pixels': low_count,
            'low_risk_wrong_pixels': low_wrong,
            'boundary_pixels': boundary['u_eq_0p8'],
            'boundary_wrong_pixels': boundary['u_eq_0p8_teacher_wrong'],
        },
    }

class ExactSolverTest(unittest.TestCase):

    def test_checker_exact_solver_matches_core_within_1e8(self):
        values = torch.linspace(0, 1, 4097).repeat(2).numpy().astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'u.npy'
            np.save(path, values, allow_pickle=False)
            core = solve_o12_budget_parameter(torch.from_numpy(values.copy()))
            checker = GATE.solve_budget_exact_cache(path)
        self.assertLessEqual(abs(core.b - checker['b']), 1e-8)
        self.assertGreaterEqual(np.exp(checker['b']), 1.25)

    def test_unreliable_and_full_scalar_statistics_are_distinct(self):
        values = torch.linspace(0, 1, 4097).numpy().astype(np.float32)
        solved = solve_o12_budget_parameter(torch.from_numpy(values.copy()))
        scalars = branch_scalars(values, solved.b)
        self.assertNotAlmostEqual(
            scalars['unreliable_only']['arithmetic'],
            scalars['full_budgeted']['arithmetic'],
            places=5,
        )


class ParameterGateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.values = torch.linspace(0, 1, 4097).numpy().astype(np.float32)
        self.cache = Path(self.directory.name) / 'u.npy'
        np.save(self.cache, self.values, allow_pickle=False)
        self.parameters = make_parameters(self.cache, self.values)

    def tearDown(self):
        self.directory.cleanup()

    def test_parameter_gate_passes_exact_synthetic_population(self):
        with mock.patch.dict(
            GATE.EXPECTED_NATIVE_VALID, {'train': self.values.size}
        ), mock.patch.dict(GATE.EXPECTED_DATASET_SIZE, {'train': 1}):
            result = GATE.evaluate_parameters(
                self.parameters, GATE.source_sha256()
            )
        failed = [key for key, value in result['checks'].items() if not value]
        self.assertEqual(failed, [])

    def test_wrong_unreliable_scalar_is_rejected(self):
        payload = copy.deepcopy(self.parameters)
        payload['branch_scalar_temperatures']['unreliable_only']['arithmetic'] += .01
        with mock.patch.dict(
            GATE.EXPECTED_NATIVE_VALID, {'train': self.values.size}
        ), mock.patch.dict(GATE.EXPECTED_DATASET_SIZE, {'train': 1}):
            result = GATE.evaluate_parameters(payload, GATE.source_sha256())
        self.assertFalse(
            result['checks']['unreliable_only_arithmetic_scalar_matches']
        )

    def test_smoke_parameters_cannot_pass_formal_gate(self):
        payload = copy.deepcopy(self.parameters)
        payload['formal_full_run'] = False
        with mock.patch.dict(
            GATE.EXPECTED_NATIVE_VALID, {'train': self.values.size}
        ), mock.patch.dict(GATE.EXPECTED_DATASET_SIZE, {'train': 1}):
            result = GATE.evaluate_parameters(payload, GATE.source_sha256())
        self.assertFalse(result['pass'])
        self.assertFalse(result['checks']['formal_full_run'])


    def test_runtime_world_size_tamper_is_rejected(self):
        payload = copy.deepcopy(self.parameters)
        payload['runtime_identity']['world_size'] = 2
        with mock.patch.dict(
            GATE.EXPECTED_NATIVE_VALID, {'train': self.values.size}
        ), mock.patch.dict(GATE.EXPECTED_DATASET_SIZE, {'train': 1}):
            result = GATE.evaluate_parameters(payload, GATE.source_sha256())
        self.assertFalse(
            result['checks']['runtime_single_process_rank_zero']
        )


    def test_requested_npu_one_with_current_zero_is_rejected(self):
        payload = copy.deepcopy(self.parameters)
        payload['runtime_identity'].update({
            'requested_device': 'npu:1',
            'requested_npu_index': 1,
            'resolved_device': 'npu:1',
            'npu_current_device': 0,
        })
        with mock.patch.dict(
            GATE.EXPECTED_NATIVE_VALID, {'train': self.values.size}
        ), mock.patch.dict(GATE.EXPECTED_DATASET_SIZE, {'train': 1}):
            result = GATE.evaluate_parameters(payload, GATE.source_sha256())
        self.assertFalse(
            result['checks']['runtime_requested_npu_index_matches_current']
        )

    def test_solve_zero_train_zero_val_one_passes_joint_runtime_gate(self):
        solve = runtime_fields(2001, requested_npu_index=0)
        train = runtime_fields(2002, requested_npu_index=0)
        val = runtime_fields(2003, requested_npu_index=1)
        provenance = GATE.joint_runtime_device_provenance(
            solve, train, val
        )
        self.assertTrue(
            provenance['all_artifacts_single_process_single_npu']
        )
        self.assertTrue(all(provenance['per_artifact_valid'].values()))
        self.assertFalse(provenance['same_actual_npu'])
        self.assertFalse(provenance['same_actual_npu_is_gate'])
    def test_scan_protocol_tamper_is_rejected(self):
        payload = copy.deepcopy(self.parameters)
        payload['scan_protocol']['workers'] = 1
        with mock.patch.dict(
            GATE.EXPECTED_NATIVE_VALID, {'train': self.values.size}
        ), mock.patch.dict(GATE.EXPECTED_DATASET_SIZE, {'train': 1}):
            result = GATE.evaluate_parameters(payload, GATE.source_sha256())
        self.assertFalse(result['checks']['scan_protocol_exact'])

class SplitGateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.values = np.concatenate((
            torch.linspace(0, 1, 4097).numpy().astype(np.float32),
            np.asarray([.6, .7, .8, .9], dtype=np.float32),
        ))
        self.cache = Path(self.directory.name) / 'u.npy'
        np.save(self.cache, self.values, allow_pickle=False)
        self.parameters_path = Path(self.directory.name) / 'parameters.json'
        self.parameters = make_parameters(self.cache, self.values)
        self.parameters_path.write_text(
            json.dumps(self.parameters, allow_nan=False), encoding='utf-8'
        )
        self.report = make_split_report(
            'train', self.cache, self.values, self.parameters,
            sha256(self.parameters_path),
        )

        self.o11_reference = make_o11_reference(self.report)
    def tearDown(self):
        self.directory.cleanup()

    def evaluate(self, report):
        with mock.patch.dict(
            GATE.EXPECTED_NATIVE_VALID, {'train': self.values.size}
        ), mock.patch.dict(
            GATE.EXPECTED_DATASET_SIZE, {'train': 1}
        ), mock.patch.object(
            GATE, 'load_o11_reference', return_value=self.o11_reference
        ):
            return GATE.evaluate_split(
                report, 'train', self.parameters,
                sha256(self.parameters_path), GATE.source_sha256(),
            )

    def test_split_gate_checks_nonempty_closures_and_effective_temperature(self):
        result = self.evaluate(self.report)
        failed = [key for key, value in result['checks'].items() if not value]
        self.assertEqual(failed, [])

    def test_effective_harmonic_tamper_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['effective_temperature']['harmonic_mean'] += .01
        result = self.evaluate(report)
        self.assertFalse(
            result['checks']['effective_harmonic_mean_is_three_times_temperature']
        )

    def test_empty_stratification_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['stratified_diagnostics']['class'] = []
        result = self.evaluate(report)
        self.assertFalse(result['checks']['class_strata_21_rows'])


    def test_exact_boundary_tamper_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['boundary_counts']['u_eq_0p8'] += 1
        result = self.evaluate(report)
        self.assertFalse(result['checks']['u_eq_0p8_matches_exact_u_cache'])
        self.assertFalse(
            result['checks']['u_eq_0p8_count_matches_frozen_o11_boundary']
        )

    def test_frozen_o11_wrong_bin_tamper_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['risk_quantile_bins'][8]['teacher_wrong_count'] += 1
        report['risk_quantile_bins'][9]['teacher_wrong_count'] -= 1
        report['target_by_risk_decile'] = copy.deepcopy(
            report['risk_quantile_bins']
        )
        result = self.evaluate(report)
        self.assertFalse(
            result['checks']['risk_bin_wrong_counts_match_frozen_o11']
        )

    def test_actual_loss_student_probe_tamper_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['target_numeric_maxima']['student_softmax_max_abs_error'] = 1e-4
        result = self.evaluate(report)
        self.assertFalse(
            result['checks']['actual_loss_student_probe_within_tau']
        )

    def test_effective_temperature_threshold_tamper_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['effective_temperature']['greater_than_1p5_count'] -= 1
        result = self.evaluate(report)
        self.assertFalse(
            result['checks'][
                'effective_temperature_greater_than_1p5_count_matches_cache'
            ]
        )

class SmokePathTest(unittest.TestCase):
    def test_max_samples_alias_and_smoke_default_are_noncanonical(self):
        with mock.patch(
            'sys.argv',
            ['diagnose', '--stage', 'solve', '--split', 'train',
             '--max-samples', '8'],
        ):

            args = DIAG.parse_args()
        output, cache, formal = DIAG.output_paths(args)
        self.assertEqual(args.max_images, 8)
        self.assertFalse(formal)
        self.assertEqual(output.parent, Path('/tmp'))
        self.assertEqual(cache.parent, Path('/tmp'))

    def test_partial_smoke_rejects_canonical_output(self):
        args = SimpleNamespace(
            max_images=8, stage='solve', split='train',
            output=str(DIAG.DEFAULT_DIR / 'o12_budget_parameters.json'),
            risk_cache=None,
        )
        with self.assertRaisesRegex(ValueError, 'partial smoke'):
            DIAG.output_paths(args)


if __name__ == '__main__':
    unittest.main()
