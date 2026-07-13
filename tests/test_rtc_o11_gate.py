import argparse
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'scripts' / 'diagnostics' / 'check_rtc_o11_gate.py'
SPEC = importlib.util.spec_from_file_location('check_rtc_o11_gate', MODULE_PATH)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_report(split, cdf_path, cdf_sha):
    split_contract = GATE.EXPECTED_SPLIT_CONTRACT[split]
    checker_sha = sha256(MODULE_PATH)
    source_sha = {
        'build_rtc_cdf': '1' * 64,
        'diagnose_rtc_routing': '2' * 64,
        'rtc_temperature': '3' * 64,
        'check_rtc_o11_gate': checker_sha,
    }
    rtc_config = {
        'assess_temperature': 1.0,
        'route_quantile': 0.8,
        'route_width': 0.05,
        'reliable_temperature': 0.5,
        'neutral_temperature': 1.0,
        'unreliable_temperature': 2.0,
        'alpha_reliable': 1.0,
        'alpha_unreliable': 1.0,
        'enable_reliable': True,
        'enable_unreliable': True,
        'bisection_iterations': 16,
        'kd_temperature_power': 0.0,
        'coefficient_a': 0.0,
        'reliability_mode': 'confidence',
    }
    definition = {
        'reliability_definition_id': 'neg_log_top1_confidence_v1',
        'reliability_formula': (
            'r=-log(clamp(max(softmax(z/T_assess)),eps,1-eps))'
        ),
        'active_terms': ['confidence'],
        'coefficient_a_active': False,
        'reliability_epsilon': 1e-8,
    }
    cdf_metadata = {
        'full_dataset_scan': True,
        'processed_images': 10582,
        'dataset_size': 10582,
        'reliability_mode': 'confidence',
        'coefficient_a': 0.0,
        'teacher_sha256': GATE.EXPECTED_TEACHER_SHA256,
        'source_sha256': source_sha,
        **definition,
    }
    wrong_counts = [0, 0, 1, 1, 2, 3, 5, 8, 30, 50]
    wrong_rates = [count / 100 for count in wrong_counts]
    risk_bins = [
        {
            'bin': index,
            'population': 'all native valid pixels (micro)',
            'count': 100,
            'teacher_wrong_count': wrong_count,
            'teacher_wrong_rate_native_proxy': wrong_rate,
        }
        for index, (wrong_count, wrong_rate) in enumerate(
            zip(wrong_counts, wrong_rates)
        )
    ]
    return {
        'schema_version': 1,
        'phase': 'O1.1',
        'primary_reliability_mode': 'confidence',
        'split': split,
        'processed_images': split_contract['dataset_size'],
        'dataset_size': split_contract['dataset_size'],
        'formal_full_run': True,
        # Deliberately false: the independent O1.1 gate must not trust/reuse it.
        'all_checks_pass': False,
        'native_correctness_semantics': 'native-grid teacher proxy',
        'cdf_path': str(cdf_path),
        'cdf_sha256': cdf_sha,
        'cdf_metadata': cdf_metadata,
        'cdf_metadata_checks': {
            'full_scan': True,
            'definition_matches': True,
            'source_matches': True,
        },
        'critical_config': {
            'rtc_config': rtc_config,
            'reliability_definition': definition,
            'teacher_output_temp': 3.0,
            'teacher_model': 'deeplabv3',
            'teacher_backbone': 'resnet101',
            'teacher_sha256': GATE.EXPECTED_TEACHER_SHA256,
            'num_classes': 21,
            'teacher_output_grid': 'native',
            'valid_mask_resize': 'nearest',
            'native_correctness_semantics': 'native-grid teacher proxy',
            'route_invariance_temperatures': [1.0, 3.0],
        },
        'rtc_config': rtc_config,
        'input_provenance': {
            'list_sha256': split_contract['list_sha256'],
            'teacher_sha256': GATE.EXPECTED_TEACHER_SHA256,
            'augmentation_seed': 2025,
            'ranking_seed': 3407,
            'crop_size': [512, 512],
            'scale': split_contract['scale'],
            'mirror': split_contract['mirror'],
            'command_args': {
                'phase': 'O1.1',
                'batch_size': split_contract['batch_size'],
                'workers': 0,
                'max_images': 0,
                'ranking_max_pixels_per_image': 1024,
                'seed': 2025,
                'ranking_seed': 3407,
                'crop_size': [512, 512],
            },
        },
        'code_provenance': {
            'git_commit': '5' * 40,
            'source_sha256': source_sha,
        },
        'population': {
            'valid_native_pixels': 1000,
            'finite_valid_pixels': 1000,
            'nonfinite_valid_pixels': 0,
        },
        'global_wrong_rate': 0.10,
        'ranking_sample': {
            'max_pixels_per_image': 1024,
            'seed': 3407,
        },
        'high_risk_coverage': 0.18,
        'low_risk_coverage': 0.80,
        'neutral_quantile_coverage': 0.02,
        'high_risk_wrong_precision': 75 / 180,
        'high_risk_wrong_recall': 0.75,
        'high_risk_enrichment': (75 / 180) / 0.10,
        'low_risk_wrong_rate': 0.025,
        'risk_routing_counts': {
            'semantics': (
                'high: u>0.8; low: u<0.8; boundary: u==0.8'
            ),
            'valid_native_pixels': 1000,
            'teacher_wrong_pixels': 100,
            'high_risk_pixels': 180,
            'high_risk_wrong_pixels': 75,
            'low_risk_pixels': 800,
            'low_risk_wrong_pixels': 20,
            'boundary_pixels': 20,
            'boundary_wrong_pixels': 5,
        },
        'risk_quantile_bins': risk_bins,
        'risk_quantile_spearman': GATE.spearman_rank_correlation(wrong_rates),
        'risk_quantile_pairwise_monotonic_agreement': 1.0,
        'target_residual': {'count': 1000, 'p95': 1e-4},
        'fallback_count': 0,
        'fallback_rate': 0.0,
        'tie_count': 0,
        'tie_rate': 0.0,
        'direction_violations_reliable': 0,
        'direction_violations_unreliable': 0,
        'argmax_disagreements_after_teacher_output_temp': 0,
        'route_invariance_tout1_vs_tout3': {
            'all_route_fields_identical': True,
            'mismatch_counts': {
                'reliability': 0,
                'reliability_quantile': 0,
                'gate_reliable': 0,
                'gate_unreliable': 0,
                'valid_mask': 0,
                'finite_mask': 0,
            },
        },
    }


class RankingDiagnosticsTest(unittest.TestCase):
    def test_tie_aware_spearman(self):
        self.assertAlmostEqual(
            GATE.spearman_rank_correlation([0, 0, 1, 1, 2]),
            0.9486832980505138,
        )
        self.assertEqual(
            GATE.spearman_rank_correlation([5, 4, 3, 2, 1]),
            -1.0,
        )
        self.assertIsNone(GATE.spearman_rank_correlation([1, 1, 1]))


    def test_pairwise_monotonic_agreement_treats_ties_as_concordant(self):
        self.assertEqual(
            GATE.pairwise_monotonic_agreement([0, 0, 1]),
            {'pair_count': 3, 'concordant_count': 3, 'agreement': 1.0},
        )
        self.assertEqual(
            GATE.pairwise_monotonic_agreement([3, 2, 1]),
            {'pair_count': 3, 'concordant_count': 0, 'agreement': 0.0},
        )


class SplitGateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cdf_path = Path(self.directory.name) / 'confidence_cdf.pt'
        self.cdf_path.write_bytes(b'frozen confidence cdf')
        self.report = make_report(
            'train', self.cdf_path, sha256(self.cdf_path)
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_pass_is_independent_of_legacy_all_checks_flag(self):
        evaluation = GATE.evaluate_split(self.report, 'train')
        self.assertTrue(evaluation['pass'])
        self.assertAlmostEqual(
            evaluation['metrics']['high_risk_enrichment_recomputed'],
            (75 / 180) / 0.10,
        )
        self.assertAlmostEqual(
            evaluation['metrics']['risk_quantile_spearman_recomputed'],
            self.report['risk_quantile_spearman'],
        )

    def test_u_equal_0p8_boundary_is_not_counted_as_high_risk(self):
        evaluation = GATE.evaluate_split(self.report, 'train')
        top_two_bin_pixels = sum(
            row['count'] for row in self.report['risk_quantile_bins'][8:]
        )
        self.assertEqual(top_two_bin_pixels, 200)
        self.assertEqual(
            evaluation['metrics']['risk_routing_counts']['high_risk_pixels'],
            180,
        )
        self.assertEqual(
            evaluation['metrics']['risk_metrics_recomputed_from_routing_counts'][
                'high_risk_coverage'
            ],
            0.18,
        )

    def test_rejects_weak_enrichment_and_nonmonotonic_deciles(self):
        report = copy.deepcopy(self.report)
        report['all_checks_pass'] = True
        wrong_counts = [
            row['teacher_wrong_count'] for row in report['risk_quantile_bins']
        ]
        for row, wrong_count in zip(
            report['risk_quantile_bins'], reversed(wrong_counts)
        ):
            row['teacher_wrong_count'] = wrong_count
            row['teacher_wrong_rate_native_proxy'] = wrong_count / row['count']
        report['risk_routing_counts'].update({
            'high_risk_wrong_pixels': 20,
            'low_risk_wrong_pixels': 75,
            'boundary_wrong_pixels': 5,
        })
        report['global_wrong_rate'] = 0.10
        report['high_risk_wrong_precision'] = 20 / 180
        report['high_risk_wrong_recall'] = 0.20
        report['high_risk_enrichment'] = (20 / 180) / 0.10
        report['low_risk_wrong_rate'] = 75 / 800
        rates = [
            row['teacher_wrong_rate_native_proxy']
            for row in report['risk_quantile_bins']
        ]
        report['risk_quantile_spearman'] = GATE.spearman_rank_correlation(rates)
        report['risk_quantile_pairwise_monotonic_agreement'] = (
            GATE.pairwise_monotonic_agreement(rates)['agreement']
        )
        evaluation = GATE.evaluate_split(report, 'train')
        self.assertFalse(evaluation['pass'])
        self.assertFalse(
            evaluation['checks']['high_risk_enrichment_recomputed_ge_2']
        )
        self.assertFalse(
            evaluation['checks'][
                'risk_quantile_pairwise_monotonic_agreement_ge_0p90'
            ]
        )

    def test_spearman_mismatch_is_reported_but_not_gated(self):
        report = copy.deepcopy(self.report)
        report['risk_quantile_spearman'] = 0.95
        evaluation = GATE.evaluate_split(report, 'train')
        self.assertTrue(evaluation['pass'])
        self.assertFalse(
            evaluation['metrics'][
                'risk_quantile_spearman_reported_matches_recomputation'
            ]
        )

    def test_rejects_reported_pairwise_agreement_mismatch(self):
        report = copy.deepcopy(self.report)
        report['risk_quantile_pairwise_monotonic_agreement'] = 0.95
        evaluation = GATE.evaluate_split(report, 'train')
        self.assertFalse(evaluation['pass'])
        self.assertFalse(
            evaluation['checks'][
                'reported_risk_quantile_pairwise_monotonic_agreement_matches_recomputation'
            ]
        )

    def test_rejects_mechanism_or_definition_drift(self):
        report = copy.deepcopy(self.report)
        report['fallback_count'] = 1
        report['cdf_metadata']['reliability_definition_id'] = (
            'different_definition'
        )
        report['critical_config']['reliability_definition'][
            'active_terms'
        ] = ['variance']
        evaluation = GATE.evaluate_split(report, 'train')
        self.assertFalse(evaluation['pass'])
        self.assertFalse(
            evaluation['checks']['fallback_count_and_rate_zero']
        )
        self.assertFalse(
            evaluation['checks']['cdf_reliability_definition_id_exact']
        )
        self.assertFalse(
            evaluation['checks']['critical_reliability_definition_exact']
        )

    def test_rejects_routing_partition_and_bin_wrong_count_mismatch(self):
        report = copy.deepcopy(self.report)
        report['risk_routing_counts']['boundary_pixels'] += 1
        report['risk_quantile_bins'][0]['teacher_wrong_count'] += 1
        evaluation = GATE.evaluate_split(report, 'train')
        self.assertFalse(evaluation['pass'])
        self.assertFalse(
            evaluation['checks']['risk_routing_count_partition_exact']
        )
        self.assertFalse(
            evaluation['checks']['risk_bin_wrong_rates_match_exact_counts']
        )
        self.assertFalse(
            evaluation['checks'][
                'risk_bin_count_and_wrong_count_totals_match_routing'
            ]
        )
    def test_rejects_frozen_split_provenance_drift(self):
        report = copy.deepcopy(self.report)
        report['input_provenance']['list_sha256'] = 'f' * 64
        report['input_provenance']['command_args']['batch_size'] = 3
        evaluation = GATE.evaluate_split(report, 'train')
        self.assertFalse(evaluation['pass'])
        self.assertFalse(evaluation['checks']['split_list_sha256_exact'])
        self.assertFalse(evaluation['checks']['split_batch_size_exact'])

    def test_rejects_checker_source_sha_not_matching_current_file(self):
        report = copy.deepcopy(self.report)
        report['code_provenance']['source_sha256'][
            'check_rtc_o11_gate'
        ] = 'f' * 64
        report['cdf_metadata']['source_sha256'][
            'check_rtc_o11_gate'
        ] = 'f' * 64
        evaluation = GATE.evaluate_split(report, 'train')
        self.assertFalse(evaluation['pass'])
        self.assertFalse(
            evaluation['checks'][
                'checker_sha256_matches_cdf_and_diagnostic_sources'
            ]
        )


class JointGateTest(unittest.TestCase):
    def run_main(self, train, val, cdf_path, strict=False):
        directory = cdf_path.parent
        train_path = directory / 'train.json'
        val_path = directory / 'val.json'
        output_path = directory / 'joint.json'
        train_path.write_text(json.dumps(train), encoding='utf-8')
        val_path.write_text(json.dumps(val), encoding='utf-8')
        args = argparse.Namespace(
            train_json=str(train_path),
            val_json=str(val_path),
            output=str(output_path),
            strict=strict,
        )
        with mock.patch.object(GATE, 'parse_args', return_value=args):
            with redirect_stdout(io.StringIO()):
                GATE.main()
        return json.loads(output_path.read_text(encoding='utf-8'))

    def test_passing_joint_gate_writes_independent_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            cdf_path = Path(directory) / 'confidence_cdf.pt'
            cdf_path.write_bytes(b'frozen confidence cdf')
            digest = sha256(cdf_path)
            train = make_report('train', cdf_path, digest)
            val = make_report('val', cdf_path, digest)
            result = self.run_main(train, val, cdf_path)
        self.assertTrue(result['joint_gate_pass'])
        self.assertEqual(result['phase'], 'O1.1')
        self.assertEqual(
            result['gate_profile'], 'confidence_only_non_synonymous_v1'
        )
        self.assertTrue(result['train_evaluation']['pass'])
        self.assertTrue(result['val_evaluation']['pass'])
        self.assertEqual(result['checker_sha256'], sha256(MODULE_PATH))

    def test_strict_failure_exits_two_and_still_writes_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            cdf_path = Path(directory) / 'confidence_cdf.pt'
            cdf_path.write_bytes(b'frozen confidence cdf')
            digest = sha256(cdf_path)
            train = make_report('train', cdf_path, digest)
            val = make_report('val', cdf_path, digest)
            val['critical_config']['rtc_config']['coefficient_a'] = 1.0
            output_path = Path(directory) / 'joint.json'
            with self.assertRaises(SystemExit) as raised:
                self.run_main(train, val, cdf_path, strict=True)
            self.assertEqual(raised.exception.code, 2)
            result = json.loads(output_path.read_text(encoding='utf-8'))
        self.assertFalse(result['joint_gate_pass'])
        self.assertFalse(result['joint_checks']['critical_config_exact_match'])


if __name__ == '__main__':
    unittest.main()
