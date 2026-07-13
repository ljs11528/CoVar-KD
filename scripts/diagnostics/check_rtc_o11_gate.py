#!/usr/bin/env python3
"""Independent joint gate for the Phase O1.1 confidence-only diagnosis.

This checker deliberately does not reuse the O1 AP comparison or trust the
per-split ``all_checks_pass`` flag.  It recomputes the pre-registered routing
quality and risk-decile monotonicity criteria from the diagnostic summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'runs' / 'diagnostics' / 'phaseO_o11'

GATE_PROFILE = 'confidence_only_non_synonymous_v1'
EXPECTED_PHASE = 'O1.1'
EXPECTED_RELIABILITY_MODE = 'confidence'
EXPECTED_DEFINITION_ID = 'neg_log_top1_confidence_v1'
EXPECTED_ACTIVE_TERMS = ['confidence']
EXPECTED_COEFFICIENT_A = 0.0
EXPECTED_EPSILON = 1e-8
EXPECTED_RELIABILITY_FORMULA = (
    'r=-log(clamp(max(softmax(z/T_assess)),eps,1-eps))'
)
EXPECTED_RELIABILITY_DEFINITION = {
    'reliability_definition_id': EXPECTED_DEFINITION_ID,
    'reliability_formula': EXPECTED_RELIABILITY_FORMULA,
    'active_terms': EXPECTED_ACTIVE_TERMS,
    'coefficient_a_active': False,
    'reliability_epsilon': EXPECTED_EPSILON,
}

EXPECTED_TEACHER_SHA256 = (
    'ac49b2c7720b21d565072e974e4404fcb009ba106288d95d1a4bb25f09c3fe58'
)
EXPECTED_RTC_CONFIG = {
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
EXPECTED_SPLIT_CONTRACT = {
    'train': {
        'list_sha256': (
            'd1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24'
        ),
        'dataset_size': 10582,
        'batch_size': 4,
        'scale': True,
        'mirror': True,
    },
    'val': {
        'list_sha256': (
            'cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d'
        ),
        'dataset_size': 1449,
        'batch_size': 1,
        'scale': False,
        'mirror': False,
    },
}
EXPECTED_CHECKER_SOURCE_KEY = 'check_rtc_o11_gate'

MIN_HIGH_RISK_COVERAGE = 0.15
MAX_HIGH_RISK_COVERAGE = 0.25
MIN_HIGH_RISK_ENRICHMENT = 2.0
MIN_HIGH_RISK_WRONG_RECALL = 0.70
MIN_RISK_QUANTILE_PAIRWISE_MONOTONIC_AGREEMENT = 0.90
MAX_RESIDUAL_P95 = 1e-3
EXPECTED_RISK_BIN_COUNT = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Validate matching full train/val confidence-only RTC O1.1 '
            'diagnostic artifacts with independent non-AP gates.'
        )
    )
    parser.add_argument(
        '--train-json',
        default=str(DEFAULT_DIR / 'rtc_confidence_routing_train.json'),
    )
    parser.add_argument(
        '--val-json',
        default=str(DEFAULT_DIR / 'rtc_confidence_routing_val.json'),
    )
    parser.add_argument(
        '--output',
        default=str(DEFAULT_DIR / 'o11_confidence_gate.json'),
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        default=False,
        help='Exit with status 2 when the joint gate does not pass.',
    )
    return parser.parse_args()


def load_json(path):
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'JSON root must be an object: {path}')
    return payload


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_nonnegative_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def numeric_equal(left, right, *, atol=1e-12):
    return (
        is_finite_number(left)
        and is_finite_number(right)
        and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    )


def all_dict_values_true(payload):
    return (
        isinstance(payload, dict)
        and bool(payload)
        and all(value is True for value in payload.values())
    )


def valid_sha256_mapping(payload):
    return (
        isinstance(payload, dict)
        and bool(payload)
        and all(
            isinstance(key, str)
            and bool(key)
            and isinstance(value, str)
            and len(value) == 64
            and all(character in '0123456789abcdef' for character in value)
            for key, value in payload.items()
        )
    )


def resolved_reported_path(payload, key):
    value = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value:
        return None
    return str(Path(value).resolve())


def frozen_contract_checks(payload, expected_split, checker_sha256):
    expected = EXPECTED_SPLIT_CONTRACT.get(expected_split, {})
    critical = payload.get('critical_config')
    critical = critical if isinstance(critical, dict) else {}
    critical_rtc = critical.get('rtc_config')
    provenance = payload.get('input_provenance')
    provenance = provenance if isinstance(provenance, dict) else {}
    command_args = provenance.get('command_args')
    command_args = command_args if isinstance(command_args, dict) else {}
    ranking = payload.get('ranking_sample')
    ranking = ranking if isinstance(ranking, dict) else {}
    code_provenance = payload.get('code_provenance')
    code_provenance = (
        code_provenance if isinstance(code_provenance, dict) else {}
    )
    code_sources = code_provenance.get('source_sha256')
    cdf_metadata = payload.get('cdf_metadata')
    cdf_metadata = cdf_metadata if isinstance(cdf_metadata, dict) else {}
    cdf_sources = cdf_metadata.get('source_sha256')
    return {
        'frozen_rtc_config_exact': critical_rtc == EXPECTED_RTC_CONFIG,
        'root_primary_reliability_mode_confidence': (
            payload.get('primary_reliability_mode')
            == EXPECTED_RELIABILITY_MODE
        ),
        'frozen_teacher_output_temp_3': numeric_equal(
            critical.get('teacher_output_temp'), 3.0
        ),
        'frozen_teacher_model_deeplabv3': (
            critical.get('teacher_model') == 'deeplabv3'
        ),
        'frozen_teacher_backbone_resnet101': (
            critical.get('teacher_backbone') == 'resnet101'
        ),
        'frozen_num_classes_21': critical.get('num_classes') == 21,
        'frozen_teacher_output_grid_native': (
            critical.get('teacher_output_grid') == 'native'
        ),
        'frozen_valid_mask_resize_nearest': (
            critical.get('valid_mask_resize') == 'nearest'
        ),
        'frozen_route_invariance_temperatures_1_3': (
            critical.get('route_invariance_temperatures') == [1.0, 3.0]
        ),
        'split_dataset_size_exact': (
            payload.get('dataset_size') == expected.get('dataset_size')
        ),
        'split_list_sha256_exact': (
            provenance.get('list_sha256') == expected.get('list_sha256')
        ),
        'frozen_teacher_sha256_exact': (
            provenance.get('teacher_sha256') == EXPECTED_TEACHER_SHA256
            and critical.get('teacher_sha256') == EXPECTED_TEACHER_SHA256
            and cdf_metadata.get('teacher_sha256') == EXPECTED_TEACHER_SHA256
        ),
        'frozen_augmentation_seed_2025': (
            provenance.get('augmentation_seed') == 2025
            and command_args.get('seed') == 2025
        ),
        'frozen_ranking_seed_3407': (
            provenance.get('ranking_seed') == 3407
            and command_args.get('ranking_seed') == 3407
            and ranking.get('seed') == 3407
        ),
        'frozen_ranking_cap_1024': (
            command_args.get('ranking_max_pixels_per_image') == 1024
            and ranking.get('max_pixels_per_image') == 1024
        ),
        'frozen_workers_zero': command_args.get('workers') == 0,
        'formal_max_images_zero': command_args.get('max_images') == 0,
        'split_batch_size_exact': (
            command_args.get('batch_size') == expected.get('batch_size')
        ),
        'frozen_crop_size_512': (
            provenance.get('crop_size') == [512, 512]
            and command_args.get('crop_size') == [512, 512]
        ),
        'split_effective_scale_exact': (
            provenance.get('scale') is expected.get('scale')
        ),
        'split_effective_mirror_exact': (
            provenance.get('mirror') is expected.get('mirror')
        ),
        'checker_sha256_well_formed': (
            isinstance(checker_sha256, str) and len(checker_sha256) == 64
        ),
        'checker_sha256_matches_cdf_and_diagnostic_sources': (
            isinstance(code_sources, dict)
            and isinstance(cdf_sources, dict)
            and code_sources.get(EXPECTED_CHECKER_SOURCE_KEY)
            == checker_sha256
            == cdf_sources.get(EXPECTED_CHECKER_SOURCE_KEY)
        ),
    }


def average_ranks(values):
    """Return one-based average ranks, using exact equality for ties."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def spearman_rank_correlation(values):
    """Spearman rho between increasing risk-bin index and ``values``."""
    if len(values) < 2 or not all(is_finite_number(value) for value in values):
        return None
    x_ranks = average_ranks(list(range(len(values))))
    y_ranks = average_ranks([float(value) for value in values])
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    covariance = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_ranks, y_ranks)
    )
    x_square_sum = sum((value - x_mean) ** 2 for value in x_ranks)
    y_square_sum = sum((value - y_mean) ** 2 for value in y_ranks)
    denominator = math.sqrt(x_square_sum * y_square_sum)
    if denominator == 0.0:
        return None
    return covariance / denominator


def reported_spearman_value(payload):
    value = payload.get('risk_quantile_spearman')
    if is_finite_number(value):
        return float(value)
    if isinstance(value, dict):
        for key in ('spearman_rho', 'rho'):
            if is_finite_number(value.get(key)):
                return float(value[key])
    return None


def pairwise_monotonic_agreement(values):
    """Fraction of ordered bin pairs whose error rates are nondecreasing."""
    if len(values) < 2 or not all(is_finite_number(value) for value in values):
        return None
    pair_count = len(values) * (len(values) - 1) // 2
    concordant_count = sum(
        float(values[left]) <= float(values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return {
        'pair_count': pair_count,
        'concordant_count': concordant_count,
        'agreement': concordant_count / pair_count,
    }


def reported_pairwise_agreement_value(payload):
    value = payload.get('risk_quantile_pairwise_monotonic_agreement')
    return float(value) if is_finite_number(value) else None


def evaluate_routing_counts(payload):
    counts = payload.get('risk_routing_counts')
    required_fields = (
        'valid_native_pixels',
        'teacher_wrong_pixels',
        'high_risk_pixels',
        'high_risk_wrong_pixels',
        'low_risk_pixels',
        'low_risk_wrong_pixels',
        'boundary_pixels',
        'boundary_wrong_pixels',
    )
    structure_valid = (
        isinstance(counts, dict)
        and all(field in counts for field in required_fields)
    )
    values_valid = structure_valid and all(
        is_nonnegative_int(counts.get(field)) for field in required_fields
    )
    if not values_valid:
        return {
            'structure_valid': bool(structure_valid),
            'all_counts_nonnegative_integers': False,
            'subgroup_count_partition_exact': False,
            'subgroup_wrong_partition_exact': False,
            'wrong_counts_within_corresponding_counts': False,
            'matches_population_valid_native_pixels': False,
            'counts': counts if isinstance(counts, dict) else None,
            'metrics': {},
        }

    valid = counts['valid_native_pixels']
    wrong = counts['teacher_wrong_pixels']
    high = counts['high_risk_pixels']
    high_wrong = counts['high_risk_wrong_pixels']
    low = counts['low_risk_pixels']
    low_wrong = counts['low_risk_wrong_pixels']
    boundary = counts['boundary_pixels']
    boundary_wrong = counts['boundary_wrong_pixels']
    population = payload.get('population')
    population_valid = (
        population.get('valid_native_pixels')
        if isinstance(population, dict) else None
    )
    subgroup_count_partition = high + low + boundary == valid
    subgroup_wrong_partition = high_wrong + low_wrong + boundary_wrong == wrong
    wrong_counts_in_range = (
        wrong <= valid
        and high_wrong <= high
        and low_wrong <= low
        and boundary_wrong <= boundary
    )
    metrics = {}
    if valid > 0 and wrong > 0 and high > 0 and low > 0:
        global_wrong_rate = wrong / valid
        high_wrong_precision = high_wrong / high
        metrics = {
            'global_wrong_rate': global_wrong_rate,
            'high_risk_coverage': high / valid,
            'low_risk_coverage': low / valid,
            'neutral_quantile_coverage': boundary / valid,
            'high_risk_wrong_precision': high_wrong_precision,
            'high_risk_wrong_recall': high_wrong / wrong,
            'high_risk_enrichment': high_wrong_precision / global_wrong_rate,
            'low_risk_wrong_rate': low_wrong / low,
        }
    return {
        'structure_valid': bool(structure_valid),
        'all_counts_nonnegative_integers': bool(values_valid),
        'subgroup_count_partition_exact': bool(subgroup_count_partition),
        'subgroup_wrong_partition_exact': bool(subgroup_wrong_partition),
        'wrong_counts_within_corresponding_counts': bool(wrong_counts_in_range),
        'matches_population_valid_native_pixels': (
            is_nonnegative_int(population_valid) and valid == population_valid
        ),
        'counts': {field: counts[field] for field in required_fields},
        'metrics': metrics,
    }


def evaluate_risk_bins(payload):
    rows = payload.get('risk_quantile_bins')
    population = payload.get('population')
    expected_population = (
        population.get('valid_native_pixels')
        if isinstance(population, dict) else None
    )
    structure_valid = isinstance(rows, list) and len(rows) == EXPECTED_RISK_BIN_COUNT
    ordered_rows = []
    if structure_valid:
        try:
            ordered_rows = sorted(rows, key=lambda row: row.get('bin', -1))
        except (AttributeError, TypeError):
            structure_valid = False

    bin_indices_valid = structure_valid and all(
        isinstance(row, dict)
        and isinstance(row.get('bin'), int)
        and not isinstance(row.get('bin'), bool)
        and row.get('bin') == index
        for index, row in enumerate(ordered_rows)
    )
    counts_valid = bin_indices_valid and all(
        isinstance(row.get('count'), int)
        and not isinstance(row.get('count'), bool)
        and row.get('count') > 0
        for row in ordered_rows
    )
    wrong_counts_valid = bin_indices_valid and all(
        is_nonnegative_int(row.get('teacher_wrong_count'))
        and row['teacher_wrong_count'] <= row.get('count', -1)
        for row in ordered_rows
    )
    rates_valid = wrong_counts_valid and all(
        is_finite_number(row.get('teacher_wrong_rate_native_proxy'))
        and 0.0 <= float(row['teacher_wrong_rate_native_proxy']) <= 1.0
        for row in ordered_rows
    )
    rates_match_counts = rates_valid and all(
        numeric_equal(
            row['teacher_wrong_rate_native_proxy'],
            row['teacher_wrong_count'] / row['count'],
        )
        for row in ordered_rows
    )
    count_sum = (
        sum(row['count'] for row in ordered_rows)
        if counts_valid else None
    )
    counts_match_population = (
        counts_valid
        and isinstance(expected_population, int)
        and not isinstance(expected_population, bool)
        and count_sum == expected_population
    )
    wrong_count_sum = (
        sum(row['teacher_wrong_count'] for row in ordered_rows)
        if wrong_counts_valid else None
    )
    routing_counts = payload.get('risk_routing_counts')
    routing_valid = (
        routing_counts.get('valid_native_pixels')
        if isinstance(routing_counts, dict) else None
    )
    routing_wrong = (
        routing_counts.get('teacher_wrong_pixels')
        if isinstance(routing_counts, dict) else None
    )
    totals_match_routing = (
        count_sum is not None
        and wrong_count_sum is not None
        and count_sum == routing_valid
        and wrong_count_sum == routing_wrong
    )
    wrong_rates = (
        [float(row['teacher_wrong_rate_native_proxy']) for row in ordered_rows]
        if rates_valid else []
    )
    recomputed_rho = spearman_rank_correlation(wrong_rates)
    reported_rho = reported_spearman_value(payload)
    reported_matches = (
        recomputed_rho is not None
        and reported_rho is not None
        and numeric_equal(recomputed_rho, reported_rho, atol=1e-9)
    )
    pairwise = pairwise_monotonic_agreement(wrong_rates)
    recomputed_pairwise = (
        pairwise['agreement'] if pairwise is not None else None
    )
    reported_pairwise = reported_pairwise_agreement_value(payload)
    reported_pairwise_matches = (
        recomputed_pairwise is not None
        and reported_pairwise is not None
        and numeric_equal(recomputed_pairwise, reported_pairwise, atol=1e-9)
    )
    return {
        'structure_valid': bool(structure_valid),
        'bin_indices_exact_0_to_9': bool(bin_indices_valid),
        'all_ten_bins_nonempty': bool(counts_valid),
        'wrong_counts_valid': bool(wrong_counts_valid),
        'wrong_rates_finite_probabilities': bool(rates_valid),
        'wrong_rates_match_exact_counts': bool(rates_match_counts),
        'counts_match_native_valid_population': bool(counts_match_population),
        'counts_and_wrong_counts_match_routing_totals': bool(
            totals_match_routing
        ),
        'count_sum': count_sum,
        'wrong_count_sum': wrong_count_sum,
        'expected_population': expected_population,
        'wrong_rates': wrong_rates,
        'recomputed_spearman_rho': recomputed_rho,
        'reported_spearman_rho': reported_rho,
        'reported_spearman_matches_recomputation': bool(reported_matches),
        'pairwise_monotonic_pair_count': (
            pairwise['pair_count'] if pairwise is not None else None
        ),
        'pairwise_monotonic_concordant_count': (
            pairwise['concordant_count'] if pairwise is not None else None
        ),
        'recomputed_pairwise_monotonic_agreement': recomputed_pairwise,
        'reported_pairwise_monotonic_agreement': reported_pairwise,
        'reported_pairwise_matches_recomputation': bool(
            reported_pairwise_matches
        ),
        'pairwise_monotonic_agreement_ge_0p90': (
            recomputed_pairwise is not None
            and recomputed_pairwise
            >= MIN_RISK_QUANTILE_PAIRWISE_MONOTONIC_AGREEMENT
        ),
    }


def evaluate_split(payload, expected_split):
    population = payload.get('population')
    population = population if isinstance(population, dict) else {}
    residual = payload.get('target_residual')
    residual = residual if isinstance(residual, dict) else {}
    route = payload.get('route_invariance_tout1_vs_tout3')
    route = route if isinstance(route, dict) else {}
    route_mismatches = route.get('mismatch_counts')
    rtc_config = payload.get('rtc_config')
    rtc_config = rtc_config if isinstance(rtc_config, dict) else {}
    critical_config = payload.get('critical_config')
    critical_config = critical_config if isinstance(critical_config, dict) else {}
    critical_rtc = critical_config.get('rtc_config')
    critical_rtc = critical_rtc if isinstance(critical_rtc, dict) else {}
    critical_definition = critical_config.get('reliability_definition')
    critical_definition = (
        critical_definition if isinstance(critical_definition, dict) else {}
    )
    cdf_metadata = payload.get('cdf_metadata')
    cdf_metadata = cdf_metadata if isinstance(cdf_metadata, dict) else {}
    code_provenance = payload.get('code_provenance')
    code_provenance = code_provenance if isinstance(code_provenance, dict) else {}
    code_sources = code_provenance.get('source_sha256')
    cdf_sources = cdf_metadata.get('source_sha256')

    valid_pixels = population.get('valid_native_pixels')
    finite_pixels = population.get('finite_valid_pixels')
    nonfinite_pixels = population.get('nonfinite_valid_pixels')
    processed_images = payload.get('processed_images')
    dataset_size = payload.get('dataset_size')

    coverage = payload.get('high_risk_coverage')
    global_wrong = payload.get('global_wrong_rate')
    high_wrong = payload.get('high_risk_wrong_precision')
    recall = payload.get('high_risk_wrong_recall')
    low_wrong = payload.get('low_risk_wrong_rate')
    reported_enrichment = payload.get('high_risk_enrichment')
    routing_counts = evaluate_routing_counts(payload)
    risk_bins = evaluate_risk_bins(payload)
    recomputed_metrics = routing_counts['metrics']
    recomputed_enrichment = recomputed_metrics.get('high_risk_enrichment')

    reported_metrics_match_routing_counts = all(
        numeric_equal(payload.get(key), recomputed_metrics.get(key), atol=1e-9)
        for key in (
            'global_wrong_rate',
            'high_risk_coverage',
            'low_risk_coverage',
            'neutral_quantile_coverage',
            'high_risk_wrong_precision',
            'high_risk_wrong_recall',
            'high_risk_enrichment',
            'low_risk_wrong_rate',
        )
    )

    checker_sha256 = file_sha256(Path(__file__).resolve())
    frozen_checks = frozen_contract_checks(
        payload, expected_split, checker_sha256
    )
    checks = {
        **frozen_checks,
        'schema_version_1': payload.get('schema_version') == 1,
        'phase_o1_1': payload.get('phase') == EXPECTED_PHASE,
        'split_exact': payload.get('split') == expected_split,
        'formal_full_run': payload.get('formal_full_run') is True,
        'processed_images_equal_positive_dataset_size': (
            isinstance(processed_images, int)
            and not isinstance(processed_images, bool)
            and isinstance(dataset_size, int)
            and not isinstance(dataset_size, bool)
            and processed_images == dataset_size
            and dataset_size > 0
        ),
        'native_valid_pixels_positive': (
            isinstance(valid_pixels, int)
            and not isinstance(valid_pixels, bool)
            and valid_pixels > 0
        ),
        'native_finite_partition_complete': (
            isinstance(valid_pixels, int)
            and isinstance(finite_pixels, int)
            and isinstance(nonfinite_pixels, int)
            and finite_pixels + nonfinite_pixels == valid_pixels
        ),
        'native_nonfinite_valid_pixels_zero': nonfinite_pixels == 0,
        'high_risk_coverage_in_0p15_0p25': (
            is_finite_number(recomputed_metrics.get('high_risk_coverage'))
            and MIN_HIGH_RISK_COVERAGE
            <= float(recomputed_metrics['high_risk_coverage'])
            <= MAX_HIGH_RISK_COVERAGE
        ),
        'global_wrong_rate_positive_probability': (
            is_finite_number(recomputed_metrics.get('global_wrong_rate'))
            and 0.0 < float(recomputed_metrics['global_wrong_rate']) <= 1.0
        ),
        'high_risk_wrong_precision_probability': (
            is_finite_number(recomputed_metrics.get('high_risk_wrong_precision'))
            and 0.0
            <= float(recomputed_metrics['high_risk_wrong_precision'])
            <= 1.0
        ),
        'high_risk_enrichment_recomputed_ge_2': (
            recomputed_enrichment is not None
            and recomputed_enrichment >= MIN_HIGH_RISK_ENRICHMENT
        ),
        'reported_enrichment_matches_recomputation': (
            recomputed_enrichment is not None
            and numeric_equal(reported_enrichment, recomputed_enrichment)
        ),
        'high_risk_wrong_recall_ge_0p70': (
            is_finite_number(recomputed_metrics.get('high_risk_wrong_recall'))
            and MIN_HIGH_RISK_WRONG_RECALL
            <= float(recomputed_metrics['high_risk_wrong_recall'])
            <= 1.0
        ),
        'low_risk_wrong_rate_below_global': (
            is_finite_number(recomputed_metrics.get('low_risk_wrong_rate'))
            and is_finite_number(recomputed_metrics.get('global_wrong_rate'))
            and 0.0
            <= float(recomputed_metrics['low_risk_wrong_rate'])
            < float(recomputed_metrics['global_wrong_rate'])
        ),
        'risk_routing_counts_structure_valid': (
            routing_counts['structure_valid']
        ),
        'risk_routing_counts_nonnegative_integers': (
            routing_counts['all_counts_nonnegative_integers']
        ),
        'risk_routing_count_partition_exact': (
            routing_counts['subgroup_count_partition_exact']
        ),
        'risk_routing_wrong_partition_exact': (
            routing_counts['subgroup_wrong_partition_exact']
        ),
        'risk_routing_wrong_counts_in_range': (
            routing_counts['wrong_counts_within_corresponding_counts']
        ),
        'risk_routing_valid_count_matches_population': (
            routing_counts['matches_population_valid_native_pixels']
        ),
        'reported_risk_metrics_match_routing_count_recomputation': (
            reported_metrics_match_routing_counts
        ),
        'risk_bins_structure_valid': risk_bins['structure_valid'],
        'risk_bin_wrong_counts_valid': risk_bins['wrong_counts_valid'],
        'risk_bin_wrong_rates_match_exact_counts': (
            risk_bins['wrong_rates_match_exact_counts']
        ),
        'risk_bin_count_and_wrong_count_totals_match_routing': (
            risk_bins['counts_and_wrong_counts_match_routing_totals']
        ),
        'risk_bin_indices_exact_0_to_9': risk_bins['bin_indices_exact_0_to_9'],
        'all_ten_risk_bins_nonempty': risk_bins['all_ten_bins_nonempty'],
        'risk_bin_wrong_rates_valid': risk_bins['wrong_rates_finite_probabilities'],
        'risk_bin_counts_match_native_valid_population': (
            risk_bins['counts_match_native_valid_population']
        ),
        'risk_quantile_pair_count_exactly_45': (
            risk_bins['pairwise_monotonic_pair_count'] == 45
        ),
        'reported_risk_quantile_pairwise_monotonic_agreement_matches_recomputation': (
            risk_bins['reported_pairwise_matches_recomputation']
        ),
        'risk_quantile_pairwise_monotonic_agreement_ge_0p90': (
            risk_bins['pairwise_monotonic_agreement_ge_0p90']
        ),
        'fallback_count_and_rate_zero': (
            payload.get('fallback_count') == 0
            and numeric_equal(payload.get('fallback_rate'), 0.0)
        ),
        'tie_count_and_rate_zero': (
            payload.get('tie_count') == 0
            and numeric_equal(payload.get('tie_rate'), 0.0)
        ),
        'direction_violations_zero': (
            payload.get('direction_violations_reliable') == 0
            and payload.get('direction_violations_unreliable') == 0
        ),
        'teacher_output_temperature_argmax_mismatch_zero': (
            payload.get('argmax_disagreements_after_teacher_output_temp') == 0
        ),
        'tout_route_mismatch_counts_zero': (
            route.get('all_route_fields_identical') is True
            and isinstance(route_mismatches, dict)
            and bool(route_mismatches)
            and all(value == 0 for value in route_mismatches.values())
        ),
        'active_solved_residual_population_positive': (
            isinstance(residual.get('count'), int)
            and not isinstance(residual.get('count'), bool)
            and residual.get('count') > 0
        ),
        'active_solved_residual_p95_lt_1e_3': (
            is_finite_number(residual.get('p95'))
            and 0.0 <= float(residual['p95']) < MAX_RESIDUAL_P95
        ),
        'root_and_critical_rtc_config_exact_match': (
            bool(rtc_config) and rtc_config == critical_rtc
        ),
        'critical_reliability_mode_confidence': (
            critical_rtc.get('reliability_mode') == EXPECTED_RELIABILITY_MODE
        ),
        'critical_route_quantile_0p8': numeric_equal(
            critical_rtc.get('route_quantile'), 0.8
        ),
        'critical_coefficient_a_zero': numeric_equal(
            critical_rtc.get('coefficient_a'), EXPECTED_COEFFICIENT_A
        ),
        'critical_reliability_definition_exact': (
            critical_definition == EXPECTED_RELIABILITY_DEFINITION
        ),
        'cdf_reliability_mode_confidence': (
            cdf_metadata.get('reliability_mode') == EXPECTED_RELIABILITY_MODE
        ),
        'cdf_coefficient_a_zero': numeric_equal(
            cdf_metadata.get('coefficient_a'), EXPECTED_COEFFICIENT_A
        ),
        'cdf_reliability_definition_id_exact': (
            cdf_metadata.get('reliability_definition_id')
            == EXPECTED_DEFINITION_ID
        ),
        'cdf_reliability_formula_exact': (
            cdf_metadata.get('reliability_formula')
            == EXPECTED_RELIABILITY_FORMULA
        ),
        'cdf_active_terms_confidence_only': (
            cdf_metadata.get('active_terms') == EXPECTED_ACTIVE_TERMS
        ),
        'cdf_coefficient_a_inactive': (
            cdf_metadata.get('coefficient_a_active') is False
        ),
        'cdf_reliability_epsilon_1e_8': numeric_equal(
            cdf_metadata.get('reliability_epsilon'), EXPECTED_EPSILON
        ),
        'cdf_was_complete_full_dataset_scan': (
            cdf_metadata.get('full_dataset_scan') is True
            and isinstance(cdf_metadata.get('processed_images'), int)
            and cdf_metadata.get('processed_images')
            == cdf_metadata.get('dataset_size')
            and cdf_metadata.get('dataset_size', 0) > 0
        ),
        'cdf_metadata_checks_all_true': all_dict_values_true(
            payload.get('cdf_metadata_checks')
        ),
        'source_sha256_metadata_well_formed': (
            valid_sha256_mapping(code_sources)
            and valid_sha256_mapping(cdf_sources)
        ),
        'diagnostic_sources_match_frozen_cdf_sources': (
            valid_sha256_mapping(code_sources)
            and code_sources == cdf_sources
        ),
        'teacher_sha_matches_critical_and_frozen_cdf_metadata': (
            isinstance(
                payload.get('input_provenance', {}).get('teacher_sha256'), str
            )
            and payload.get('input_provenance', {}).get('teacher_sha256')
            == critical_config.get('teacher_sha256')
            == cdf_metadata.get('teacher_sha256')
        ),
    }
    return {
        'pass': all(checks.values()),
        'checks': checks,
        'metrics': {
            'high_risk_coverage': coverage,
            'global_wrong_rate': global_wrong,
            'high_risk_wrong_precision': high_wrong,
            'high_risk_enrichment_reported': reported_enrichment,
            'high_risk_enrichment_recomputed': recomputed_enrichment,
            'high_risk_wrong_recall': recall,
            'low_risk_wrong_rate': low_wrong,
            'risk_quantile_spearman_reported': (
                risk_bins['reported_spearman_rho']
            ),
            'risk_quantile_spearman_recomputed': (
                risk_bins['recomputed_spearman_rho']
            ),
            'risk_quantile_spearman_reported_matches_recomputation': (
                risk_bins['reported_spearman_matches_recomputation']
            ),
            'risk_quantile_pairwise_monotonic_agreement_reported': (
                risk_bins['reported_pairwise_monotonic_agreement']
            ),
            'risk_quantile_pairwise_monotonic_agreement_recomputed': (
                risk_bins['recomputed_pairwise_monotonic_agreement']
            ),
            'risk_quantile_pairwise_monotonic_concordant_count': (
                risk_bins['pairwise_monotonic_concordant_count']
            ),
            'risk_quantile_wrong_rates': risk_bins['wrong_rates'],
            'risk_routing_counts': routing_counts['counts'],
            'risk_metrics_recomputed_from_routing_counts': recomputed_metrics,
            'target_residual_p95': residual.get('p95'),
        },
    }


def main():
    args = parse_args()
    checker_sha256 = file_sha256(Path(__file__).resolve())
    train_path = Path(args.train_json).resolve()
    val_path = Path(args.val_json).resolve()
    output_path = Path(args.output).resolve()
    errors = []

    train = None
    val = None
    try:
        train = load_json(train_path)
    except Exception as exc:
        errors.append(f'train_json: {type(exc).__name__}: {exc}')
    try:
        val = load_json(val_path)
    except Exception as exc:
        errors.append(f'val_json: {type(exc).__name__}: {exc}')

    train_evaluation = (
        evaluate_split(train, 'train') if isinstance(train, dict) else None
    )
    val_evaluation = (
        evaluate_split(val, 'val') if isinstance(val, dict) else None
    )

    train_config = train.get('critical_config') if isinstance(train, dict) else None
    val_config = val.get('critical_config') if isinstance(val, dict) else None
    configs_match = (
        isinstance(train_config, dict)
        and isinstance(val_config, dict)
        and train_config == val_config
    )
    critical_config = train_config if configs_match else None
    config_fingerprint = (
        canonical_fingerprint(critical_config)
        if critical_config is not None else None
    )

    train_cdf_path = resolved_reported_path(train or {}, 'cdf_path')
    val_cdf_path = resolved_reported_path(val or {}, 'cdf_path')
    cdf_paths_match = (
        train_cdf_path is not None and train_cdf_path == val_cdf_path
    )
    cdf_path = train_cdf_path or val_cdf_path
    train_cdf_sha = train.get('cdf_sha256') if isinstance(train, dict) else None
    val_cdf_sha = val.get('cdf_sha256') if isinstance(val, dict) else None
    cdf_shas_match = (
        isinstance(train_cdf_sha, str)
        and bool(train_cdf_sha)
        and train_cdf_sha == val_cdf_sha
    )
    actual_cdf_sha = None
    actual_cdf_exists = False
    if cdf_path is not None:
        actual_cdf_file = Path(cdf_path)
        actual_cdf_exists = actual_cdf_file.is_file()
        if actual_cdf_exists:
            try:
                actual_cdf_sha = file_sha256(actual_cdf_file)
            except Exception as exc:
                errors.append(f'cdf_sha256: {type(exc).__name__}: {exc}')

    train_teacher_sha = (
        train.get('input_provenance', {}).get('teacher_sha256')
        if isinstance(train, dict) else None
    )
    val_teacher_sha = (
        val.get('input_provenance', {}).get('teacher_sha256')
        if isinstance(val, dict) else None
    )
    train_sources = (
        train.get('code_provenance', {}).get('source_sha256')
        if isinstance(train, dict) else None
    )
    val_sources = (
        val.get('code_provenance', {}).get('source_sha256')
        if isinstance(val, dict) else None
    )
    joint_checks = {
        'train_json_loaded': train is not None,
        'val_json_loaded': val is not None,
        'train_split_gate_pass': (
            train_evaluation is not None and train_evaluation['pass']
        ),
        'val_split_gate_pass': (
            val_evaluation is not None and val_evaluation['pass']
        ),
        'cdf_paths_exact_match': cdf_paths_match,
        'cdf_reported_sha256_exact_match': cdf_shas_match,
        'actual_cdf_file_exists': actual_cdf_exists,
        'actual_cdf_sha256_matches_both_reports': (
            actual_cdf_sha is not None
            and actual_cdf_sha == train_cdf_sha
            and actual_cdf_sha == val_cdf_sha
        ),
        'critical_config_exact_match': configs_match,
        'teacher_sha256_exact_match': (
            isinstance(train_teacher_sha, str)
            and bool(train_teacher_sha)
            and train_teacher_sha == val_teacher_sha
        ),
        'native_correctness_semantics_exact_match': (
            isinstance(train, dict)
            and isinstance(val, dict)
            and isinstance(train.get('native_correctness_semantics'), str)
            and bool(train.get('native_correctness_semantics'))
            and train.get('native_correctness_semantics')
            == val.get('native_correctness_semantics')
        ),
        'cdf_metadata_exact_match': (
            isinstance(train, dict)
            and isinstance(val, dict)
            and isinstance(train.get('cdf_metadata'), dict)
            and train.get('cdf_metadata') == val.get('cdf_metadata')
        ),
        'source_sha256_metadata_exact_match': (
            valid_sha256_mapping(train_sources)
            and train_sources == val_sources
        ),
    }
    joint_gate_pass = not errors and all(joint_checks.values())
    result = {
        'schema_version': 1,
        'phase': EXPECTED_PHASE,
        'gate_profile': GATE_PROFILE,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'joint_gate_pass': bool(joint_gate_pass),
        'checker_sha256': checker_sha256,
        'thresholds': {
            'high_risk_coverage_min_inclusive': MIN_HIGH_RISK_COVERAGE,
            'high_risk_coverage_max_inclusive': MAX_HIGH_RISK_COVERAGE,
            'high_risk_enrichment_min_inclusive': MIN_HIGH_RISK_ENRICHMENT,
            'high_risk_wrong_recall_min_inclusive': (
                MIN_HIGH_RISK_WRONG_RECALL
            ),
            'risk_quantile_pairwise_monotonic_agreement_min_inclusive': (
                MIN_RISK_QUANTILE_PAIRWISE_MONOTONIC_AGREEMENT
            ),
            'active_solved_residual_p95_max_exclusive': MAX_RESIDUAL_P95,
        },
        'reliability_definition': {
            'mode': EXPECTED_RELIABILITY_MODE,
            'coefficient_a': EXPECTED_COEFFICIENT_A,
            **EXPECTED_RELIABILITY_DEFINITION,
        },
        'cdf_path': cdf_path,
        'cdf_sha256': train_cdf_sha or val_cdf_sha,
        'actual_cdf_sha256': actual_cdf_sha,
        'train_reported_cdf_sha256': train_cdf_sha,
        'val_reported_cdf_sha256': val_cdf_sha,
        'train_json': str(train_path),
        'val_json': str(val_path),
        'config_fingerprint': config_fingerprint,
        'critical_config': critical_config,
        'joint_checks': joint_checks,
        'train_evaluation': train_evaluation,
        'val_evaluation': val_evaluation,
        'errors': errors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f'wrote={output_path}')
    if args.strict and not joint_gate_pass:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
