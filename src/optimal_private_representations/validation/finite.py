"""Finite-alphabet implementation checks of the recorded uncapped learner.

Source lineage: finite checks from verification/training_learning_checks.py,
plus ordered/exhaustive checks from verification/ordered_private_learning.py
in the supplied 20261122 JMLR-core research archive. Adaptations: pure-alpha
API only, no smooth-grid cases, and explicit current model-field checks.

These deterministic checks cover ideal Laplace transcript identities and
numerical implementation consistency. They neither prove the statistical
minimax theorem nor certify floating-point differential privacy. Seeds are
for reproducible synthetic tests only; no private dataset is included.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from itertools import product
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from ..legacy.calibration import fit_ordered_private_dp
from ..legacy.learner import (
    FinitePrivateTaskModel,
    fit_finite_private_task_learner,
    fractional_histogram,
    rescale_nonnegative_block,
    training_parameters,
)

TOLERANCE = 2e-10


def fit_training_private(states, labels, cells, epsilon_query, alpha_train, rng=None):
    """Test-local adapter keeping the archive checks easy to compare."""
    return fit_finite_private_task_learner(
        states, labels, cells=cells, epsilon_query=epsilon_query,
        alpha_train=alpha_train, rng=rng)


def randomized_response(bins, epsilon):
    """Independent small-case RR matrix for the exhaustive test reference."""
    bins = np.asarray(bins, dtype=int)
    k = int(bins.max()) + 1
    result = np.empty((len(bins), k))
    for i, assigned in enumerate(bins):
        for j in range(k):
            result[i, j] = (math.exp(epsilon) if assigned == j else 1.0) / (k - 1 + math.exp(epsilon))
    return result

class ScriptedNoise:
    """Supply exact perturbations and verify every requested noise coordinate."""

    def __init__(self, perturbations):
        self.perturbations = [np.asarray(value, dtype=float)
                              for value in perturbations]
        self.calls = []

    def laplace(self, loc, scale, size=None):
        index = len(self.calls)
        if index >= len(self.perturbations):
            raise AssertionError("unexpected additional noise request")
        value = self.perturbations[index]
        shape = () if size is None else tuple(size)
        assert value.shape == shape
        assert loc == 0 and np.isfinite(scale) and scale > 0
        self.calls.append({"scale": float(scale), "shape": list(shape)})
        return float(value) if size is None else value.copy()

    def assert_consumed(self):
        assert len(self.calls) == len(self.perturbations)


class NoNoise:
    def laplace(self, *args, **kwargs):
        raise AssertionError("data-independent branch requested noise")


def manual_histogram(states, labels, cells, normalized=False):
    """Independent record-by-record reference, without bincount."""
    result = np.zeros((cells, 2))
    for state, label in zip(states, labels):
        result[state, 0] += 1 - label
        result[state, 1] += label
    return result / len(states) if normalized else result


def manual_rescale(values, target):
    positive = [max(float(value), 0.0) for value in values]
    total = math.fsum(positive)
    if total == 0:
        return np.array([target] + [0.0] * (len(positive) - 1))
    return np.array([target * value / total for value in positive])


def model_equal(first, second):
    first, second = asdict(first), asdict(second)
    assert first.keys() == second.keys()
    for name, value in first.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, second[name], rtol=0, atol=2e-14)
        else:
            assert value == second[name], name


def raw_transcript_centers(states, labels, cells):
    n1 = len(states) // 2
    return (
        manual_histogram(states[:n1], labels[:n1], cells),
        manual_histogram(states[n1:], labels[n1:], cells, normalized=True),
        np.asarray(float(np.mean(labels[n1:]))),
    )


def check_sensitivities_and_transcript():
    """Replace complete records in either fixed sample block, never both."""
    states = np.array([0, 1, 0, 2, 1, 2, 0, 3])
    labels = np.array([0., 1., .25, .8, 0., 1., .4, .6])
    cells, n = 4, len(states)
    n1, calibration_size = n // 2, n - n // 2
    alpha = .73
    scales = (2 / alpha, 4 / (calibration_size * alpha),
              2 / (calibration_size * alpha))
    base = raw_transcript_centers(states, labels, cells)
    max_count_change = max_normalized_change = max_mean_change = 0.0
    maximum_log_ratio = 0.0
    neighbors = density_checks = pilot_changes = calibration_changes = 0
    for changed in range(n):
        for state, label in product(range(cells), [0., .25, 1.]):
            alternative_states, alternative_labels = states.copy(), labels.copy()
            alternative_states[changed], alternative_labels[changed] = state, label
            if np.array_equal(alternative_states, states) and np.array_equal(alternative_labels, labels):
                continue
            alternative = raw_transcript_centers(alternative_states, alternative_labels, cells)
            neighbors += 1
            block_changes = [float(np.abs(left - right).sum())
                             for left, right in zip(base, alternative)]
            if changed < n1:
                pilot_changes += 1
                assert block_changes[1] == block_changes[2] == 0
                max_count_change = max(max_count_change, block_changes[0])
                assert block_changes[0] <= 2 + TOLERANCE
            else:
                calibration_changes += 1
                assert block_changes[0] == 0
                max_normalized_change = max(max_normalized_change, block_changes[1])
                max_mean_change = max(max_mean_change, block_changes[2])
                assert block_changes[1] <= 2 / calibration_size + TOLERANCE
                assert block_changes[2] <= 1 / calibration_size + TOLERANCE
            # An adjacent dataset changes just one block. The calibration
            # histogram and mean use alpha/2 each; the pilot uses alpha.
            sensitivity_bound = sum(change / scale
                                    for change, scale in zip(block_changes, scales))
            assert sensitivity_bound <= alpha + TOLERANCE
            transcripts = [base, alternative]
            transcripts.append(tuple(left + 20 * np.sign(left - right)
                                     for left, right in zip(base, alternative)))
            transcripts.append(tuple(np.arange(left.size).reshape(left.shape) / 7 - 1.3
                                     for left in base))
            for output in transcripts:
                # Normalizers cancel because scales and dimensions are public.
                log_ratio = sum(float((np.abs(value - right) -
                                       np.abs(value - left)).sum()) / scale
                                for value, left, right, scale in
                                zip(output, base, alternative, scales))
                assert abs(log_ratio) <= sensitivity_bound + TOLERANCE
                maximum_log_ratio = max(maximum_log_ratio, abs(log_ratio))
                density_checks += 1
    # The bound is attained by some complete-record replacements, so a
    # smaller sensitivity caused by protecting labels alone would fail here.
    np.testing.assert_allclose([max_count_change, max_normalized_change, max_mean_change],
                               [2, 2 / calibration_size, 1 / calibration_size], atol=TOLERANCE)
    assert maximum_log_ratio >= alpha - TOLERANCE
    np.testing.assert_allclose(fractional_histogram(states, labels, cells),
                               manual_histogram(states, labels, cells))
    return {
        "neighbor_pairs": neighbors,
        "pilot_record_replacements": pilot_changes,
        "calibration_record_replacements": calibration_changes,
        "ideal_density_identity_checks": density_checks,
        "maximum_count_histogram_l1_change": max_count_change,
        "maximum_normalized_histogram_l1_change": max_normalized_change,
        "maximum_mean_change": max_mean_change,
        "pure_budget": alpha,
        "maximum_absolute_transcript_log_density_ratio": maximum_log_ratio,
        "accounting": "One changed record affects one fixed block; calibration splits alpha into two halves.",
    }


def check_rescaling():
    cases = [
        ([-4., -2., -1.], .7), ([0., 0., 0.], 1.),
        ([1., 3., -5.], .25), ([1., 3., -5.], 0.),
        ([1e308, 1e308, -1e308], .6),
        ([np.nextafter(0., 1.), np.nextafter(0., 1.), 0.], .8),
    ]
    mass_error = 0.0
    for vector, target in cases:
        result = rescale_nonnegative_block(vector, target)
        assert np.all(np.isfinite(result)) and np.all(result >= 0)
        mass_error = max(mass_error, abs(float(result.sum()) - target))
        assert abs(result.sum() - target) <= TOLERANCE
    np.testing.assert_allclose(rescale_nonnegative_block([-4., -2., -1.], .7), [.7, 0., 0.])
    np.testing.assert_allclose(rescale_nonnegative_block([1e308, 1e308, -1e308], .6), [.3, .3, 0.])
    # Verify the deterministic perturbation inequality, including empty blocks.
    perturbation_residual = 0.0
    bound_cases = 0
    for original in [[0., 0., 0.], [.1, .2, .3], [.4, 0., .6]]:
        original = np.array(original)
        for noise in [[-2., -2., -2.], [.5, -.4, .2], [0., 0., 0.]]:
            noise = np.array(noise)
            for target in [0., .3, 1.]:
                result = rescale_nonnegative_block(original + noise, target)
                residual = (np.abs(result - original).sum() -
                            2 * np.abs(noise).sum() - abs(target - original.sum()))
                perturbation_residual = max(perturbation_residual, float(residual))
                assert residual <= TOLERANCE
                bound_cases += 1
    return {"mass_and_edge_cases": len(cases), "rescaling_bound_cases": bound_cases,
            "maximum_mass_error": mass_error,
            "maximum_positive_rescaling_bound_residual": perturbation_residual}


def check_controlled_construction():
    cells = 6
    states = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    labels = np.array([0., 1., .2, .9, .1, .8, .05, .9, .3, .75, .15, .95])
    pilot_noise = np.array([[.2, -.1], [-.2, .3], [2., 0.],
                            [0., 3.], [-1., -2.], [1., .4]])
    calibration_noise = np.array([[.02, .04], [.06, -.03], [.03, .02],
                                  [-.01, .08], [.09, .01], [.02, -.01]])
    scenarios = [
        (pilot_noise, calibration_noise, -.12, float(np.log(4))),
        (pilot_noise, np.full((cells, 2), -3.), -.7, float(np.log(2))),
        (pilot_noise, np.full((cells, 2), -3.), .8, 0.),
    ]
    maximum_error = exhaustive_error = 0.0
    absent_cells = cells - len(np.unique(states))
    for pilot_error, histogram_error, mean_error, epsilon_query in scenarios:
        rng = ScriptedNoise([pilot_error, histogram_error, mean_error])
        model = fit_training_private(states, labels, cells, epsilon_query, .7, rng=rng)
        rng.assert_consumed()
        _, alpha = training_parameters(.7)
        expected_scales = [2 / alpha, 4 / (6 * alpha), 2 / (6 * alpha)]
        np.testing.assert_allclose([call["scale"] for call in rng.calls], expected_scales)
        assert [call["shape"] for call in rng.calls] == [[cells, 2], [cells, 2], []]
        original_pilot, original_histogram, original_mean = raw_transcript_centers(states, labels, cells)
        noisy_pilot = np.maximum(original_pilot + pilot_error, 0)
        expected_scores = ((noisy_pilot[:, 1] + .5) /
                           (noisy_pilot[:, 0] + noisy_pilot[:, 1] + 1))
        expected_mean = min(1., max(0., float(original_mean) + mean_error))
        expected_joint = np.column_stack([
            manual_rescale(original_histogram[:, 0] + histogram_error[:, 0], 1 - expected_mean),
            manual_rescale(original_histogram[:, 1] + histogram_error[:, 1], expected_mean),
        ])
        for actual, expected in [(model.pilot_scores, expected_scores),
                                 (model.sanitized_p, expected_joint.sum(axis=1)),
                                 (model.sanitized_s, expected_joint[:, 1])]:
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, atol=TOLERANCE)
        assert model.sanitized_mean == expected_mean
        assert np.any(np.abs(model.pilot_scores[2:] - .5) > .1)
        assert np.all(model.sanitized_s >= 0)
        assert np.all(model.sanitized_s <= model.sanitized_p)
        np.testing.assert_allclose([model.sanitized_p.sum(), model.sanitized_s.sum()],
                                   [1, model.sanitized_mean], atol=TOLERANCE)
        rows = model.probabilities(np.arange(cells))
        bayes = (model.sanitized_s @ rows) / (model.sanitized_p @ rows)
        np.testing.assert_allclose(model.decoder, bayes, atol=TOLERANCE)
        actual_risk = (model.sanitized_mean - 2 * model.sanitized_s @ rows @ model.decoder
                       + model.sanitized_p @ rows @ (model.decoder ** 2))
        exhaustive = exhaustive_ordered_isotonic(model.sanitized_p, model.sanitized_s,
                                                model.pilot_scores, epsilon_query)
        exhaustive_error = max(exhaustive_error, abs(float(actual_risk) - exhaustive))
        assert abs(actual_risk - exhaustive) <= TOLERANCE
        if epsilon_query == 0:
            np.testing.assert_allclose(model.decoder, [expected_mean])
            np.testing.assert_array_equal(rows, np.ones((cells, 1)))
    return {"controlled_transcripts": len(scenarios), "declared_cells": cells,
            "unobserved_cells_receiving_pilot_and_calibration_noise": absent_cells,
            "maximum_manual_construction_difference": maximum_error,
            "maximum_exhaustive_ordered_risk_difference": exhaustive_error}


def check_release_postprocessing():
    cells, n = 5, 10
    first_states = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    first_labels = np.linspace(0, 1, n)
    second_states = np.array([4, 3, 2, 4, 2, 3, 4, 3, 2, 4])
    second_labels = np.linspace(.9, .1, n)
    target_transcript = (np.arange(2 * cells).reshape(cells, 2) / 4 + .2,
                         np.array([[.1, .2], [.05, .1], [.4, .2], [.1, .3], [.2, .02]]),
                         np.asarray(.37))
    outputs = []
    for states, labels in [(first_states, first_labels), (second_states, second_labels)]:
        centers = raw_transcript_centers(states, labels, cells)
        rng = ScriptedNoise([target - center for target, center in zip(target_transcript, centers)])
        outputs.append(fit_training_private(states, labels, cells, np.log(4), 1., rng=rng))
        rng.assert_consumed()
    model_equal(*outputs)
    expected_fields = {
        "cells", "sample_size", "epsilon_query", "alpha_train",
        "kappa", "pure_training_budget", "pilot_scores", "thresholds", "decoder",
        "sanitized_p", "sanitized_s", "sanitized_mean", "branch",
    }
    assert {field.name for field in fields(FinitePrivateTaskModel)} == expected_fields
    constants = 0
    for alpha_train, size in [(0., 10), (.001, 10), (1., 1)]:
        a = fit_training_private(np.zeros(size, dtype=int), np.zeros(size), cells,
                                 np.log(2), alpha_train, rng=NoNoise())
        b = fit_training_private(np.full(size, cells - 1), np.ones(size), cells,
                                 np.log(2), alpha_train, rng=NoNoise())
        model_equal(a, b)
        assert a.branch == "data_independent_constant"
        constants += 1
    return {"different_raw_datasets_with_identical_private_transcript": len(outputs),
            "returned_fields_are_public_or_postprocessed": True,
            "data_independent_branch_checks": constants,
            "raw_counts_labels_support_or_empirical_risk_returned": False}


def score_cdf_errors(scores, population_p, population_s, empirical_p, empirical_s):
    groups, inverse = np.unique(scores, return_inverse=True)
    delta_p = np.bincount(inverse, weights=population_p - empirical_p, minlength=len(groups))
    delta_s = np.bincount(inverse, weights=population_s - empirical_s, minlength=len(groups))
    return float(np.abs(np.cumsum(delta_p)).max()), float(np.abs(np.cumsum(delta_s)).max())


def risk(p, s, q, decoder):
    return float(s.sum() - 2 * s @ q @ decoder + p @ q @ (decoder ** 2))


def centered_loss_expectation(p, s, q, decoder, center):
    centered = decoder - center
    u, v = q @ centered, q @ (centered ** 2)
    return float(p @ v - 2 * (s - center * p) @ u)


def check_centered_risk_and_query_channel(seed):
    rng = np.random.default_rng(seed)
    inequalities = candidates = 0
    maximum_cdf_residual = maximum_histogram_residual = maximum_risk_residual = 0.0
    maximum_privacy_residual = maximum_span_residual = maximum_centering_error = 0.0
    maximum_exhaustive_difference = 0.0
    zero_query_cases = 0
    for cells, epsilon_query, replicate in product([3, 5, 7], [0., .08, np.log(2), np.log(4)], range(3)):
        sample_size = [20, 31, 48][replicate]
        population_p = rng.dirichlet(np.full(cells, .7))
        population_m = rng.uniform(.02, .98, cells)
        population_s = population_p * population_m
        states = rng.choice(cells, sample_size, p=population_p)
        labels = (rng.binomial(1, population_m[states]).astype(float) if replicate % 2 == 0
                  else rng.uniform(0, 1, sample_size))
        model = fit_training_private(states, labels, cells, epsilon_query, .7,
                                     rng=np.random.default_rng(seed + inequalities + 1))
        assert model.branch == "private_pilot_and_calibration"
        raw_joint = manual_histogram(states[sample_size // 2:], labels[sample_size // 2:],
                                     cells, normalized=True)
        empirical_p, empirical_s = raw_joint.sum(axis=1), raw_joint[:, 1]
        sanitized_joint = np.column_stack([model.sanitized_p - model.sanitized_s,
                                          model.sanitized_s])
        discrepancy = float(np.abs(sanitized_joint - raw_joint).sum())
        assert discrepancy <= 2 + TOLERANCE
        rho = math.tanh(epsilon_query / 2)
        dp, ds = score_cdf_errors(model.pilot_scores, population_p, population_s,
                                  empirical_p, empirical_s)
        deviation_bound = 3 * rho ** 2 * dp + 4 * rho ** 2 * ds
        fitted_q = model.probabilities(np.arange(cells))
        comparator = fit_ordered_private_dp(population_p, population_s,
                                            model.pilot_scores, epsilon_query)
        comparator_q, comparator_a = comparator["q"], comparator["decoder"]
        shift = model.sanitized_mean - model.sanitized_p @ comparator_q @ comparator_a
        centered_comparator = comparator_a + shift
        for q, decoder in [(fitted_q, model.decoder), (comparator_q, centered_comparator)]:
            center_error = abs(model.sanitized_p @ q @ decoder - model.sanitized_mean)
            maximum_centering_error = max(maximum_centering_error, float(center_error))
            assert center_error <= TOLERANCE
            p_loss = centered_loss_expectation(population_p, population_s, q, decoder, model.sanitized_mean)
            empirical_loss = centered_loss_expectation(empirical_p, empirical_s, q, decoder, model.sanitized_mean)
            private_loss = centered_loss_expectation(model.sanitized_p, model.sanitized_s,
                                                     q, decoder, model.sanitized_mean)
            cdf_residual = abs(p_loss - empirical_loss) - deviation_bound
            histogram_residual = abs(empirical_loss - private_loss) - 3 * rho ** 2 * discrepancy
            maximum_cdf_residual = max(maximum_cdf_residual, cdf_residual)
            maximum_histogram_residual = max(maximum_histogram_residual, histogram_residual)
            assert max(cdf_residual, histogram_residual) <= TOLERANCE
            candidates += 1
        residual = (risk(population_p, population_s, fitted_q, model.decoder)
                    - risk(population_p, population_s, comparator_q, comparator_a)
                    - shift ** 2 - 2 * deviation_bound - 6 * rho ** 2 * discrepancy)
        maximum_risk_residual = max(maximum_risk_residual, residual)
        assert residual <= TOLERANCE
        privacy_residual = float(np.max(fitted_q.max(axis=0)
                                        - math.exp(epsilon_query) * fitted_q.min(axis=0)))
        maximum_privacy_residual = max(maximum_privacy_residual, privacy_residual)
        maximum_span_residual = max(maximum_span_residual, float(np.ptp(model.decoder) - rho))
        np.testing.assert_allclose(fitted_q.sum(axis=1), 1, atol=TOLERANCE)
        assert np.all(fitted_q > 0)
        assert max(privacy_residual, np.ptp(model.decoder) - rho) <= TOLERANCE
        exhaustive = exhaustive_ordered_isotonic(model.sanitized_p, model.sanitized_s,
                                                model.pilot_scores, epsilon_query)
        exhaustive_difference = abs(risk(model.sanitized_p, model.sanitized_s,
                                         fitted_q, model.decoder) - exhaustive)
        maximum_exhaustive_difference = max(maximum_exhaustive_difference, exhaustive_difference)
        assert exhaustive_difference <= TOLERANCE
        if epsilon_query == 0:
            zero_query_cases += 1
            np.testing.assert_allclose(model.decoder, [model.sanitized_mean], atol=TOLERANCE)
            np.testing.assert_array_equal(fitted_q, np.ones((cells, 1)))
            indices = model.sample_representations(np.arange(cells), np.random.default_rng(0))
            np.testing.assert_array_equal(indices, np.zeros(cells, dtype=int))
        inequalities += 1
    return {
        "deterministic_dataset_cases": inequalities,
        "centered_candidates_checked": candidates,
        "zero_query_privacy_cases": zero_query_cases,
        "maximum_positive_cdf_bound_residual": maximum_cdf_residual,
        "maximum_positive_histogram_bound_residual": maximum_histogram_residual,
        "maximum_positive_centered_risk_bound_residual": maximum_risk_residual,
        "maximum_positive_full_alphabet_query_privacy_residual": maximum_privacy_residual,
        "maximum_positive_decoder_span_residual": maximum_span_residual,
        "maximum_sanitized_centering_error": maximum_centering_error,
        "maximum_exhaustive_ordered_risk_difference": maximum_exhaustive_difference,
    }


def weighted_isotonic(values, weights):
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index + 1, weight, weight * value])
        while len(blocks) >= 2 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
    result = np.empty(len(values))
    for left, right, weight, total in blocks:
        result[left:right] = total / weight
    return result


def exhaustive_ordered_isotonic(p, s, scores, epsilon):
    """Independent exhaustive partitions with a weighted isotonic decoder."""
    distinct, inverse = np.unique(scores, return_inverse=True)
    gp = np.bincount(inverse, weights=p)
    gs = np.bincount(inverse, weights=s)
    active = gp > 0
    gp, gs = gp[active], gs[active]
    smallest = np.inf
    for cuts in product([0, 1], repeat=len(gp) - 1):
        bins = np.r_[0, np.cumsum(cuts)]
        q = randomized_response(bins, epsilon)
        mass, moment = gp @ q, gs @ q
        decoder = weighted_isotonic(moment / mass, mass)
        risk = float(gs.sum() - 2 * (moment @ decoder) + mass @ decoder ** 2)
        smallest = min(smallest, risk)
    return smallest


def verify_ordered_algorithm(seed=20261124):
    rng = np.random.default_rng(seed)
    differences = []
    privacy_residual = mean_residual = decoder_residual = variation_residual = 0.0
    empty_cases = tied_cases = 0
    for cells in [3, 5, 8]:
        for epsilon in [0.0, .03, float(np.log(2)), float(np.log(4))]:
            for replicate in range(6):
                population_p = rng.dirichlet(np.full(cells, .8))
                population_m = rng.uniform(0, 1, cells)
                counts = rng.multinomial([2, 7, 30, 100, 12, 5][replicate],
                                        np.column_stack([population_p * (1-population_m),
                                                         population_p * population_m]).ravel()).reshape(cells, 2)
                p = counts.sum(axis=1) / counts.sum()
                s = counts[:, 1] / counts.sum()
                scores = rng.uniform(0, 1, cells)
                if replicate % 2:
                    scores = np.round(scores, 1)
                fitted = fit_ordered_private_dp(p, s, scores, epsilon)
                exhaustive = exhaustive_ordered_isotonic(p, s, scores, epsilon)
                differences.append(abs(exhaustive - (s.sum() - fitted["explained_second_moment"])))
                empty_cases += int(np.any(p == 0))
                tied_cases += int(len(np.unique(scores)) < cells)
                q, a = fitted["q"], fitted["decoder"]
                rho = float(np.tanh(epsilon / 2))
                privacy_residual = max(privacy_residual, float(np.max(q.max(axis=0) - np.exp(epsilon) * q.min(axis=0))))
                decoder_residual = max(decoder_residual, float(np.ptp(a) - rho))
                mean_residual = max(mean_residual, abs(float(p @ q @ a - s.sum())))
                centered = a - s.sum()
                order = np.argsort(scores, kind="stable")
                u, v = (q @ centered)[order], (q @ (centered ** 2))[order]
                assert np.min(np.diff(u), initial=0.0) >= -1e-12
                variation_residual = max(variation_residual,
                                         float(np.abs(np.diff(v)).sum() - rho ** 3))
    close_scores = np.array([.5, np.nextafter(.5, 1.0)])
    close_fit = fit_ordered_private_dp(np.array([.5, .5]),
                                      np.array([0., .5]), close_scores, np.log(2))
    assert np.array_equal(close_fit["bins"], [0, 1])
    assert max(differences) < 1e-10
    assert max(privacy_residual, mean_residual, decoder_residual, variation_residual) < 1e-10
    return {"seed": seed, "exhaustive_isotonic_checks": len(differences),
            "checks_with_empty_cells": empty_cases, "checks_with_tied_scores": tied_cases,
            "maximum_dp_exhaustive_difference": max(differences),
            "maximum_privacy_residual": max(0.0, privacy_residual),
            "maximum_centering_residual": mean_residual,
            "maximum_decoder_range_residual": max(0.0, decoder_residual),
            "maximum_centered_second_moment_variation_residual": max(0.0, variation_residual)}

def check_pure_budget_edges_and_reproducibility(seed):
    budgets = [0., 1e-12, .001, .02, .3, 1., 10., 1000.]
    for alpha_train in budgets:
        kappa, alpha0 = training_parameters(alpha_train)
        assert kappa == math.tanh(alpha_train / 2)
        assert alpha0 == min(1., alpha_train)
        assert 0 <= kappa <= alpha0 <= alpha_train
    states = np.tile(np.arange(5), 8)
    labels = np.linspace(0, 1, len(states))
    first = fit_training_private(states, labels, 5, np.log(2), .7,
                                 rng=np.random.default_rng(seed))
    second = fit_training_private(states, labels, 5, np.log(2), .7,
                                  rng=np.random.default_rng(seed))
    model_equal(first, second)
    assert first.branch == "private_pilot_and_calibration"
    queries = np.arange(first.cells)
    representations = first.sample_representations(queries, np.random.default_rng(seed + 1))
    predictions = first.predict(queries, np.random.default_rng(seed + 1))
    np.testing.assert_array_equal(predictions, first.decode(representations))
    assert np.all((predictions >= 0) & (predictions <= 1))
    assert first.probabilities(np.array([], dtype=int)).shape == (0, len(first.decoder))
    assert first.predict(np.array([], dtype=int), np.random.default_rng(seed)).shape == (0,)
    for epsilon in [0., np.log(4)]:
        single = fit_training_private(np.zeros(20, dtype=int), np.linspace(0, 1, 20),
                                      1, epsilon, .7, rng=np.random.default_rng(seed))
        np.testing.assert_allclose(single.decoder, [single.sanitized_mean])
        np.testing.assert_array_equal(single.probabilities(np.array([0])), [[1.]])
    return {
        "pure_training_budget_edge_cases": len(budgets),
        "single_cell_cases": 2,
        "fitting_seed_reproducibility_checked": True,
        "sample_decode_predict_consistency_checked": True,
        "empty_query_arrays_checked": True,
        "seeds_are_test_only_not_part_of_privacy_guarantee": True,
    }


def run_checks(seed=20261129):
    started = time.perf_counter()
    output = {
        "status": "passed",
        "scope": "Finite alphabet, pure complete-record training DP and all-pairs query DP; deterministic ideal-transcript and numerical consistency checks only.",
        "seed_for_reproducible_checks_only": seed,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "full_record_sensitivity_and_parallel_accounting": check_sensitivities_and_transcript(),
        "histogram_rescaling": check_rescaling(),
        "controlled_noise_construction": check_controlled_construction(),
        "release_postprocessing": check_release_postprocessing(),
        "centered_risk_and_full_alphabet_query_privacy": check_centered_risk_and_query_channel(seed),
        "ordered_dp_exhaustive_and_tie_checks": verify_ordered_algorithm(seed + 1000),
        "pure_budget_edges_and_reproducibility": check_pure_budget_edges_and_reproducibility(seed + 2000),
    }
    output["elapsed_seconds"] = time.perf_counter() - started
    return output


if __name__ == "__main__":
    result = run_checks()
    from ..io import prepare_output
    target = prepare_output("outputs/validation") / "finite_checks.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
