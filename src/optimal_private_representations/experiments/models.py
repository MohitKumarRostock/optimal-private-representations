"""Finite-law baselines and analytic population-risk evaluation.

All experiment fits explicitly use the legacy uncapped optimizer to preserve
the recorded algorithms and timing provenance. All methods except
nonprivate_fit have the stated ideal central privacy accounting. Every
returned query channel is defined on the full declared input alphabet.
fit_seconds is an internal diagnostic, not a private release. Evaluating a
channel row or its mean is not a private release of a confidential query.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..legacy.learner import (
    _labels, _positive_integer, _query_privacy, _states,
    fit_finite_private_task_learner, fractional_histogram,
    rescale_nonnegative_block, training_parameters,
)
from ..legacy.calibration import fit_ordered_private_dp


METHODS = (
    "attaining", "direct_histogram", "private_mean",
    "no_mean_calibration", "public_order", "nonprivate_fit",
)


def _constant(cells, value=0.5):
    return {"q": np.ones((cells, 1)),
            "decoder": np.array([float(value)]), "k": 1}


def _normalize_joint(noisy_joint):
    """Postprocess the entire clipped histogram; use a fixed public fallback."""
    clipped = np.maximum(np.asarray(noisy_joint, dtype=float), 0.0)
    if clipped.ndim != 2 or clipped.shape[1] != 2 or not np.all(np.isfinite(clipped)):
        raise ValueError("joint histogram must be a finite cells-by-two array")
    maximum = float(clipped.max())
    if maximum == 0:
        return np.full(clipped.shape, 1.0 / clipped.size)
    scaled = clipped / maximum
    return scaled / scaled.sum()


def _moments_and_scores(joint):
    p = joint.sum(axis=1)
    s = joint[:, 1]
    scores = np.full(len(p), 0.5)
    np.divide(s, p, out=scores, where=p > 0)
    return p, s, scores


def _pilot(states, labels, cells, alpha0, rng):
    # Keep the exact recorded uncapped pilot call shape/order to couple synthetic runs.
    histogram = fractional_histogram(states, labels, cells)
    noisy = np.maximum(histogram + rng.laplace(
        0.0, 2.0 / alpha0, size=(cells, 2)), 0.0)
    return (noisy[:, 1] + 0.5) / (noisy.sum(axis=1) + 1.0)


def _calibrate_mean(states, labels, cells, alpha0, rng):
    """The unchanged mathematical recorded uncapped half-budget calibration recipe."""
    observations = len(states)
    histogram = fractional_histogram(states, labels, cells, normalize=True)
    noisy = histogram + rng.laplace(
        0.0, 4.0 / (observations * alpha0), size=(cells, 2))
    mean = float(np.clip(labels.mean() + rng.laplace(
        0.0, 2.0 / (observations * alpha0)), 0.0, 1.0))
    joint = np.column_stack([
        rescale_nonnegative_block(noisy[:, 0], 1.0 - mean),
        rescale_nonnegative_block(noisy[:, 1], mean),
    ])
    return joint


def fit_model(method, states, labels, cells, epsilon_query, alpha_train, rng=None):
    """Fit one named baseline on the common complete-record fitting sample.

    Return keys include q (cells by k), decoder (k,), k, method, branch,
    alpha_train, pure_training_budget, epsilon_query, sample_size, cells,
    training_private, and fit_seconds. The final field is a nonprivate timing
    diagnostic. RNG seeds shared across synthetic methods are for simulation;
    no joint DP guarantee for releasing all fitted models is asserted.

    At alpha_train=0 all training-private methods return the data-independent
    constant 1/2. nonprivate_fit ignores that budget and remains nonprivate.
    At epsilon_query=0 the channel is constant across inputs, while its decoder
    may still be estimated from fitting data subject to the training budget.
    """
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    cells = _positive_integer(cells, "cells")
    states = _states(states, cells)
    labels = _labels(labels, len(states))
    epsilon_query = _query_privacy(epsilon_query)
    kappa, alpha0 = training_parameters(alpha_train)
    rng = np.random.default_rng() if rng is None else rng
    n = len(states)
    branch = method

    if method == "nonprivate_fit":
        joint = fractional_histogram(states, labels, cells, normalize=True)
        p, s, scores = _moments_and_scores(joint)
        fitted = fit_ordered_private_dp(p, s, scores, epsilon_query)
    elif alpha0 == 0.0:
        fitted = _constant(cells)
        branch = "data_independent_zero_training_budget"
    elif method == "attaining":
        model = fit_finite_private_task_learner(
            states, labels, cells=cells, epsilon_query=epsilon_query,
            alpha_train=alpha_train, rng=rng)
        fitted = {"q": model.probabilities(np.arange(cells)),
                  "decoder": model.decoder, "k": len(model.decoder)}
        branch = model.branch
    elif method in ("no_mean_calibration", "public_order") and (n == 1 or n * kappa < 1):
        # Match the attaining procedure's public, data-independent branch.
        fitted = _constant(cells)
        branch = "data_independent_small_sample"
    elif method == "private_mean":
        scale = 1.0 / (n * alpha0)
        if not math.isfinite(scale):
            fitted = _constant(cells)
            branch = "data_independent_unrepresentable_noise_scale"
        else:
            mean = float(np.clip(labels.mean() + rng.laplace(0.0, scale), 0.0, 1.0))
            fitted = _constant(cells, mean)
    elif method == "direct_histogram":
        scale = 2.0 / (n * alpha0)
        if not math.isfinite(scale):
            fitted = _constant(cells)
            branch = "data_independent_unrepresentable_noise_scale"
        else:
            histogram = fractional_histogram(states, labels, cells, normalize=True)
            joint = _normalize_joint(histogram + rng.laplace(
                0.0, scale, size=(cells, 2)))
            p, s, scores = _moments_and_scores(joint)
            fitted = fit_ordered_private_dp(p, s, scores, epsilon_query)
    else:
        pilot_size = n // 2
        calibration_states, calibration_labels = states[pilot_size:], labels[pilot_size:]
        if method == "no_mean_calibration":
            scores = _pilot(states[:pilot_size], labels[:pilot_size], cells, alpha0, rng)
            histogram = fractional_histogram(
                calibration_states, calibration_labels, cells, normalize=True)
            joint = _normalize_joint(histogram + rng.laplace(
                0.0, 2.0 / (len(calibration_states) * alpha0), size=(cells, 2)))
        else:  # public_order: numeric indices, no pilot fit or pilot randomness.
            scores = np.arange(cells, dtype=float)
            joint = _calibrate_mean(
                calibration_states, calibration_labels, cells, alpha0, rng)
        p, s = joint.sum(axis=1), joint[:, 1]
        fitted = fit_ordered_private_dp(p, s, scores, epsilon_query)

    q = np.array(fitted["q"], dtype=float, copy=True)
    decoder = np.array(fitted["decoder"], dtype=float, copy=True)
    return {
        "q": q, "decoder": decoder, "k": int(fitted["k"]),
        "method": method, "branch": branch,
        "cells": cells, "sample_size": n,
        "epsilon_query": epsilon_query, "alpha_train": float(alpha_train),
        "pure_training_budget": None if method == "nonprivate_fit" else alpha0,
        "training_private": method != "nonprivate_fit",
        "fit_seconds": time.perf_counter() - started,
        "nonprivate_diagnostic_fields": ("fit_seconds",),
    }


def _law(p, m):
    p, m = np.asarray(p, dtype=float), np.asarray(m, dtype=float)
    if (p.ndim != 1 or m.shape != p.shape or not len(p)
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(m))
            or np.any(p < 0) or not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-10)
            or np.any(m < 0) or np.any(m > 1)):
        raise ValueError("p must be a probability vector and m a matching vector in [0,1]")
    return p / p.sum(), m


def analytic_risk(p, m, second, q, decoder):
    """Exactly integrate fitting-independent target and query randomness.

    p[i]=P(H=i), m[i]=E[Y|H=i]. second can be the vector E[Y^2|H=i]
    (use m for Bernoulli targets) or the scalar unconditional E[Y^2].
    No fresh query/label Monte Carlo is used. The fitting sample and fitted
    model are treated as fixed in this evaluation.
    """
    p, m = _law(p, m)
    q, decoder = np.asarray(q, dtype=float), np.asarray(decoder, dtype=float)
    if (q.ndim != 2 or q.shape[0] != len(p) or decoder.shape != (q.shape[1],)
            or q.shape[1] == 0 or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(decoder)) or np.any(q < 0)
            or not np.allclose(q.sum(axis=1), 1.0, rtol=0, atol=1e-10)
            or np.any(decoder < -1e-12) or np.any(decoder > 1.0 + 1e-12)):
        raise ValueError("q must be a stochastic matrix and decoder a matching bounded vector")
    second = np.asarray(second, dtype=float)
    if second.ndim == 0:
        if (not np.isfinite(second) or second < p @ (m * m) - 1e-12
                or second > p @ m + 1e-12):
            raise ValueError("unconditional second moment is inconsistent with a [0,1] target")
        target_variance = float(second - p @ (m * m))
    elif second.shape == m.shape:
        if (not np.all(np.isfinite(second)) or np.any(second < m * m - 1e-12)
                or np.any(second > m + 1e-12)):
            raise ValueError("conditional second moments must lie between m^2 and m")
        target_variance = float(p @ (second - m * m))
    else:
        raise ValueError("second must be a scalar or one conditional moment per cell")
    routing_error = float(np.sum(p[:, None] * q * (m[:, None] - decoder[None, :]) ** 2))
    return max(0.0, target_variance + routing_error)


# Short alias for experiment scripts using the mathematical risk notation.
risk = analytic_risk


def population_oracle(p, m, epsilon_query, second=None):
    """Known-law private oracle using true conditional-mean ordering.

    This comparator is not learned and has no training privacy requirement.
    Return q, decoder, k and population explained moments. Include risk when
    second is provided; no target-noise model is assumed when it is omitted.
    """
    started = time.perf_counter()
    p, m = _law(p, m)
    epsilon_query = _query_privacy(epsilon_query)
    fitted = fit_ordered_private_dp(p, p * m, m, epsilon_query)
    result = {
        "q": fitted["q"].copy(), "decoder": fitted["decoder"].copy(),
        "k": int(fitted["k"]), "method": "population_oracle",
        "epsilon_query": epsilon_query, "training_private": False,
        "population_mean": float(p @ m),
        "explained_second_moment": float(fitted["explained_second_moment"]),
        "explained_variance": float(fitted["explained_second_moment"] - (p @ m) ** 2),
        "fit_seconds": time.perf_counter() - started,
        "nonprivate_diagnostic_fields": ("fit_seconds",),
    }
    if second is not None:
        result["risk"] = analytic_risk(p, m, second, result["q"], result["decoder"])
    return result
