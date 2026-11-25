"""Finite-alphabet private task learning with ordered calibration.

This legacy API preserves the original uncapped search. The legacy module
retains the uncapped search used for the recorded simulations and timings.
The ideal mechanism protects complete fitting records under pure central
privacy and each randomized fresh query under all-pairs local privacy.
Encoder, alphabet, sample size, and fixed index split are public.

This float64 research implementation is not a certified machine-arithmetic
DP release. Public fixed seeds are for synthetic reproducibility only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .calibration import fit_ordered_private_dp


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite positive integer") from error
    if integer < 1 or integer != value:
        raise ValueError(f"{name} must be a finite positive integer")
    return integer


def _states(values, cells, allow_empty=False):
    values = np.asarray(values)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or (len(values) == 0 and not allow_empty)):
        raise ValueError("cell indices must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values >= cells):
        raise ValueError("cell indices must belong to the declared alphabet")
    return values.astype(np.int64)


def _labels(values, observations):
    values = np.asarray(values, dtype=float)
    if values.shape != (observations,) or observations == 0:
        raise ValueError("each nonempty sample requires one label per observation")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("labels must be finite and belong to [0, 1]")
    return values


def training_parameters(alpha_train):
    """Return kappa=tanh(alpha/2) and the sufficient pure budget min(1,alpha)."""
    if (not np.isscalar(alpha_train) or not np.isfinite(alpha_train)
            or alpha_train < 0):
        raise ValueError("alpha_train must be finite and nonnegative")
    alpha_train = float(alpha_train)
    return math.tanh(alpha_train / 2), min(1.0, alpha_train)


def _query_privacy(epsilon_query):
    if (not np.isscalar(epsilon_query) or not np.isfinite(epsilon_query)
            or not 0 <= epsilon_query <= np.log(4)):
        raise ValueError("epsilon_query must belong to [0, log(4)]")
    return float(epsilon_query)


def fractional_histogram(states, labels, cells, normalize=False):
    """INTERNAL nonprivate statistic; do not release this raw histogram."""
    cells = _positive_integer(cells, "cells")
    states = _states(states, cells)
    labels = _labels(labels, len(states))
    result = np.column_stack([
        np.bincount(states, weights=1 - labels, minlength=cells),
        np.bincount(states, weights=labels, minlength=cells)])
    return result / len(states) if normalize else result


def rescale_nonnegative_block(noisy_values, target_mass):
    """Postprocess a noisy block to fixed mass, using public cell zero if empty."""
    values = np.asarray(noisy_values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("noisy block must be a nonempty finite vector")
    if not np.isfinite(target_mass) or not 0 <= target_mass <= 1:
        raise ValueError("target_mass must belong to [0, 1]")
    clipped = np.maximum(values, 0)
    maximum = float(clipped.max())
    if maximum > 0:
        # First scaling avoids overflow when summing large finite coordinates.
        scaled = clipped / maximum
        return target_mass * (scaled / scaled.sum())
    result = np.zeros_like(clipped)
    result[0] = target_mass
    return result


@dataclass(frozen=True)
class FinitePrivateTaskModel:
    """Released model: public settings and sanitized quantities only.

    No raw labels, counts, observed support, empirical risk, or RNG is retained.
    Array fields are private releases or their postprocessing; queries below
    must still be randomized when their input state is confidential.
    """

    cells: int
    sample_size: int
    epsilon_query: float
    alpha_train: float
    kappa: float
    pure_training_budget: float
    pilot_scores: np.ndarray
    thresholds: np.ndarray
    decoder: np.ndarray
    sanitized_p: np.ndarray
    sanitized_s: np.ndarray
    sanitized_mean: float
    branch: str

    def bins(self, states):
        """Deterministic internal/evaluation assignments; NOT a private query release."""
        states = _states(states, self.cells, allow_empty=True)
        return np.searchsorted(self.thresholds, self.pilot_scores[states], side="right")

    def probabilities(self, states):
        """Evaluation of query rows; NOT a private release of the supplied states."""
        bins = self.bins(states)
        c = math.expm1(self.epsilon_query)
        return ((1 + c * (bins[:, None] == np.arange(len(self.decoder))))
                / (len(self.decoder) + c))

    def expected_prediction(self, states):
        """Analytic expectation for evaluation; NOT a private query release."""
        return self.probabilities(states) @ self.decoder

    def sample_representations(self, states, rng=None):
        """Sample the private RR output index once for each supplied query state.

        Each individual fresh query has the stated epsilon_query guarantee.
        Repeated releases for the same confidential input require composition.
        """
        rng = np.random.default_rng() if rng is None else rng
        bins = self.bins(states)
        c = math.expm1(self.epsilon_query)
        k = len(self.decoder)
        indices = rng.integers(k, size=len(bins))
        retain = rng.random(len(bins)) < c / (k + c)
        indices[retain] = bins[retain]
        return indices

    def decode(self, private_indices):
        """Postprocess already-private indices through the fitted decoder."""
        private_indices = np.asarray(private_indices)
        if (not np.issubdtype(private_indices.dtype, np.integer)
                or np.any(private_indices < 0)
                or np.any(private_indices >= len(self.decoder))):
            raise ValueError("private indices must belong to the output alphabet")
        return self.decoder[private_indices]

    def predict(self, states, rng=None):
        """Sample and decode a private query output for each supplied state."""
        return self.decode(self.sample_representations(states, rng=rng))


def fit_finite_private_task_learner(states, labels, *, cells, epsilon_query,
                                    alpha_train, rng=None):
    """Fit the two-block learner for a public finite alphabet.

    Parameters
    ----------
    states : integer array, shape (n,)
        Encoded input states in the public alphabet {0,...,cells-1}.
    labels : array, shape (n,)
        Real target values in [0,1].
    cells : positive integer
        Full public alphabet size, including states absent from the sample.
    epsilon_query : float in [0,log(4)]
        Pure all-pairs privacy of each individual randomized fresh query.
    alpha_train : finite nonnegative float
        Pure complete-record replacement privacy budget for fitting.
    rng : NumPy Generator or compatible object, optional
        Fresh private randomness by default. Published fixed seeds are for
        reproducible tests, not for a private deployed fit.

    Notes
    -----
    The fixed index split uses floor(n/2) pilot and ceil(n/2) calibration
    records. The internal pure budget is min(1,alpha_train). Both disjoint
    blocks receive this budget; calibration divides its own budget equally
    between its fractional histogram and separately released mean.
    """
    cells = _positive_integer(cells, "cells")
    states = _states(states, cells)
    labels = _labels(labels, len(states))
    epsilon_query = _query_privacy(epsilon_query)
    kappa, alpha0 = training_parameters(alpha_train)
    rng = np.random.default_rng() if rng is None else rng
    n = len(states)
    if n == 1 or n * kappa < 1:
        uniform = np.full(cells, 1 / cells)
        return FinitePrivateTaskModel(
            cells, n, epsilon_query, float(alpha_train), kappa, alpha0,
            np.full(cells, .5), np.array([]), np.array([.5]),
            uniform, .5 * uniform, .5, "data_independent_constant")

    pilot_size = n // 2
    calibration_size = n - pilot_size
    pilot_histogram = fractional_histogram(states[:pilot_size], labels[:pilot_size], cells)
    noisy_pilot = np.maximum(pilot_histogram + rng.laplace(
        0., 2 / alpha0, size=(cells, 2)), 0)
    pilot_scores = (noisy_pilot[:, 1] + .5) / (noisy_pilot.sum(axis=1) + 1)

    histogram = fractional_histogram(states[pilot_size:], labels[pilot_size:],
                                     cells, normalize=True)
    noisy_histogram = histogram + rng.laplace(
        0., 4 / (calibration_size * alpha0), size=(cells, 2))
    sanitized_mean = float(np.clip(
        labels[pilot_size:].mean() + rng.laplace(
            0., 2 / (calibration_size * alpha0)), 0, 1))
    sanitized_joint = np.column_stack([
        rescale_nonnegative_block(noisy_histogram[:, 0], 1 - sanitized_mean),
        rescale_nonnegative_block(noisy_histogram[:, 1], sanitized_mean)])
    sanitized_p = sanitized_joint.sum(axis=1)
    sanitized_s = sanitized_joint[:, 1]
    fitted = fit_ordered_private_dp(sanitized_p, sanitized_s, pilot_scores, epsilon_query)
    return FinitePrivateTaskModel(
        cells, n, epsilon_query, float(alpha_train), kappa, alpha0,
        pilot_scores, fitted["thresholds"], fitted["decoder"],
        sanitized_p, sanitized_s, sanitized_mean, "private_pilot_and_calibration")
