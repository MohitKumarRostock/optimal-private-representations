"""Ordered randomized-response calibration on sanitized moments.

The default search examines every active output count, as in the recorded runs.
Explicit max_outputs selects the corresponding comparison family. Calling
this optimizer on raw empirical moments does not itself make fitting private.
The exactness claim concerns real arithmetic; this implementation uses float64.
"""
from __future__ import annotations

import numpy as np

def randomized_response(bins, epsilon):
    bins = np.asarray(bins, dtype=int)
    k = int(bins.max()) + 1
    c = np.expm1(float(epsilon))
    return (1 + c * (bins[:, None] == np.arange(k))) / (k + c)


def fit_ordered_private_dp(p, s, scores, epsilon, max_outputs=None):
    """Fit interval RR bins with nondecreasing empirical Bayes decoders.

    p[i] is a cell probability, s[i] its first target moment, and scores[i]
    a fixed score for this calibration problem. Equal scores are aggregated. Every input
    cell, including a cell absent from calibration, receives a private row.
    The DP uses O(d^3 log(d+1)) arithmetic and O(d^3) storage, where d is
    the number of distinct scores with positive supplied mass. This array API
    additionally groups K cell scores in O(K log K) and materializes K*k
    channel entries; deployment can instead retain thresholds and decoder.
    If max_outputs is supplied, optimize over at most that many outputs;
    the corresponding time and storage become O(B*d^2*log(d+1)) and
    O(B*d^2), where B=min(max_outputs,d). The default is unrestricted.
    """
    p, s, scores = (np.asarray(x, dtype=float) for x in (p, s, scores))
    if p.ndim != 1 or s.shape != p.shape or scores.shape != p.shape or not len(p):
        raise ValueError("p, s, and scores must be nonempty equal-length vectors")
    if not np.isfinite(epsilon) or epsilon < 0 or not np.isclose(p.sum(), 1.0, atol=1e-12):
        raise ValueError("epsilon must be nonnegative and p must sum to one")
    if not all(np.all(np.isfinite(x)) for x in (p, s, scores)) or np.any(p < 0) or np.any(s < 0) or np.any(s > p):
        raise ValueError("require finite scores and 0 <= s <= p")
    distinct, inverse = np.unique(scores, return_inverse=True)
    grouped_p = np.bincount(inverse, weights=p)
    grouped_s = np.bincount(inverse, weights=s)
    active = grouped_p > 0
    support = distinct[active]
    pp = np.r_[0.0, np.cumsum(grouped_p[active])]
    ss = np.r_[0.0, np.cumsum(grouped_s[active])]
    d = len(support)
    if max_outputs is not None and (
            isinstance(max_outputs, (bool, np.bool_))
            or not isinstance(max_outputs, (int, np.integer))
            or max_outputs < 1):
        raise ValueError("max_outputs must be a positive integer or None")
    limit = d if max_outputs is None else min(d, int(max_outputs))
    mu = float(s.sum())
    if epsilon == 0 or d == 1:
        return {"q": np.ones((len(p), 1)), "decoder": np.array([mu]),
                "bins": np.zeros(len(p), dtype=int), "k": 1,
                "thresholds": np.array([]), "scores": scores,
                "explained_second_moment": mu * mu}
    c = float(np.expm1(epsilon))
    interval_mean = np.full((d + 1, d + 1), np.nan)
    gain = np.full_like(interval_mean, -np.inf)
    for left in range(d):
        right = np.arange(left + 1, d + 1)
        denominator = 1 + c * (pp[right] - pp[left])
        numerator = mu + c * (ss[right] - ss[left])
        interval_mean[left, right] = numerator / denominator
        gain[left, right] = numerator ** 2 / denominator

    # State (k,s,t): k intervals covering [0,t), last interval [s,t).
    value = np.full((limit + 1, d + 1, d + 1), -np.inf)
    previous = np.full(value.shape, -1, dtype=np.int32)
    value[1, 0, 1:] = gain[0, 1:]
    for k in range(2, limit + 1):
        for left in range(k - 1, d):
            starts = np.arange(k - 2, left)
            valid = np.isfinite(value[k - 1, starts, left])
            starts = starts[valid]
            if not len(starts):
                continue
            order = np.argsort(interval_mean[starts, left], kind="stable")
            starts = starts[order]
            means = interval_mean[starts, left]
            candidate_values = value[k - 1, starts, left]
            prefix_choice = np.empty(len(starts), dtype=int)
            chosen = 0
            for j in range(len(starts)):
                if candidate_values[j] > candidate_values[chosen]:
                    chosen = j
                prefix_choice[j] = chosen
            for right in range(left + 1, d + 1):
                stop = np.searchsorted(means, interval_mean[left, right],
                                       side="right") - 1
                if stop >= 0:
                    chosen = prefix_choice[stop]
                    value[k, left, right] = candidate_values[chosen] + gain[left, right]
                    previous[k, left, right] = starts[chosen]
    last_starts = np.argmax(value[1:, :, d], axis=1)
    objectives = np.array([value[k, last_starts[k - 1], d] / (k + c)
                           for k in range(1, limit + 1)])
    output_count = int(np.argmax(objectives)) + 1
    intervals = []
    k, right = output_count, d
    left = int(last_starts[k - 1])
    while k:
        intervals.append((left, right))
        earlier = int(previous[k, left, right])
        right, left = left, earlier
        k -= 1
    intervals.reverse()
    thresholds = np.array([support[right] for _, right in intervals[:-1]])
    bins = np.searchsorted(thresholds, scores, side="right")
    q = randomized_response(bins, epsilon)
    decoder = (s @ q) / (p @ q)
    second_moment = float(np.sum((p @ q) * decoder ** 2))
    assert np.min(np.diff(decoder), initial=0.0) >= -1e-12
    assert abs(second_moment - objectives[output_count - 1]) < 1e-10
    return {"q": q, "decoder": decoder, "bins": bins, "k": output_count,
            "thresholds": thresholds, "scores": scores,
            "explained_second_moment": second_moment}
