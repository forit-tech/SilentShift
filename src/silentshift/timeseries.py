"""Autocorrelation handling.

Every two-sample test used in this project assumes independent draws. Server telemetry
violates that badly: consecutive minutes are near-copies of each other. Two consequences,
both measured rather than assumed:

1. **Nominal significance levels are meaningless.** A KS test on 500 autocorrelated rows
   behaves like a test on far fewer independent ones, so its p-value is wildly optimistic.
   This is why thresholds are calibrated empirically.

2. **Distance statistics saturate.** With enough dependent samples, *any* window is
   distinguishable from *any* reference, because the test is picking up the difference
   between two particular trajectories rather than between two distributions. A classifier
   two-sample test hits AUC 1.0 on drift-free data and stops carrying information at all.

Thinning attacks the cause instead of the symptom: keep every tau-th row, where tau is the
integrated autocorrelation time estimated on the reference. The sample gets smaller but its
members are closer to independent, and the statistic recovers a usable dynamic range.
"""

from __future__ import annotations

import numpy as np


def autocorrelation_time(x: np.ndarray, max_lag: int = 600, threshold: float = 0.3) -> int:
    """First lag at which mean absolute autocorrelation drops below `threshold`.

    A blunt estimator, deliberately. The integrated autocorrelation time of a
    non-stationary series is not well defined, and a more elaborate estimate would imply a
    precision the data does not support. What matters is the order of magnitude: whether
    consecutive rows are effectively one sample or fifty.

    The 0.3 threshold is a measured compromise, not a convention. On SMD the mean absolute
    autocorrelation is still 0.29-0.57 at lag 100 and does not decay monotonically -- it dips
    near lag 300 and rises again near lag 600, which is a daily cycle rather than noise.
    Thinning at 0.1 would demand steps of 231-1124 rows and leave a handful of samples per
    window; at 0.3 the step is roughly 70-190 rows, which keeps ~20 effective samples in a
    2500-row window. Both numbers are in the README so the reader can see the trade rather than
    take the constant on faith.
    """
    if x.ndim != 2:
        raise ValueError("expected a 2-D array of shape (rows, features)")
    n = x.shape[0]
    max_lag = int(min(max_lag, n // 4))
    if max_lag < 1:
        return 1

    centred = x - x.mean(axis=0, keepdims=True)
    var = np.mean(centred**2, axis=0)
    active = var > 1e-12
    if not np.any(active):
        return 1
    centred = centred[:, active]
    var = var[active]

    for lag in range(1, max_lag + 1):
        cov = np.mean(centred[lag:] * centred[:-lag], axis=0)
        rho = np.mean(np.abs(cov / var))
        if rho < threshold:
            return lag
    return max_lag


def thin(x: np.ndarray, step: int) -> np.ndarray:
    """Keep every `step`-th row.

    Offset zero always, so thinning is deterministic and a window compared against a
    reference is not accidentally sampled on a different phase of a daily cycle.
    """
    step = max(1, int(step))
    return x[::step] if step > 1 else x


def effective_sample_size(n: int, tau: int) -> int:
    return max(1, int(n // max(1, tau)))
