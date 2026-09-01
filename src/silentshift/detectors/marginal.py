"""Per-feature (marginal) drift statistics.

Every detector here reduces the problem to D independent one-dimensional comparisons and
aggregates with a max. That is what most production drift monitoring actually does, which
is why these are the baselines to beat rather than straw men.

It also means every detector in this module is, by construction, blind to a change that
leaves all marginals intact. `injection.catalogue.correlation_break` produces exactly such
a change, so a recall near zero there is the expected and correct outcome.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .base import Detector, Thinning

_EPS = 1e-10


class KSDetector(Detector):
    """Max over features of the two-sample Kolmogorov-Smirnov statistic.

    The statistic, not the p-value. KS p-values assume i.i.d. samples; server telemetry is
    strongly autocorrelated, so the nominal level is meaningless here and thresholding on
    it would produce a false-alarm rate that has no relation to the one requested. The
    threshold is calibrated empirically instead.
    """

    def __init__(self, thinning: Thinning = "none") -> None:
        super().__init__(thinning)
        self.name = "ks_max"
        self._ref: np.ndarray | None = None

    def fit_reference(self, reference: np.ndarray) -> None:
        self._calibrate_thinning(reference)
        self._ref = np.ascontiguousarray(self._thin(reference))

    def _per_feature(self, window: np.ndarray) -> np.ndarray:
        assert self._ref is not None, "fit_reference must be called first"
        window = self._thin(window)
        d = window.shape[1]
        out = np.empty(d, dtype=np.float64)
        for j in range(d):
            out[j] = stats.ks_2samp(
                self._ref[:, j], window[:, j], method="asymp"
            ).statistic
        return out

    def score(self, window: np.ndarray) -> float:
        return float(np.max(self._per_feature(window)))

    def attribute(self, window: np.ndarray) -> np.ndarray:
        return self._per_feature(window)


class KSBonferroniDetector(KSDetector):
    """KS p-values with a Bonferroni correction, kept to quantify the multiple-testing effect.

    Thirty-eight simultaneous tests per window inflate the alarm rate badly if the
    correction is skipped. This detector exists so the README can show the size of that
    effect with a number instead of asserting it.
    """

    def __init__(self, thinning: Thinning = "none") -> None:
        super().__init__(thinning)
        self.name = "ks_bonferroni"

    def score(self, window: np.ndarray) -> float:
        assert self._ref is not None, "fit_reference must be called first"
        window = self._thin(window)
        d = window.shape[1]
        pvals = np.empty(d, dtype=np.float64)
        for j in range(d):
            pvals[j] = stats.ks_2samp(self._ref[:, j], window[:, j], method="asymp").pvalue
        adjusted = np.clip(pvals * d, _EPS, 1.0)
        return float(-np.log10(np.min(adjusted)))


class PSIDetector(Detector):
    """Max Population Stability Index over features.

    Bin edges come from reference quantiles and are frozen with the reference: recomputing
    them per window would compare each sample against its own bins and drive PSI toward
    zero exactly when the distribution moves.
    """

    def __init__(self, n_bins: int = 10, thinning: Thinning = "none") -> None:
        super().__init__(thinning)
        self.name = "psi_max"
        self.n_bins = n_bins
        self._edges: list[np.ndarray] = []
        self._ref_frac: np.ndarray | None = None

    def fit_reference(self, reference: np.ndarray) -> None:
        self._calibrate_thinning(reference)
        reference = self._thin(reference)
        d = reference.shape[1]
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)
        self._edges = []
        fracs = np.empty((d, self.n_bins), dtype=np.float64)
        for j in range(d):
            edges = np.quantile(reference[:, j], qs)
            edges = np.unique(edges)
            if edges.size < 2:  # constant feature
                edges = np.array([reference[0, j] - 0.5, reference[0, j] + 0.5])
            edges[0], edges[-1] = -np.inf, np.inf
            self._edges.append(edges)
            counts, _ = np.histogram(reference[:, j], bins=edges)
            frac = counts / max(1, counts.sum())
            padded = np.full(self.n_bins, _EPS)
            padded[: frac.size] = np.maximum(frac, _EPS)
            fracs[j] = padded
        self._ref_frac = fracs

    def _per_feature(self, window: np.ndarray) -> np.ndarray:
        assert self._ref_frac is not None, "fit_reference must be called first"
        window = self._thin(window)
        d = window.shape[1]
        out = np.zeros(d, dtype=np.float64)
        for j in range(d):
            counts, _ = np.histogram(window[:, j], bins=self._edges[j])
            frac = counts / max(1, counts.sum())
            cur = np.full(self.n_bins, _EPS)
            cur[: frac.size] = np.maximum(frac, _EPS)
            ref = self._ref_frac[j]
            out[j] = float(np.sum((cur - ref) * np.log(cur / ref)))
        return out

    def score(self, window: np.ndarray) -> float:
        return float(np.max(self._per_feature(window)))

    def attribute(self, window: np.ndarray) -> np.ndarray:
        return self._per_feature(window)


class WassersteinDetector(Detector):
    """Max over features of the 1-D Wasserstein distance, scaled by reference sigma.

    Scaling matters: without it the max is dominated by whichever feature happens to have
    the widest range, and the detector silently becomes a single-feature monitor.
    """

    def __init__(self, thinning: Thinning = "none") -> None:
        super().__init__(thinning)
        self.name = "wasserstein_max"
        self._ref: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    def fit_reference(self, reference: np.ndarray) -> None:
        self._calibrate_thinning(reference)
        reference = self._thin(reference)
        self._ref = np.ascontiguousarray(reference)
        s = np.std(reference, axis=0)
        self._sigma = np.where(s > 1e-12, s, 1.0)

    def _per_feature(self, window: np.ndarray) -> np.ndarray:
        assert self._ref is not None and self._sigma is not None
        window = self._thin(window)
        d = window.shape[1]
        out = np.empty(d, dtype=np.float64)
        for j in range(d):
            out[j] = stats.wasserstein_distance(self._ref[:, j], window[:, j]) / self._sigma[j]
        return out

    def score(self, window: np.ndarray) -> float:
        return float(np.max(self._per_feature(window)))

    def attribute(self, window: np.ndarray) -> np.ndarray:
        return self._per_feature(window)


class JensenShannonDetector(Detector):
    """Max Jensen-Shannon divergence over features, on reference-defined bins."""

    def __init__(self, n_bins: int = 20, thinning: Thinning = "none") -> None:
        super().__init__(thinning)
        self.name = "js_max"
        self.n_bins = n_bins
        self._edges: list[np.ndarray] = []
        self._ref_p: np.ndarray | None = None

    def fit_reference(self, reference: np.ndarray) -> None:
        self._calibrate_thinning(reference)
        reference = self._thin(reference)
        d = reference.shape[1]
        self._edges = []
        ps = np.empty((d, self.n_bins), dtype=np.float64)
        for j in range(d):
            lo, hi = float(np.min(reference[:, j])), float(np.max(reference[:, j]))
            if hi - lo < 1e-12:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, self.n_bins + 1)
            edges[0], edges[-1] = -np.inf, np.inf
            self._edges.append(edges)
            counts, _ = np.histogram(reference[:, j], bins=edges)
            ps[j] = np.maximum(counts / max(1, counts.sum()), _EPS)
        self._ref_p = ps

    def _per_feature(self, window: np.ndarray) -> np.ndarray:
        assert self._ref_p is not None
        window = self._thin(window)
        d = window.shape[1]
        out = np.empty(d, dtype=np.float64)
        for j in range(d):
            counts, _ = np.histogram(window[:, j], bins=self._edges[j])
            q = np.maximum(counts / max(1, counts.sum()), _EPS)
            p = self._ref_p[j]
            m = 0.5 * (p + q)
            kl_pm = float(np.sum(p * np.log2(p / m)))
            kl_qm = float(np.sum(q * np.log2(q / m)))
            out[j] = 0.5 * (kl_pm + kl_qm)
        return out

    def score(self, window: np.ndarray) -> float:
        return float(np.max(self._per_feature(window)))

    def attribute(self, window: np.ndarray) -> np.ndarray:
        return self._per_feature(window)
