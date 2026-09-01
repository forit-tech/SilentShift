"""Detectors that look at the joint distribution.

These exist because the marginal family in `marginal.py` cannot, even in principle, see a
change that rearranges the dependence structure while leaving every one-dimensional
distribution untouched. If the joint detectors do not beat the marginal ones on
`correlation_break`, the multivariate machinery is not earning its cost and the README
should say so.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .base import Detector, Thinning, safe_std


class ClassifierTwoSampleDetector(Detector):
    """Classifier two-sample test.

    Train a classifier to tell reference rows from window rows. If the two samples come
    from the same distribution the best achievable held-out AUC is 0.5, so
    `max(0, 2*AUC - 1)` is a distance-like statistic bounded in [0, 1].

    Logistic regression rather than a boosted tree: with a few hundred rows per window and
    38 features, a flexible learner overfits the split and the AUC stops meaning anything.
    A linear decision boundary on standardised features detects location and, through the
    interaction of correlated inputs, a good deal of dependence change too.
    """

    def __init__(
        self, seed: int = 0, max_reference: int = 2000, thinning: Thinning = "none"
    ) -> None:
        super().__init__(thinning)
        self.name = "c2st"
        self.seed = seed
        self.max_reference = max_reference
        self._ref: np.ndarray | None = None
        self._last_coef: np.ndarray | None = None

    def fit_reference(self, reference: np.ndarray) -> None:
        self._calibrate_thinning(reference)
        reference = self._thin(reference)
        rng = np.random.default_rng(self.seed)
        if reference.shape[0] > self.max_reference:
            idx = rng.choice(reference.shape[0], self.max_reference, replace=False)
            reference = reference[np.sort(idx)]
        self._ref = np.ascontiguousarray(reference)

    def _prepare(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._ref is not None, "fit_reference must be called first"
        window = self._thin(window)
        rng = np.random.default_rng(self.seed)
        n = min(self._ref.shape[0], window.shape[0])
        # Balanced classes: an unbalanced two-sample test inflates AUC through the prior
        # rather than through any distributional difference.
        ref_idx = rng.choice(self._ref.shape[0], n, replace=False)
        win_idx = rng.choice(window.shape[0], n, replace=False)
        x = np.vstack([self._ref[np.sort(ref_idx)], window[np.sort(win_idx)]])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        return x, y

    def _fit_split(self, window: np.ndarray) -> tuple[float, np.ndarray]:
        x, y = self._prepare(window)
        rng = np.random.default_rng(self.seed + 1)
        order = rng.permutation(x.shape[0])
        x, y = x[order], y[order]
        cut = x.shape[0] // 2
        x_tr, x_te, y_tr, y_te = x[:cut], x[cut:], y[:cut], y[cut:]
        if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
            return 0.0, np.zeros(x.shape[1])

        mu, sd = x_tr.mean(axis=0), safe_std(x_tr)
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit((x_tr - mu) / sd, y_tr)
        proba = model.predict_proba((x_te - mu) / sd)[:, 1]
        auc = float(roc_auc_score(y_te, proba))
        return auc, np.abs(model.coef_.ravel())

    def score(self, window: np.ndarray) -> float:
        auc, coef = self._fit_split(window)
        self._last_coef = coef
        return float(max(0.0, 2.0 * auc - 1.0))

    def attribute(self, window: np.ndarray) -> np.ndarray:
        if self._last_coef is None:
            self.score(window)
        assert self._last_coef is not None
        return self._last_coef


class PCAReconstructionDetector(Detector):
    """Reconstruction error against a basis fitted on the reference window.

    A subspace learned from reference behaviour stops explaining the data when the
    dependence structure moves, even if no individual feature shifts. That makes this the
    cheapest joint-distribution detector available, and the natural comparison point for
    whether anything heavier is justified.
    """

    def __init__(
        self, n_components: float = 0.95, seed: int = 0, thinning: Thinning = "none"
    ) -> None:
        super().__init__(thinning)
        self.name = "pca_recon"
        self.n_components = n_components
        self.seed = seed
        self._pca: PCA | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._ref_error: float = 1.0

    def fit_reference(self, reference: np.ndarray) -> None:
        self._mu = reference.mean(axis=0)
        self._sd = safe_std(reference)
        z = (reference - self._mu) / self._sd
        max_comp = min(z.shape[0], z.shape[1])
        pca = PCA(n_components=self.n_components, svd_solver="full", random_state=self.seed)
        pca.fit(z)
        if pca.n_components_ >= max_comp:  # degenerate: nothing is discarded, error is ~0
            pca = PCA(n_components=max(1, max_comp - 1), svd_solver="full", random_state=self.seed)
            pca.fit(z)
        self._pca = pca
        # Normalising by the reference error makes the statistic comparable across machines,
        # which otherwise differ by orders of magnitude in residual scale.
        self._ref_error = max(float(np.mean(self._residuals(reference))), 1e-12)

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        assert self._pca is not None and self._mu is not None and self._sd is not None
        z = (x - self._mu) / self._sd
        recon = self._pca.inverse_transform(self._pca.transform(z))
        return np.mean((z - recon) ** 2, axis=1)

    def score(self, window: np.ndarray) -> float:
        return float(np.mean(self._residuals(window)) / self._ref_error)

    def attribute(self, window: np.ndarray) -> np.ndarray:
        assert self._pca is not None and self._mu is not None and self._sd is not None
        z = (window - self._mu) / self._sd
        recon = self._pca.inverse_transform(self._pca.transform(z))
        per_feature = np.mean((z - recon) ** 2, axis=0)
        return per_feature


class IsolationForestDetector(Detector):
    """Mean isolation-forest anomaly score over the window.

    Included as the thing people reach for. It scores points, not distributions, so
    aggregating it over a window is a category error dressed as a drift detector — and
    measuring how badly that performs is more useful than asserting it.
    """

    def __init__(self, n_estimators: int = 100, seed: int = 0, thinning: Thinning = "none") -> None:
        super().__init__(thinning)
        self.name = "iforest_mean"
        self.n_estimators = n_estimators
        self.seed = seed
        self._model: IsolationForest | None = None
        self._ref_mean: float = 0.0

    def fit_reference(self, reference: np.ndarray) -> None:
        model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, contamination="auto"
        )
        model.fit(reference)
        self._model = model
        self._ref_mean = float(np.mean(-model.score_samples(reference)))

    def score(self, window: np.ndarray) -> float:
        assert self._model is not None, "fit_reference must be called first"
        return float(np.mean(-self._model.score_samples(window)) - self._ref_mean)


class RandomAlarmDetector(Detector):
    """Uniform noise, independent of the data.

    The degenerate control. Any detector whose detection curve is not clearly above this
    one at matched false-alarm rate has not detected anything, and this makes that
    comparison explicit rather than assumed.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__("none")
        self.name = "random"
        self._rng = np.random.default_rng(seed)

    def fit_reference(self, reference: np.ndarray) -> None:
        return None

    def score(self, window: np.ndarray) -> float:
        return float(self._rng.random())
