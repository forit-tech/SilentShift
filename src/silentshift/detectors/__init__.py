"""Detector registry.

Construction is centralised so a config file can name detectors as strings and a run is
reproducible from that file alone.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Detector
from .marginal import (
    JensenShannonDetector,
    KSBonferroniDetector,
    KSDetector,
    PSIDetector,
    WassersteinDetector,
)
from .multivariate import (
    ClassifierTwoSampleDetector,
    IsolationForestDetector,
    PCAReconstructionDetector,
    RandomAlarmDetector,
)

_REGISTRY: dict[str, Callable[[int], Detector]] = {
    "ks_max": lambda seed: KSDetector(),
    "ks_bonferroni": lambda seed: KSBonferroniDetector(),
    "psi_max": lambda seed: PSIDetector(),
    "wasserstein_max": lambda seed: WassersteinDetector(),
    "js_max": lambda seed: JensenShannonDetector(),
    "c2st": lambda seed: ClassifierTwoSampleDetector(seed=seed),
    # "_thinned" variants subsample by the estimated autocorrelation time before testing.
    # See silentshift.timeseries: without it these statistics saturate on this data.
    "ks_max_thinned": lambda seed: KSDetector(thinning="auto"),
    "psi_max_thinned": lambda seed: PSIDetector(thinning="auto"),
    "wasserstein_max_thinned": lambda seed: WassersteinDetector(thinning="auto"),
    "js_max_thinned": lambda seed: JensenShannonDetector(thinning="auto"),
    "c2st_thinned": lambda seed: ClassifierTwoSampleDetector(seed=seed, thinning="auto"),
    "pca_recon": lambda seed: PCAReconstructionDetector(seed=seed),
    "iforest_mean": lambda seed: IsolationForestDetector(seed=seed),
    "random": lambda seed: RandomAlarmDetector(seed=seed),
}

MARGINAL_DETECTORS = frozenset(
    {
        "ks_max", "ks_bonferroni", "psi_max", "wasserstein_max", "js_max",
        "ks_max_thinned", "psi_max_thinned", "wasserstein_max_thinned", "js_max_thinned",
    }
)


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build(name: str, seed: int = 0) -> Detector:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown detector {name!r}; available: {available()}") from None
    return factory(seed)


__all__ = ["MARGINAL_DETECTORS", "Detector", "available", "build"]
