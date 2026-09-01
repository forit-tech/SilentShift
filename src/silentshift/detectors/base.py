"""Detector interface.

A detector sees a reference sample and a candidate window and returns a single scalar that
grows with evidence of change. It does *not* decide whether an alarm fires: thresholds are
calibrated separately, on drift-free data, because the nominal significance levels of the
two-sample tests below are wrong on autocorrelated telemetry.

`attribute` returns a per-feature contribution vector for the same window. It is scored
against injected ground truth, not merely plotted.

Every detector can optionally thin its inputs — see `silentshift.timeseries` for why that
matters more than it sounds like it should.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..timeseries import autocorrelation_time, thin

Thinning = str | int


class Detector(ABC):
    """Common contract for every drift detector in the study."""

    name: str

    def __init__(self, thinning: Thinning = "none") -> None:
        self.thinning: Thinning = thinning
        self._tau = 1

    # -- thinning -------------------------------------------------------------

    def _calibrate_thinning(self, reference: np.ndarray) -> None:
        """Fix the thinning step from the reference, once per reference.

        Estimating tau from the *window* instead would let the drift itself change the
        sampling rate, which would confound the statistic with the thing it is measuring.
        """
        if self.thinning == "auto":
            self._tau = autocorrelation_time(reference)
        elif isinstance(self.thinning, int):
            self._tau = max(1, self.thinning)
        else:
            self._tau = 1

    def _thin(self, x: np.ndarray) -> np.ndarray:
        return thin(x, self._tau)

    @property
    def tau(self) -> int:
        return self._tau

    # -- interface ------------------------------------------------------------

    @abstractmethod
    def fit_reference(self, reference: np.ndarray) -> None:
        """Absorb the reference sample. Called again whenever the window policy moves it."""

    @abstractmethod
    def score(self, window: np.ndarray) -> float:
        """Higher means more evidence that `window` came from a different distribution."""

    def attribute(self, window: np.ndarray) -> np.ndarray:
        """Per-feature contribution to the current score.

        The default is uninformative on purpose: a detector that cannot attribute should say
        so rather than return a plausible-looking vector.
        """
        return np.zeros(window.shape[1], dtype=np.float64)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, thinning={self.thinning!r})"


def safe_std(x: np.ndarray, axis: int = 0, floor: float = 1e-12) -> np.ndarray:
    """Standard deviation with a floor, so constant columns cannot produce inf or nan."""
    s = np.std(x, axis=axis)
    return np.where(s > floor, s, 1.0)
