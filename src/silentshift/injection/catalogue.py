"""Controlled drift injection.

This module is the ground truth generator, and it is written deliberately *without* any
knowledge of how drift is detected. It describes what changed in the data-generating
process; `silentshift.detectors` decides how to notice. Keeping the two apart is the only
thing that stops the evaluation from being circular.

Magnitudes are expressed in units of the per-feature standard deviation measured on the
pre-onset portion of the same stream. SMD ships globally min-max scaled values, so raw
units carry no meaning and a scale-relative magnitude is the only comparable one.

The catalogue includes changes that specific detector families cannot see by construction.
`correlation_break` permutes post-onset values within a feature, so every marginal is
bit-for-bit identical before and after; any statistic computed per feature on marginals —
KS, PSI, per-feature Wasserstein — is provably blind to it. That is the point, not an
oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ScenarioName = Literal[
    "none",
    "sudden_shift",
    "gradual_shift",
    "incremental_shift",
    "variance_shift",
    "correlation_break",
]

SCENARIOS: tuple[ScenarioName, ...] = (
    "none",
    "sudden_shift",
    "gradual_shift",
    "incremental_shift",
    "variance_shift",
    "correlation_break",
)


@dataclass(frozen=True)
class DriftSpec:
    """Ground truth for one injected stream."""

    scenario: ScenarioName
    onset: int  # row index where the change begins; -1 for the drift-free scenario
    affected: tuple[int, ...]  # zero-indexed feature columns
    magnitude: float  # in units of pre-onset per-feature sigma
    params: dict[str, float] = field(default_factory=dict)

    @property
    def is_drift_free(self) -> bool:
        return self.scenario == "none"

    def to_record(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "onset": self.onset,
            "affected": list(self.affected),
            "magnitude": self.magnitude,
            **{f"param_{k}": v for k, v in self.params.items()},
        }


def choose_features(
    n_features: int,
    n_affected: int,
    excluded: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Pick which columns drift, skipping degenerate ones.

    `excluded` masks constant features: a shift of k sigma is undefined when sigma is zero,
    and silently injecting nothing there would inflate every detector's miss rate for a
    reason that has nothing to do with detection.
    """
    eligible = np.flatnonzero(~excluded)
    if eligible.size == 0:
        raise ValueError("no non-constant features available to inject into")
    k = min(n_affected, eligible.size)
    picked = rng.choice(eligible, size=k, replace=False)
    return tuple(sorted(int(i) for i in picked))


def _pre_onset_sigma(x: np.ndarray, onset: int, affected: tuple[int, ...]) -> np.ndarray:
    """Per-feature sigma measured only on data the detector would also have seen."""
    sigma = np.std(x[:onset, affected], axis=0)
    # A feature can be constant on the pre-onset slice while varying later. Falling back to
    # the full-series sigma keeps the injection meaningful; falling back to 0 would not.
    fallback = np.std(x[:, affected], axis=0)
    sigma = np.where(sigma > 1e-12, sigma, fallback)
    return np.where(sigma > 1e-12, sigma, 1.0)


def inject(
    x: np.ndarray,
    spec: DriftSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a copy of `x` with the drift described by `spec` applied."""
    if spec.is_drift_free:
        return x.copy()

    out = x.copy()
    onset = spec.onset
    cols = np.array(spec.affected, dtype=int)
    sigma = _pre_onset_sigma(x, onset, spec.affected)
    n_post = out.shape[0] - onset

    match spec.scenario:
        case "sudden_shift":
            out[onset:, cols] += spec.magnitude * sigma

        case "gradual_shift":
            # Mixture ramp: each post-onset row is drawn from the shifted regime with a
            # probability rising linearly over `ramp` rows. This is the textbook definition
            # of gradual drift and is distinct from `incremental_shift`, where every row
            # moves a little rather than some rows moving all the way.
            ramp = int(spec.params.get("ramp", 0)) or max(1, n_post // 4)
            p = np.clip(np.arange(n_post) / ramp, 0.0, 1.0)
            switched = rng.random(n_post) < p
            rows = onset + np.flatnonzero(switched)
            out[np.ix_(rows, cols)] += spec.magnitude * sigma

        case "incremental_shift":
            ramp = np.linspace(0.0, 1.0, num=n_post, endpoint=True)
            out[onset:, cols] += spec.magnitude * sigma * ramp[:, None]

        case "variance_shift":
            # Spread deviations around the pre-onset mean without moving the mean, so a
            # detector that only tracks location has nothing to find.
            mu = np.mean(x[:onset, cols], axis=0)
            out[onset:, cols] = mu + (out[onset:, cols] - mu) * (1.0 + spec.magnitude)

        case "correlation_break":
            # Block-permute post-onset rows of the affected columns only. Every marginal is
            # preserved exactly (it is a permutation of the same values) and short-range
            # autocorrelation survives inside a block, but the alignment with the untouched
            # columns is destroyed. Magnitude is not used: this drift is structural.
            block = max(1, int(spec.params.get("block", 1)))
            order = _block_permutation(n_post, block, rng)
            out[onset:, cols] = out[onset:, cols][order]

        case _:  # pragma: no cover - guarded by ScenarioName
            raise ValueError(f"unknown scenario {spec.scenario!r}")

    return out


def _block_permutation(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Row order produced by shuffling contiguous blocks of length `block`."""
    n_blocks = int(np.ceil(n / block))
    block_order = rng.permutation(n_blocks)
    idx = np.concatenate(
        [np.arange(b * block, min((b + 1) * block, n)) for b in block_order]
    )
    return idx[:n]


def make_spec(
    scenario: ScenarioName,
    n_rows: int,
    n_features: int,
    *,
    onset_fraction: float,
    n_affected: int,
    magnitude: float,
    ramp: int,
    block: int,
    excluded: np.ndarray,
    rng: np.random.Generator,
) -> DriftSpec:
    if scenario == "none":
        return DriftSpec(scenario="none", onset=-1, affected=(), magnitude=0.0)

    onset = round(n_rows * onset_fraction)
    affected = choose_features(n_features, n_affected, excluded, rng)
    params: dict[str, float] = {}
    if scenario == "gradual_shift":
        params["ramp"] = float(ramp)
    if scenario == "correlation_break":
        params["block"] = float(block)
    return DriftSpec(
        scenario=scenario,
        onset=onset,
        affected=affected,
        magnitude=float(magnitude),
        params=params,
    )
