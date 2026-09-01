"""Stream construction and scoring.

One *stream* is the unit of evaluation: a contiguous drift-free slice of one machine, with
at most one injected change, scored window by window. Everything downstream aggregates over
streams and never over windows, because windows inside a stream overlap and are strongly
dependent — averaging over them would manufacture precision that does not exist.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data.smd import Machine, constant_feature_mask, drift_free_segment, load_machine
from .detectors import build as build_detector
from .injection.catalogue import SCENARIOS, DriftSpec, ScenarioName, inject, make_spec
from .windows.policy import ReferencePolicy, ReferenceTracker, Window, enumerate_windows

log = logging.getLogger(__name__)

# Magnitude is meaningless for these scenarios, so the grid collapses to a single entry
# instead of silently producing identical duplicate streams under different labels.
MAGNITUDE_FREE: frozenset[ScenarioName] = frozenset({"none", "correlation_break"})


@dataclass(frozen=True)
class Stream:
    machine: str
    split: str
    seed: int
    data: np.ndarray
    spec: DriftSpec

    @property
    def stream_id(self) -> str:
        return f"{self.machine}|{self.spec.scenario}|{self.spec.magnitude:g}|{self.seed}"


def build_stream(
    machine: Machine,
    scenario: ScenarioName,
    magnitude: float,
    seed: int,
    cfg: ExperimentConfig,
    split: str,
    length: int,
    part: str = "evaluation",
) -> Stream:
    # Not `hash()`: Python randomises string hashing per process, so the choice of clean
    # slice would silently differ between runs and `make all` would not reproduce.
    rng = np.random.default_rng(
        stream_seed(f"{machine.name}|{scenario}|{magnitude:g}|{seed}|{part}")
    )
    base = drift_free_segment(machine, length, rng, part=part)
    excluded = constant_feature_mask(base)
    spec = make_spec(
        scenario,
        n_rows=base.shape[0],
        n_features=base.shape[1],
        onset_fraction=cfg.injection.onset_fraction,
        n_affected=cfg.injection.n_affected_features,
        magnitude=magnitude,
        ramp=cfg.injection.gradual_ramp,
        block=cfg.injection.block_shuffle_length,
        excluded=excluded,
        rng=rng,
    )
    data = inject(base, spec, rng)
    return Stream(machine=machine.name, split=split, seed=seed, data=data, spec=spec)


def scenario_grid(cfg: ExperimentConfig) -> list[tuple[ScenarioName, float]]:
    out: list[tuple[ScenarioName, float]] = []
    for raw in cfg.injection.scenarios:
        # Config values arrive as plain strings; narrow once, here, so the rest of the
        # pipeline can rely on the Literal type instead of re-checking it everywhere.
        if raw not in SCENARIOS:
            raise ValueError(f"unknown scenario {raw!r}; expected one of {SCENARIOS}")
        scenario = raw
        if scenario in MAGNITUDE_FREE:
            out.append((scenario, 0.0))
        else:
            out.extend((scenario, m) for m in cfg.injection.magnitudes)
    return out


def iter_streams(
    cfg: ExperimentConfig,
    split: str,
    length: int,
    scenarios: Sequence[tuple[ScenarioName, float]] | None = None,
    machines: Sequence[str] | None = None,
    part: str = "evaluation",
    seeds: Sequence[int] | None = None,
) -> Iterator[Stream]:
    grid = list(scenarios) if scenarios is not None else scenario_grid(cfg)
    names = list(machines) if machines is not None else list(cfg.data.machines_for(split))
    used_seeds = list(seeds) if seeds is not None else list(cfg.injection.seeds)
    for name in names:
        machine = load_machine(cfg.data.smd_root, name)
        for scenario, magnitude in grid:
            for seed in used_seeds:
                yield build_stream(machine, scenario, magnitude, seed, cfg, split, length, part)


def stream_seed(stream_id: str, base: int = 0) -> int:
    """Deterministic per-stream seed.

    Every stochastic detector must be seeded independently per stream. Reusing one fixed seed
    made the `random` control emit the *same* sequence on every stream, which turned the
    control into a constant and let it score a spurious recall of 1.0 -- the control has to be
    genuinely uninformative for its verdict on the other detectors to mean anything.
    Derived from the stream id rather than drawn, so runs stay reproducible.
    """
    digest = hashlib.blake2b(stream_id.encode("utf-8"), digest_size=4).digest()
    return (int.from_bytes(digest, "big") ^ base) % (2**31)


def classify_window(window: Window, onset: int) -> str:
    """Where a window sits relative to the change.

    Windows that straddle the onset contain both regimes. Counting them as false alarms
    would punish a detector for being right slightly early; counting them as detections
    would reward it for a partial view. They are excluded from both, and reported so the
    exclusion is visible rather than hidden.
    """
    if onset < 0:
        return "clean"
    if window.end <= onset:
        return "clean"
    if window.start >= onset:
        return "post"
    return "straddle"


def score_stream(
    stream: Stream,
    detector_name: str,
    policy: ReferencePolicy,
    cfg: ExperimentConfig,
    seed: int = 0,
) -> pd.DataFrame:
    """Window-level scores for one (stream, detector, policy) triple."""
    detector = build_detector(detector_name, seed=stream_seed(stream.stream_id, seed))
    windows = enumerate_windows(
        stream.data.shape[0], cfg.window.reference_size, cfg.window.size, cfg.window.stride
    )
    tracker = ReferenceTracker(stream.data, policy, cfg.window.reference_size)

    rows: list[dict[str, object]] = []
    fitted_once = False
    for window in windows:
        reference = tracker.reference_for(window)
        # A fixed reference never changes, so refitting per window would only burn time.
        if tracker.reference_changes_per_window() or not fitted_once:
            detector.fit_reference(reference)
            fitted_once = True
        chunk = stream.data[window.start : window.end]
        score = detector.score(chunk)
        rows.append(
            {
                "stream_id": stream.stream_id,
                "machine": stream.machine,
                "split": stream.split,
                "scenario": stream.spec.scenario,
                "magnitude": stream.spec.magnitude,
                "seed": stream.seed,
                "detector": detector_name,
                "policy": str(policy),
                "window": window.index,
                "window_start": window.start,
                "window_end": window.end,
                "region": classify_window(window, stream.spec.onset),
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def attribution_for_stream(
    stream: Stream,
    detector_name: str,
    cfg: ExperimentConfig,
    window_index: int,
    seed: int = 0,
) -> np.ndarray | None:
    """Per-feature contribution at one window, under the fixed-reference policy.

    Returns None when the detector does not implement attribution, so the caller can drop
    it from the attribution table instead of scoring a vector of zeros as if it were a
    prediction.
    """
    detector = build_detector(detector_name, seed=stream_seed(stream.stream_id, seed))
    windows = enumerate_windows(
        stream.data.shape[0], cfg.window.reference_size, cfg.window.size, cfg.window.stride
    )
    if not 0 <= window_index < len(windows):
        return None
    detector.fit_reference(stream.data[: cfg.window.reference_size])
    window = windows[window_index]
    chunk = stream.data[window.start : window.end]
    detector.score(chunk)
    vector = detector.attribute(chunk)
    if not np.any(vector):
        return None
    return vector
