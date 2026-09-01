"""Server Machine Dataset loader.

Layout on disk (as published with OmniAnomaly, MIT licence):

    ServerMachineDataset/
        train/machine-<g>-<i>.txt              38 comma-separated floats per row
        test/machine-<g>-<i>.txt
        test_label/machine-<g>-<i>.txt         one 0/1 per row
        interpretation_label/machine-<g>-<i>.txt   "start-end:dim,dim,..." (1-indexed dims)

Two facts about this data drive most of the design:

1. The published values are already min-max scaled to [0, 1] by the dataset authors,
   globally per machine. Absolute units are gone, so any statistic that would need raw
   scale is meaningless here, and injected magnitudes are expressed in units of the
   per-feature standard deviation of the reference window rather than in raw units.

2. `train` carries no labels and is conventionally treated as the clean half. We use it
   as the drift-free substrate. It is *not* guaranteed anomaly-free — that assumption is
   inherited from the dataset, not verified by us, and it is recorded as a limitation.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_SEGMENT = re.compile(r"^(\d+)-(\d+):([\d,]+)\s*$")


@dataclass(frozen=True)
class AnomalySegment:
    """One labelled anomaly interval from `interpretation_label`."""

    start: int
    end: int
    dimensions: tuple[int, ...]  # zero-indexed, converted from the file's 1-indexed form


@dataclass(frozen=True)
class Machine:
    name: str
    train: np.ndarray  # (T_train, 38) drift-free substrate
    test: np.ndarray  # (T_test, 38)
    test_labels: np.ndarray  # (T_test,) 0/1
    segments: tuple[AnomalySegment, ...]

    @property
    def n_features(self) -> int:
        return int(self.train.shape[1])


def machine_names(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "train").glob("machine-*.txt"))


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of one file, used to pin the exact data a run used."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def dataset_digest(root: Path) -> str:
    """Order-independent digest over every train file, for provenance records."""
    per_file = sorted(file_digest(p) for p in (root / "train").glob("machine-*.txt"))
    return hashlib.sha256("".join(per_file).encode("ascii")).hexdigest()


def _read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)


def _read_segments(path: Path) -> tuple[AnomalySegment, ...]:
    if not path.exists():
        return ()
    out: list[AnomalySegment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = _SEGMENT.match(line)
        if m is None:
            log.warning("unparsable interpretation line in %s: %r", path.name, line)
            continue
        start, end, dims = int(m.group(1)), int(m.group(2)), m.group(3)
        # The file is 1-indexed over the 38 columns; everything downstream is 0-indexed.
        zero_indexed = tuple(sorted(int(d) - 1 for d in dims.split(",") if d))
        out.append(AnomalySegment(start=start, end=end, dimensions=zero_indexed))
    return tuple(out)


@lru_cache(maxsize=32)
def load_machine(root: Path, name: str) -> Machine:
    """Load one machine. Cached because evaluation re-reads the same machines often."""
    train = _read_matrix(root / "train" / f"{name}.txt")
    test = _read_matrix(root / "test" / f"{name}.txt")
    labels = np.loadtxt(root / "test_label" / f"{name}.txt", dtype=np.int64, ndmin=1)
    segments = _read_segments(root / "interpretation_label" / f"{name}.txt")

    if train.shape[1] != test.shape[1]:
        raise ValueError(f"{name}: train has {train.shape[1]} columns, test has {test.shape[1]}")
    if labels.shape[0] != test.shape[0]:
        raise ValueError(f"{name}: {labels.shape[0]} labels for {test.shape[0]} test rows")

    return Machine(name=name, train=train, test=test, test_labels=labels, segments=segments)


CALIBRATION_FRACTION = 0.35


def segment_bounds(machine: Machine, part: str) -> tuple[int, int]:
    """Temporally disjoint halves of one machine's clean history.

    Thresholds are calibrated per machine, because the null score distribution differs
    between machines by orders of magnitude and a single global threshold is therefore
    meaningless. That is only legitimate if calibration data and evaluation data never
    overlap in time, which is what this split guarantees: calibration reads the first 35%
    of a machine's clean history, evaluation reads the remaining 65%, and no row is in both.
    """
    total = machine.train.shape[0]
    cut = int(total * CALIBRATION_FRACTION)
    match part:
        case "calibration":
            return 0, cut
        case "evaluation":
            return cut, total
        case _:
            raise ValueError(f"unknown part {part!r}")


def drift_free_segment(
    machine: Machine, length: int, rng: np.random.Generator, part: str = "evaluation"
) -> np.ndarray:
    """Take a contiguous slice of one half of the clean history, at a random offset.

    Contiguous rather than sampled: shuffling rows would destroy the autocorrelation that
    makes this substrate realistic, and autocorrelation is precisely what breaks the
    nominal false-alarm rates of the two-sample tests we evaluate.
    """
    lo, hi = segment_bounds(machine, part)
    available = hi - lo
    if length > available:
        raise ValueError(f"{machine.name}: asked for {length} rows, {part} half has {available}")
    start = lo + int(rng.integers(0, available - length + 1))
    return np.ascontiguousarray(machine.train[start : start + length])


def constant_feature_mask(x: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Features with (near) zero variance.

    SMD has several columns that are constant for a whole machine. They break scale-relative
    injection (a shift of `k` standard deviations is undefined when sigma is 0) and make
    several divergence statistics degenerate, so callers exclude them explicitly rather than
    letting a NaN propagate into a metric.
    """
    return np.std(x, axis=0) <= tol
