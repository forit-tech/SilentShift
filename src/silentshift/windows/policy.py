"""Windowing and reference-window policy.

The reference policy is not an implementation detail. A sliding reference quietly absorbs
slow drift: by the time the window has moved far enough to matter, the reference has moved
with it and the comparison is between two adjacent, nearly identical slices. A detector can
therefore score near-perfect on sudden shifts and near-zero on incremental ones purely
because of this choice, with no difference in the statistic at all.

So the policy is a first-class experimental factor, evaluated for every detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class ReferencePolicy(StrEnum):
    FIXED = "fixed"
    SLIDING = "sliding"
    RESET_ON_ALARM = "reset_on_alarm"


@dataclass(frozen=True)
class Window:
    index: int
    start: int
    end: int  # exclusive

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


def enumerate_windows(n_rows: int, reference_size: int, size: int, stride: int) -> list[Window]:
    """Windows over the post-reference part of a stream.

    The first `reference_size` rows are never scored: they are the initial reference and
    scoring them would compare a sample against itself.
    """
    if size <= 0 or stride <= 0:
        raise ValueError("window size and stride must be positive")
    if reference_size + size > n_rows:
        raise ValueError(
            f"stream of {n_rows} rows is too short for reference {reference_size} + window {size}"
        )
    out: list[Window] = []
    start = reference_size
    idx = 0
    while start + size <= n_rows:
        out.append(Window(index=idx, start=start, end=start + size))
        start += stride
        idx += 1
    return out


class ReferenceTracker:
    """Supplies the reference sample for each window under a given policy.

    `fixed` and `sliding` are stateless in effect; `reset_on_alarm` is not, so the caller
    must report alarms back via `notify_alarm`. That feedback loop is the reason this is a
    class rather than a function.
    """

    def __init__(self, stream: np.ndarray, policy: ReferencePolicy, reference_size: int) -> None:
        self.stream = stream
        self.policy = policy
        self.reference_size = reference_size
        self._anchor_end = reference_size  # reference is stream[anchor_end - size : anchor_end]

    def reference_for(self, window: Window) -> np.ndarray:
        match self.policy:
            case ReferencePolicy.FIXED:
                return self.stream[: self.reference_size]
            case ReferencePolicy.SLIDING:
                end = max(self.reference_size, window.start)
                start = max(0, end - self.reference_size)
                return self.stream[start:end]
            case ReferencePolicy.RESET_ON_ALARM:
                start = max(0, self._anchor_end - self.reference_size)
                return self.stream[start : self._anchor_end]

    def notify_alarm(self, window: Window) -> None:
        """Adopt the alarming window's neighbourhood as the new normal.

        This is what an operator does after accepting a change, and it is also how a
        detector goes permanently blind to a slow ramp: each acceptance moves the baseline
        a little further, so the next comparison never sees the cumulative drift.
        """
        if self.policy is ReferencePolicy.RESET_ON_ALARM:
            self._anchor_end = max(self.reference_size, window.end)

    def reference_changes_per_window(self) -> bool:
        return self.policy is not ReferencePolicy.FIXED
