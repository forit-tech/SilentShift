"""Threshold-free comparison, and the calibration-transfer diagnostic.

Why this module exists.

The first full run calibrated a threshold per machine on the earlier part of that machine's
clean history and applied it to the later part. It did not transfer: the measured false-alarm
rate on drift-free evaluation streams came out 4-17x the target. At thresholds that loose, the
`random` control — which ignores the data completely — reached recall 1.00 on every scenario.
That single number invalidates every recall figure computed at those thresholds, which is what
the control was put there to do.

Two questions were tangled together and have to be separated:

1. **Discriminability.** Can the statistic tell a post-onset window from a drift-free one at
   all? This is a property of the statistic and needs no threshold: `window_auc` answers it,
   and a data-independent control sits at 0.5 by construction.

2. **Calibration transfer.** Does a threshold fitted on earlier clean data still hold later?
   `calibration_transfer` answers it, and on this data the answer is no.

Ranking detectors on (1) while reporting (2) as a limitation is honest. Ranking them on
recall at a threshold that does not transfer is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .metrics import bootstrap_ci


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    y = np.concatenate([np.ones(positive.size), np.zeros(negative.size)])
    s = np.concatenate([positive, negative])
    if np.unique(s).size == 1:  # a fully saturated statistic carries no information
        return 0.5
    return float(roc_auc_score(y, s))


def window_auc(
    scores: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Separability of post-onset windows from drift-free windows, per detector and scenario.

    Negatives are windows from `scenario == "none"` streams on the *same* machines, so the
    comparison is within-fleet and does not reward a detector for merely reacting to the
    difference between machines.

    The bootstrap resamples **streams**, not windows: windows inside a stream overlap by 2000
    of 2500 rows and are not independent evidence.
    """
    drift_free = scores[scores["scenario"] == "none"]
    drifted = scores[(scores["scenario"] != "none") & (scores["region"] == "post")]

    rows: list[dict[str, object]] = []
    for (detector, scenario, magnitude), group in drifted.groupby(
        ["detector", "scenario", "magnitude"], sort=True
    ):
        null_by_machine: dict[str, np.ndarray] = {
            str(machine): sub["score"].to_numpy()
            for machine, sub in drift_free[drift_free["detector"] == detector].groupby("machine")
        }

        # Per machine, then averaged. Pooling the null across the fleet would let a detector
        # score by telling machines apart -- and the machines differ by orders of magnitude on
        # every statistic here, so that inflation would be large and invisible.
        per_stream: list[tuple[np.ndarray, np.ndarray]] = []
        for (machine, _), stream in group.groupby(["machine", "stream_id"], sort=True):
            negative = null_by_machine.get(str(machine))
            if negative is None or negative.size == 0:
                continue
            per_stream.append((stream["score"].to_numpy(), negative))
        if not per_stream:
            continue

        # One AUC per stream, computed once. Resampling streams changes *which* of these are
        # averaged, never their values, so recomputing the AUC inside the bootstrap loop would
        # repeat the same few hundred calculations a few hundred thousand times for an
        # arithmetically identical answer.
        per_stream_auc = np.array([_auc(pos, neg) for pos, neg in per_stream])
        finite = per_stream_auc[np.isfinite(per_stream_auc)]
        if finite.size == 0:
            continue
        point = float(np.mean(finite))

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, finite.size, size=(n_bootstrap, finite.size))
        draws = np.mean(finite[idx], axis=1)
        lo, hi = np.nanquantile(draws, [0.025, 0.975])

        rows.append(
            {
                "detector": detector,
                "scenario": scenario,
                "magnitude": magnitude,
                "n_streams": len(per_stream),
                "n_machines": int(group["machine"].nunique()),
                "bootstrap_draws": n_bootstrap,
                "auc": round(point, 4),
                "auc_lo": round(float(lo), 4),
                "auc_hi": round(float(hi), 4),
            }
        )
    return pd.DataFrame(rows)


def calibration_transfer(
    evaluation_scores: pd.DataFrame,
    thresholds: pd.DataFrame,
    target_alarms_per_window: float,
) -> pd.DataFrame:
    """Did the frozen threshold keep its promised false-alarm rate on later clean data?

    The gap is the headline limitation of the project: a per-machine threshold fitted on one
    stretch of drift-free history does not hold on a later stretch of the *same* machine.
    """
    drift_free = evaluation_scores[evaluation_scores["scenario"] == "none"]
    key = thresholds.set_index(["machine", "detector", "policy"])["value"].to_dict()

    rows: list[dict[str, object]] = []
    for detector, group in drift_free.groupby("detector", sort=True):
        alarms = 0
        total = 0
        for (machine, policy), sub in group.groupby(["machine", "policy"], sort=False):
            threshold = key.get((machine, detector, policy))
            if threshold is None:
                continue
            alarms += int(np.count_nonzero(sub["score"].to_numpy() > threshold))
            total += len(sub)
        if total == 0:
            continue
        achieved = alarms / total
        rows.append(
            {
                "detector": detector,
                "target_alarms_per_window": target_alarms_per_window,
                "achieved_alarms_per_window": round(achieved, 4),
                "inflation_factor": round(achieved / target_alarms_per_window, 2)
                if target_alarms_per_window > 0
                else float("nan"),
                "null_windows": total,
            }
        )
    return pd.DataFrame(rows)


def _horizon_blocks(streams: list[np.ndarray], horizon: int) -> list[np.ndarray]:
    """Every length-`horizon` block of every stream.

    Using only the first block of each drift-free stream would discard two thirds of the null
    data and leave the threshold noisier than the data allows.
    """
    blocks: list[np.ndarray] = []
    for s in streams:
        if s.size <= horizon:
            blocks.append(s)
            continue
        blocks.extend(s[i : i + horizon] for i in range(s.size - horizon + 1))
    return blocks


def _stream_alarm_rate(blocks: list[np.ndarray], threshold: float) -> float:
    """Fraction of horizon-length blocks containing at least one alarm."""
    if not blocks:
        return float("nan")
    return float(np.mean([bool(np.any(b > threshold)) for b in blocks]))


def matched_far_recall(
    scores: pd.DataFrame,
    target_stream_false_positive_rate: float,
    tolerance_windows: int,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Recall at an operating point matched on **per-stream** false positives.

    Two corrections over the obvious per-window version, both forced by the control detector.

    First, the budget is per stream over the same horizon the recall uses. A per-window budget
    of 0.125 sounds strict, but a detection is credited if *any* of the first
    `tolerance_windows + 1` windows alarms, so a data-independent scorer reaches
    ``1 - (1 - 0.125)**5 ~= 0.49``. Measuring recall against a 0.125 budget therefore compares
    detectors to a chance floor of 0.49 while implying the floor is 0.125. Calibrating on the
    same horizon makes the chance floor equal the target by construction.

    Second, a saturated statistic gets `NaN`, not zero. When a score is pinned at its maximum
    on drift-free data — `c2st` and `ks_bonferroni` both are — no threshold reaches the target
    rate, and reporting 0.0 would read as "found nothing" when the truth is "this statistic has
    no usable operating point at all".

    This is an **oracle** point: it uses drift-free data from the evaluated period and could not
    be set in deployment. It is reported to separate two different failures — a statistic that
    cannot discriminate, and a threshold that cannot be transported.
    """
    horizon = tolerance_windows + 1
    drift_free = scores[scores["scenario"] == "none"]
    drifted = scores[scores["scenario"] != "none"]

    rows: list[dict[str, object]] = []
    for detector, null_group in drift_free.groupby("detector", sort=True):
        null_streams = [
            g.sort_values("window")["score"].to_numpy()
            for _, g in null_group.groupby("stream_id", sort=True)
        ]
        if not null_streams:
            continue

        # Smallest threshold whose per-horizon false-positive rate meets the target.
        null_blocks = _horizon_blocks(null_streams, horizon)
        candidates = np.unique(np.concatenate(null_streams))
        threshold = float("nan")
        achieved = float("nan")
        for t in candidates:
            rate = _stream_alarm_rate(null_blocks, float(t))
            if rate <= target_stream_false_positive_rate:
                threshold, achieved = float(t), rate
                break

        # A threshold at or above the largest drift-free score is not an operating point: the
        # only way to satisfy the budget was to make the alarm unreachable. This is what a
        # saturated statistic looks like, and `c2st` and `ks_bonferroni` both land here because
        # their scores pile up on the maximum even with no drift present.
        saturated = not np.isfinite(threshold) or threshold >= float(candidates[-1])

        sub = drifted[drifted["detector"] == detector]
        for (scenario, magnitude), group in sub.groupby(["scenario", "magnitude"], sort=True):
            hits: list[float] = []
            for _, stream in group.groupby("stream_id", sort=True):
                post = stream[stream["region"] == "post"].sort_values("window")
                if post.empty:
                    continue
                hits.append(
                    float(np.any(post["score"].to_numpy()[:horizon] > threshold))
                    if not saturated
                    else float("nan")
                )
            if not hits:
                continue
            if saturated:
                point = lo = hi = float("nan")
            else:
                point, lo, hi = bootstrap_ci(
                    np.asarray(hits), "mean", n_samples=n_bootstrap, seed=seed
                )
            rows.append(
                {
                    "detector": detector,
                    "scenario": scenario,
                    "magnitude": magnitude,
                    "saturated": saturated,
                    "oracle_threshold": None if saturated else round(threshold, 6),
                    "achieved_stream_fpr": None if saturated else round(achieved, 4),
                    "n_streams": len(hits),
                    "recall": None if saturated else round(point, 4),
                    "recall_lo": None if saturated else round(lo, 4),
                    "recall_hi": None if saturated else round(hi, 4),
                }
            )
    return pd.DataFrame(rows)
