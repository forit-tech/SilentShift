"""Alarm extraction, calibration and metrics.

Two decisions here carry most of the weight.

**Thresholds are calibrated on drift-free calibration-split streams**, at a chosen number of
false alarms per stream. Comparing detectors at their own arbitrary scales is meaningless;
comparing them at a matched false-alarm budget is the only fair comparison, and it is also
the one an operator actually faces.

**False alarms are counted on separate drift-free streams**, never on the pre-onset part of
a drifted stream. Using the pre-onset region would make the false-alarm rate depend on where
the onset happens to sit, which is an artefact of the injection rather than a property of
the detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOWS_NOT_DETECTED = np.inf


@dataclass(frozen=True)
class Threshold:
    machine: str
    detector: str
    policy: str
    value: float
    achieved_alarms_per_window: float


def calibrate_thresholds(
    calibration_scores: pd.DataFrame, target_alarms_per_window: float
) -> pd.DataFrame:
    """Lowest threshold respecting the false-alarm budget, per machine, detector and policy.

    Per machine, not global. The null score distribution varies between machines by orders
    of magnitude -- on drift-free SMD data the median PCA reconstruction ratio spans more
    than three decades across the fleet -- so one global cut-off would be set by whichever
    machine happens to be noisiest and would say nothing about any of them. Production
    monitoring baselines per entity for exactly this reason.

    The budget is expressed **per window**, not per stream, so that calibration streams and
    evaluation streams of different lengths are directly comparable. An earlier version used a
    per-stream budget with 4-window calibration streams and 16-window evaluation streams; the
    two were not comparable and the measured false-alarm rate missed its target by 3-4x.

    Lowest admissible rather than any: raising the threshold further would only cost sensitivity, and
    picking the largest admissible one would flatter whichever detector has the heaviest
    score tail.
    """
    clean = calibration_scores[calibration_scores["region"] == "clean"]
    if clean.empty:
        raise ValueError("calibration set contains no clean windows")

    rows: list[Threshold] = []
    for (machine, detector, policy), group in clean.groupby(
        ["machine", "detector", "policy"], sort=True
    ):
        n_windows = len(group)
        scores = np.sort(group["score"].to_numpy())
        # Candidate thresholds are the observed scores; anything between two observations
        # produces an identical alarm count.
        candidates = np.unique(scores)
        chosen = float(candidates[-1]) + 1.0
        achieved = 0.0
        for t in candidates:
            alarms = int(np.count_nonzero(scores > t))
            rate = alarms / n_windows
            if rate <= target_alarms_per_window:
                chosen, achieved = float(t), rate
                break
        rows.append(
            Threshold(
                machine=str(machine),
                detector=str(detector),
                policy=str(policy),
                value=chosen,
                achieved_alarms_per_window=achieved,
            )
        )
    return pd.DataFrame([r.__dict__ for r in rows])


def _first_alarm_delay(group: pd.DataFrame, threshold: float) -> float:
    post = group[group["region"] == "post"].sort_values("window")
    if post.empty:
        return WINDOWS_NOT_DETECTED
    alarmed = post[post["score"] > threshold]
    if alarmed.empty:
        return WINDOWS_NOT_DETECTED
    first_post_window = int(post["window"].iloc[0])
    return float(int(alarmed["window"].iloc[0]) - first_post_window)


def summarise_streams(scores: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """One row per (stream, detector, policy): delay, false alarms, detection flag."""
    key = thresholds.set_index(["machine", "detector", "policy"])["value"].to_dict()
    rows: list[dict[str, object]] = []
    group_cols = ["stream_id", "machine", "detector", "policy"]
    for (stream_id, machine, detector, policy), group in scores.groupby(group_cols, sort=False):
        threshold = key.get((machine, detector, policy))
        if threshold is None:
            continue
        clean = group[group["region"] == "clean"]
        delay = _first_alarm_delay(group, threshold)
        first = group.iloc[0]
        rows.append(
            {
                "stream_id": stream_id,
                "machine": machine,
                "scenario": first["scenario"],
                "magnitude": first["magnitude"],
                "seed": first["seed"],
                "detector": detector,
                "policy": policy,
                "delay": delay,
                "detected": bool(np.isfinite(delay)),
                "false_alarms": int(np.count_nonzero(clean["score"].to_numpy() > threshold)),
                "clean_windows": len(clean),
                "straddle_windows": int(np.count_nonzero(group["region"] == "straddle")),
            }
        )
    return pd.DataFrame(rows)


def detection_within(summary: pd.DataFrame, tolerance: float) -> pd.Series:
    """Detected *and* within the tolerance window. Late detection is not detection."""
    return summary["detected"] & (summary["delay"] <= tolerance)


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    n_samples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap over streams. Returns (point, low, high).

    Resampling is over streams, which are the independent units. Resampling windows would
    treat overlapping slices of one stream as independent evidence and shrink the interval
    to something meaningless.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"),) * 3

    if statistic == "mean":
        point = float(np.mean(values))
    elif statistic == "median":
        point = float(np.median(values))
    else:
        raise ValueError(f"unsupported statistic {statistic!r}")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_samples, values.size))
    resampled = values[idx]
    draws = np.mean(resampled, axis=1) if statistic == "mean" else np.median(resampled, axis=1)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(low), float(high)


def detectability_table(
    summary: pd.DataFrame, tolerance: float, n_bootstrap: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Recall by detector x scenario x magnitude, with bootstrap intervals over streams."""
    summary = summary.copy()
    summary["hit"] = detection_within(summary, tolerance).astype(float)
    rows: list[dict[str, object]] = []
    for (detector, scenario, magnitude), group in summary.groupby(
        ["detector", "scenario", "magnitude"], sort=True
    ):
        point, low, high = bootstrap_ci(
            group["hit"].to_numpy(), "mean", n_samples=n_bootstrap, seed=seed
        )
        finite = group.loc[np.isfinite(group["delay"]), "delay"].to_numpy()
        rows.append(
            {
                "detector": detector,
                "scenario": scenario,
                "magnitude": magnitude,
                "n_streams": len(group),
                "recall": point,
                "recall_lo": low,
                "recall_hi": high,
                "median_delay": float(np.median(finite)) if finite.size else float("nan"),
                # No false-alarm column here on purpose. Counting alarms in the pre-onset
                # region of a drifted stream makes the rate depend on where the onset was
                # placed, which is a property of the injection. It is measured on drift-free
                # streams instead -- see false_alarm_rate_*.csv.
            }
        )
    return pd.DataFrame(rows)


def intervals_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    """Whether two confidence intervals overlap.

    Used to refuse a superiority claim. Non-overlap is a conservative test — it is stricter
    than a paired comparison — but it is the right default when the alternative is a README
    that declares a winner on a difference the data cannot support.
    """
    return not (hi_a < lo_b or hi_b < lo_a)


def attribution_scores(contribution: np.ndarray, affected: tuple[int, ...]) -> dict[str, float]:
    """Precision of the top-k attributed features against ground truth.

    One number, not two: when exactly k features are selected and exactly k are affected,
    precision and recall are arithmetically identical, and reporting both would look like two
    pieces of evidence where there is one.

    k equals the true number of affected features, which a real operator would not know, so
    this is an upper bound on attribution quality. Guessing at random scores k/n_features.
    """
    k = len(affected)
    if k == 0 or contribution.size == 0:
        return {"precision_at_k": float("nan"), "k": float(k)}
    top = set(np.argsort(contribution)[::-1][:k].tolist())
    hits = len(top & set(affected))
    return {"precision_at_k": hits / k, "k": float(k)}
