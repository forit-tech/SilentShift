"""Calibration, alarm extraction and the statistics built on them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silentshift.evaluation.metrics import (
    attribution_scores,
    bootstrap_ci,
    calibrate_thresholds,
    detectability_table,
    detection_within,
    intervals_overlap,
    summarise_streams,
)


def clean_frame(machine: str, detector: str, scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stream_id": [f"{machine}|none|0|{i // 4}" for i in range(len(scores))],
            "machine": machine,
            "detector": detector,
            "policy": "fixed",
            "region": "clean",
            "window": list(range(len(scores))),
            "score": scores,
            "scenario": "none",
            "magnitude": 0.0,
            "seed": 0,
        }
    )


# --- threshold calibration ---------------------------------------------------

def test_threshold_respects_the_false_alarm_budget() -> None:
    scores = [0.1] * 6 + [0.5, 0.9]  # 8 clean windows
    thresholds = calibrate_thresholds(clean_frame("m1", "d", scores), 0.125)
    row = thresholds.iloc[0]
    assert row["achieved_alarms_per_window"] <= 0.125
    assert sum(s > row["value"] for s in scores) / len(scores) <= 0.125


def test_threshold_is_the_lowest_admissible_one() -> None:
    scores = [0.1] * 6 + [0.5, 0.9]
    thresholds = calibrate_thresholds(clean_frame("m1", "d", scores), 0.125)
    value = float(thresholds.iloc[0]["value"])
    # Anything lower would admit more alarms than the budget allows.
    assert sum(s > 0.4 for s in scores) / len(scores) > 0.125
    assert value >= 0.5


def test_thresholds_are_independent_per_machine() -> None:
    quiet = clean_frame("quiet", "d", [0.01] * 8)
    noisy = clean_frame("noisy", "d", [100.0] * 8)
    thresholds = calibrate_thresholds(pd.concat([quiet, noisy]), 0.125)
    by_machine = thresholds.set_index("machine")["value"].to_dict()
    assert by_machine["quiet"] < by_machine["noisy"]


def test_calibration_needs_clean_windows() -> None:
    frame = clean_frame("m1", "d", [0.1, 0.2])
    frame["region"] = "post"
    with pytest.raises(ValueError, match="no clean windows"):
        calibrate_thresholds(frame, 0.125)


# --- alarm extraction --------------------------------------------------------

def stream_frame(scores_by_region: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    idx = 0
    for region, scores in scores_by_region.items():
        for s in scores:
            rows.append(
                {
                    "stream_id": "m1|sudden_shift|2|0",
                    "machine": "m1",
                    "detector": "d",
                    "policy": "fixed",
                    "region": region,
                    "window": idx,
                    "score": s,
                    "scenario": "sudden_shift",
                    "magnitude": 2.0,
                    "seed": 0,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


THRESHOLDS = pd.DataFrame(
    [{"machine": "m1", "detector": "d", "policy": "fixed", "value": 1.0,
      "achieved_alarms_per_window": 0.0}]
)


def test_delay_counts_windows_after_the_first_post_window() -> None:
    frame = stream_frame({"clean": [0.0, 0.0], "straddle": [0.0], "post": [0.0, 0.0, 5.0, 0.0]})
    summary = summarise_streams(frame, THRESHOLDS)
    assert summary.iloc[0]["delay"] == 2.0
    assert bool(summary.iloc[0]["detected"])


def test_immediate_detection_has_zero_delay() -> None:
    frame = stream_frame({"clean": [0.0], "post": [9.0, 0.0]})
    assert summarise_streams(frame, THRESHOLDS).iloc[0]["delay"] == 0.0


def test_a_stream_with_no_post_alarm_is_a_miss() -> None:
    frame = stream_frame({"clean": [0.0], "post": [0.1, 0.2]})
    summary = summarise_streams(frame, THRESHOLDS)
    assert not bool(summary.iloc[0]["detected"])
    assert np.isinf(summary.iloc[0]["delay"])


def test_pre_onset_alarms_are_false_alarms_not_early_detections() -> None:
    frame = stream_frame({"clean": [9.0, 9.0], "post": [0.0, 0.0]})
    summary = summarise_streams(frame, THRESHOLDS)
    assert summary.iloc[0]["false_alarms"] == 2
    assert not bool(summary.iloc[0]["detected"])


def test_straddling_windows_count_as_neither() -> None:
    frame = stream_frame({"clean": [0.0], "straddle": [9.0, 9.0], "post": [0.0]})
    summary = summarise_streams(frame, THRESHOLDS)
    assert summary.iloc[0]["false_alarms"] == 0
    assert not bool(summary.iloc[0]["detected"])
    assert summary.iloc[0]["straddle_windows"] == 2


def test_late_detection_is_scored_as_a_miss() -> None:
    frame = stream_frame({"clean": [0.0], "post": [0.0] * 6 + [9.0]})
    summary = summarise_streams(frame, THRESHOLDS)
    assert bool(summary.iloc[0]["detected"])
    assert not bool(detection_within(summary, tolerance=4).iloc[0])


def test_streams_without_a_threshold_are_dropped_not_defaulted() -> None:
    frame = stream_frame({"clean": [0.0], "post": [9.0]})
    other = pd.DataFrame(
        [{"machine": "other", "detector": "d", "policy": "fixed", "value": 1.0,
          "achieved_alarms_per_window": 0.0}]
    )
    assert summarise_streams(frame, other).empty


# --- bootstrap and reporting -------------------------------------------------

def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    values = np.random.default_rng(0).normal(loc=5.0, size=500)
    point, lo, hi = bootstrap_ci(values, "mean", n_samples=500, seed=1)
    assert lo < point < hi
    assert abs(point - 5.0) < 0.2


def test_bootstrap_ignores_non_finite_values() -> None:
    values = np.array([1.0, 2.0, 3.0, np.inf, np.nan])
    point, lo, hi = bootstrap_ci(values, "mean", n_samples=200, seed=1)
    assert point == pytest.approx(2.0)
    assert np.isfinite([lo, hi]).all()


def test_bootstrap_of_nothing_is_nan_not_zero() -> None:
    point, lo, hi = bootstrap_ci(np.array([np.inf, np.nan]), n_samples=10)
    assert all(np.isnan(v) for v in (point, lo, hi))


def test_interval_overlap_logic() -> None:
    assert intervals_overlap(0.1, 0.4, 0.3, 0.6)
    assert not intervals_overlap(0.1, 0.2, 0.5, 0.9)
    assert intervals_overlap(0.1, 0.9, 0.4, 0.5)  # containment counts as overlap


def test_detectability_table_reports_recall_per_cell() -> None:
    summary = pd.DataFrame(
        {
            "detector": ["d"] * 4,
            "scenario": ["sudden_shift"] * 4,
            "magnitude": [2.0] * 4,
            "delay": [0.0, 1.0, np.inf, 2.0],
            "detected": [True, True, False, True],
            "false_alarms": [0, 1, 0, 0],
        }
    )
    table = detectability_table(summary, tolerance=4, n_bootstrap=200, seed=0)
    row = table.iloc[0]
    assert row["n_streams"] == 4
    assert row["recall"] == pytest.approx(0.75)
    assert row["median_delay"] == pytest.approx(1.0)
    # No false-alarm column: it is measured on drift-free streams, not on pre-onset windows.
    assert "false_alarms_per_stream" not in row


# --- attribution -------------------------------------------------------------

def test_perfect_attribution_scores_one() -> None:
    contribution = np.array([0.0, 9.0, 0.0, 8.0, 7.0])
    assert attribution_scores(contribution, (1, 3, 4))["precision_at_k"] == pytest.approx(1.0)


def test_attribution_is_zero_when_the_ranking_is_inverted() -> None:
    contribution = np.array([9.0, 0.0, 8.0, 0.0, 7.0])
    assert attribution_scores(contribution, (1, 3))["precision_at_k"] == pytest.approx(0.0)


def test_attribution_of_an_empty_truth_set_is_nan() -> None:
    assert np.isnan(attribution_scores(np.arange(5.0), ())["precision_at_k"])
