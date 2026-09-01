"""SMD loading, splits and the leakage guarantees they encode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from silentshift.data.smd import (
    AnomalySegment,
    Machine,
    _read_segments,
    constant_feature_mask,
    drift_free_segment,
    file_digest,
    segment_bounds,
)


def fake_machine(rows: int = 10_000, features: int = 8) -> Machine:
    rng = np.random.default_rng(0)
    return Machine(
        name="machine-t-1",
        train=rng.normal(size=(rows, features)),
        test=rng.normal(size=(rows, features)),
        test_labels=np.zeros(rows, dtype=np.int64),
        segments=(),
    )


# --- splits ------------------------------------------------------------------

def test_calibration_and_evaluation_halves_do_not_overlap() -> None:
    machine = fake_machine()
    cal_lo, cal_hi = segment_bounds(machine, "calibration")
    ev_lo, ev_hi = segment_bounds(machine, "evaluation")
    assert cal_hi == ev_lo
    assert set(range(cal_lo, cal_hi)).isdisjoint(range(ev_lo, ev_hi))


def test_the_two_halves_cover_the_whole_clean_history() -> None:
    machine = fake_machine(rows=12_345)
    cal_lo, cal_hi = segment_bounds(machine, "calibration")
    ev_lo, ev_hi = segment_bounds(machine, "evaluation")
    assert cal_lo == 0
    assert ev_hi == machine.train.shape[0]
    assert (cal_hi - cal_lo) + (ev_hi - ev_lo) == machine.train.shape[0]


def test_unknown_part_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown part"):
        segment_bounds(fake_machine(), "test")


def test_segments_are_drawn_from_the_requested_half_only() -> None:
    """The guarantee that makes per-machine thresholds legitimate."""
    machine = fake_machine()
    cal_lo, cal_hi = segment_bounds(machine, "calibration")
    for seed in range(25):
        rng = np.random.default_rng(seed)
        chunk = drift_free_segment(machine, 2000, rng, part="calibration")
        # Every returned row must exist somewhere inside the calibration half.
        matches = [
            np.array_equal(chunk, machine.train[s : s + 2000])
            for s in range(cal_lo, cal_hi - 2000 + 1)
        ]
        assert any(matches)


def test_segments_are_contiguous_not_sampled() -> None:
    machine = fake_machine()
    chunk = drift_free_segment(machine, 1500, np.random.default_rng(3), part="evaluation")
    lo, hi = segment_bounds(machine, "evaluation")
    starts = [s for s in range(lo, hi - 1500 + 1)
              if np.array_equal(chunk, machine.train[s : s + 1500])]
    assert starts, "returned chunk is not a contiguous slice of the evaluation half"


def test_asking_for_more_rows_than_the_half_holds_is_an_error() -> None:
    machine = fake_machine(rows=10_000)
    with pytest.raises(ValueError, match="asked for"):
        drift_free_segment(machine, 9000, np.random.default_rng(0), part="calibration")


def test_segment_selection_is_reproducible() -> None:
    machine = fake_machine()
    a = drift_free_segment(machine, 1000, np.random.default_rng(7))
    b = drift_free_segment(machine, 1000, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


# --- interpretation labels ---------------------------------------------------

def test_interpretation_labels_are_converted_to_zero_indexed(tmp_path: Path) -> None:
    path = tmp_path / "machine-1-1.txt"
    path.write_text("15849-16368:1,9,10\n16963-17517:2,3\n", encoding="utf-8")
    segments = _read_segments(path)
    assert segments == (
        AnomalySegment(15849, 16368, (0, 8, 9)),
        AnomalySegment(16963, 17517, (1, 2)),
    )


def test_unparsable_lines_are_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "machine-1-1.txt"
    path.write_text("garbage\n10-20:1,2\n\n", encoding="utf-8")
    assert _read_segments(path) == (AnomalySegment(10, 20, (0, 1)),)


def test_missing_interpretation_file_yields_no_segments(tmp_path: Path) -> None:
    assert _read_segments(tmp_path / "absent.txt") == ()


# --- misc --------------------------------------------------------------------

def test_constant_feature_mask_flags_only_constant_columns() -> None:
    x = np.column_stack([np.arange(100.0), np.ones(100), np.zeros(100)])
    np.testing.assert_array_equal(constant_feature_mask(x), [False, True, True])


def test_file_digest_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("hello", encoding="utf-8")
    assert file_digest(a) == file_digest(b)
    b.write_text("hello!", encoding="utf-8")
    assert file_digest(a) != file_digest(b)
