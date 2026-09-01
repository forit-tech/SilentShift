"""Windowing, reference policies and the autocorrelation machinery."""

from __future__ import annotations

import numpy as np
import pytest

from silentshift.experiment import classify_window
from silentshift.timeseries import autocorrelation_time, effective_sample_size, thin
from silentshift.windows.policy import (
    ReferencePolicy,
    ReferenceTracker,
    Window,
    enumerate_windows,
)

# --- windowing ---------------------------------------------------------------

def test_windows_start_after_the_reference() -> None:
    windows = enumerate_windows(n_rows=10_000, reference_size=4000, size=2500, stride=500)
    assert windows[0].start == 4000
    assert all(w.start >= 4000 for w in windows)


def test_windows_never_run_past_the_stream() -> None:
    windows = enumerate_windows(n_rows=10_000, reference_size=4000, size=2500, stride=500)
    assert all(w.end <= 10_000 for w in windows)
    assert windows[-1].end <= 10_000


def test_window_count_matches_the_stride() -> None:
    windows = enumerate_windows(n_rows=14_000, reference_size=4000, size=2500, stride=500)
    assert len(windows) == (14_000 - 4000 - 2500) // 500 + 1
    assert [w.index for w in windows] == list(range(len(windows)))


def test_too_short_a_stream_is_an_error_not_an_empty_list() -> None:
    # Returning [] would silently produce a stream with no evaluation and a recall of 0.
    with pytest.raises(ValueError, match="too short"):
        enumerate_windows(n_rows=5000, reference_size=4000, size=2500, stride=500)


@pytest.mark.parametrize("bad", [(0, 500), (500, 0), (-1, 100)])
def test_non_positive_geometry_is_rejected(bad: tuple[int, int]) -> None:
    size, stride = bad
    with pytest.raises(ValueError, match="must be positive"):
        enumerate_windows(10_000, 1000, size, stride)


# --- region classification ---------------------------------------------------

def test_regions_split_into_clean_straddle_and_post() -> None:
    onset = 8400
    assert classify_window(Window(0, 4000, 6500), onset) == "clean"
    assert classify_window(Window(1, 8000, 10_500), onset) == "straddle"
    assert classify_window(Window(2, 8500, 11_000), onset) == "post"


def test_a_window_ending_exactly_at_onset_is_clean() -> None:
    assert classify_window(Window(0, 5900, 8400), 8400) == "clean"


def test_a_window_starting_exactly_at_onset_is_post() -> None:
    assert classify_window(Window(0, 8400, 10_900), 8400) == "post"


def test_every_window_of_a_drift_free_stream_is_clean() -> None:
    for w in enumerate_windows(14_000, 4000, 2500, 500):
        assert classify_window(w, onset=-1) == "clean"


# --- reference policies ------------------------------------------------------

@pytest.fixture
def stream() -> np.ndarray:
    return np.arange(14_000, dtype=np.float64).reshape(-1, 1)


def test_fixed_policy_always_returns_the_same_reference(stream: np.ndarray) -> None:
    tracker = ReferenceTracker(stream, ReferencePolicy.FIXED, 4000)
    windows = enumerate_windows(14_000, 4000, 2500, 500)
    first = tracker.reference_for(windows[0])
    last = tracker.reference_for(windows[-1])
    np.testing.assert_array_equal(first, last)
    np.testing.assert_array_equal(first, stream[:4000])
    assert not tracker.reference_changes_per_window()


def test_sliding_policy_tracks_the_window(stream: np.ndarray) -> None:
    tracker = ReferenceTracker(stream, ReferencePolicy.SLIDING, 4000)
    windows = enumerate_windows(14_000, 4000, 2500, 500)
    ref = tracker.reference_for(windows[4])
    assert ref.shape[0] == 4000
    # The reference ends exactly where the window begins: no overlap between the two.
    assert ref[-1, 0] == windows[4].start - 1
    assert tracker.reference_changes_per_window()


def test_reset_on_alarm_moves_the_baseline_only_after_an_alarm(stream: np.ndarray) -> None:
    tracker = ReferenceTracker(stream, ReferencePolicy.RESET_ON_ALARM, 4000)
    windows = enumerate_windows(14_000, 4000, 2500, 500)
    before = tracker.reference_for(windows[3])
    np.testing.assert_array_equal(before, stream[:4000])

    tracker.notify_alarm(windows[3])
    after = tracker.reference_for(windows[4])
    assert after[-1, 0] == windows[3].end - 1
    assert not np.array_equal(before, after)


def test_alarms_do_not_move_a_fixed_reference(stream: np.ndarray) -> None:
    tracker = ReferenceTracker(stream, ReferencePolicy.FIXED, 4000)
    windows = enumerate_windows(14_000, 4000, 2500, 500)
    tracker.notify_alarm(windows[2])
    np.testing.assert_array_equal(tracker.reference_for(windows[5]), stream[:4000])


# --- autocorrelation ---------------------------------------------------------

def ar1(n: int, rho: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=np.sqrt(1 - rho**2), size=n)
    out = np.empty(n)
    out[0] = rng.normal()
    for i in range(1, n):
        out[i] = rho * out[i - 1] + noise[i]
    return out.reshape(-1, 1)


def test_independent_data_needs_no_thinning() -> None:
    x = np.random.default_rng(0).normal(size=(5000, 4))
    assert autocorrelation_time(x, threshold=0.3) == 1


def test_a_more_persistent_process_needs_a_longer_step() -> None:
    slow = autocorrelation_time(ar1(20_000, 0.99), threshold=0.3, max_lag=2000)
    fast = autocorrelation_time(ar1(20_000, 0.80), threshold=0.3, max_lag=2000)
    assert slow > fast > 1


def test_ar1_estimate_is_near_the_analytic_lag() -> None:
    # For AR(1), rho^k = 0.3 at k = ln(0.3)/ln(rho).
    rho = 0.9
    expected = np.log(0.3) / np.log(rho)
    got = autocorrelation_time(ar1(50_000, rho, seed=3), threshold=0.3, max_lag=500)
    assert abs(got - expected) < 0.35 * expected


def test_a_constant_series_is_treated_as_independent() -> None:
    assert autocorrelation_time(np.ones((1000, 3)), threshold=0.3) == 1


def test_estimate_is_capped_by_the_available_history() -> None:
    x = ar1(400, 0.999)
    assert autocorrelation_time(x, max_lag=10_000, threshold=0.3) <= 400 // 4


def test_a_one_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-D"):
        autocorrelation_time(np.zeros(100))


def test_thinning_keeps_every_step_th_row() -> None:
    x = np.arange(100).reshape(-1, 1)
    np.testing.assert_array_equal(thin(x, 10)[:, 0], np.arange(0, 100, 10))
    np.testing.assert_array_equal(thin(x, 1), x)
    np.testing.assert_array_equal(thin(x, 0), x)  # a step below 1 is a no-op, not an error


def test_effective_sample_size_divides_by_tau() -> None:
    assert effective_sample_size(3000, 150) == 20
    assert effective_sample_size(3000, 0) == 3000
