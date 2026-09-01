"""The injection framework is the ground truth generator.

If it is wrong, every metric in the project is wrong in a way no downstream check would
catch, so these are the tests that matter most.
"""

from __future__ import annotations

import numpy as np
import pytest

from silentshift.injection.catalogue import (
    DriftSpec,
    _block_permutation,
    choose_features,
    inject,
    make_spec,
)


@pytest.fixture
def base() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(2000, 8))


def spec(scenario: str, magnitude: float = 2.0, **params: float) -> DriftSpec:
    return DriftSpec(
        scenario=scenario,  # type: ignore[arg-type]
        onset=1000,
        affected=(1, 3, 5),
        magnitude=magnitude,
        params=params,
    )


def test_drift_free_scenario_is_an_exact_copy(base: np.ndarray) -> None:
    out = inject(base, DriftSpec(scenario="none", onset=-1, affected=(), magnitude=0.0),
                 np.random.default_rng(0))
    np.testing.assert_array_equal(out, base)
    assert out is not base  # a copy, so callers cannot mutate the substrate


def test_untouched_features_are_never_modified(base: np.ndarray) -> None:
    for scenario in ("sudden_shift", "gradual_shift", "incremental_shift",
                     "variance_shift", "correlation_break"):
        out = inject(base, spec(scenario, block=50, ramp=500), np.random.default_rng(1))
        untouched = [c for c in range(base.shape[1]) if c not in (1, 3, 5)]
        np.testing.assert_array_equal(out[:, untouched], base[:, untouched])


def test_pre_onset_rows_are_never_modified(base: np.ndarray) -> None:
    for scenario in ("sudden_shift", "gradual_shift", "incremental_shift",
                     "variance_shift", "correlation_break"):
        out = inject(base, spec(scenario, block=50, ramp=500), np.random.default_rng(2))
        np.testing.assert_array_equal(out[:1000], base[:1000])


def test_sudden_shift_moves_the_mean_by_the_requested_number_of_sigmas(base: np.ndarray) -> None:
    s = spec("sudden_shift", magnitude=3.0)
    out = inject(base, s, np.random.default_rng(3))
    sigma = base[:1000, list(s.affected)].std(axis=0)
    delta = out[1000:, list(s.affected)].mean(axis=0) - base[1000:, list(s.affected)].mean(axis=0)
    np.testing.assert_allclose(delta / sigma, 3.0, rtol=1e-9)


def test_incremental_shift_ends_at_full_magnitude_and_starts_at_zero(base: np.ndarray) -> None:
    s = spec("incremental_shift", magnitude=4.0)
    out = inject(base, s, np.random.default_rng(4))
    cols = list(s.affected)
    diff = out[1000:, cols] - base[1000:, cols]
    np.testing.assert_allclose(diff[0], 0.0, atol=1e-12)
    sigma = base[:1000, cols].std(axis=0)
    np.testing.assert_allclose(diff[-1] / sigma, 4.0, rtol=1e-9)
    # Monotone by construction; a non-monotone ramp would be a different drift type.
    assert np.all(np.diff(diff[:, 0]) >= -1e-12)


def test_gradual_shift_switches_a_growing_fraction_of_rows(base: np.ndarray) -> None:
    s = spec("gradual_shift", magnitude=5.0, ramp=500)
    out = inject(base, s, np.random.default_rng(5))
    cols = list(s.affected)
    moved = np.abs(out[1000:, cols[0]] - base[1000:, cols[0]]) > 1e-9
    early = moved[:250].mean()
    late = moved[500:750].mean()
    assert early < late
    assert late == pytest.approx(1.0)  # past the ramp every row is switched


def test_variance_shift_spreads_without_moving_the_mean(base: np.ndarray) -> None:
    s = spec("variance_shift", magnitude=2.0)
    out = inject(base, s, np.random.default_rng(6))
    cols = list(s.affected)
    before_mu = base[:1000, cols].mean(axis=0)
    after_mu = out[1000:, cols].mean(axis=0)
    after_sd = out[1000:, cols].std(axis=0)
    base_sd = base[1000:, cols].std(axis=0)
    # Mean is preserved up to the difference between the two halves of the substrate.
    np.testing.assert_allclose(after_mu, before_mu, atol=0.15)
    np.testing.assert_allclose(after_sd / base_sd, 3.0, rtol=1e-9)


def test_correlation_break_preserves_every_marginal_exactly(base: np.ndarray) -> None:
    """The property the whole 'blind by construction' argument rests on.

    If this ever fails, the marginal detectors could legitimately fire on
    correlation_break and the headline finding of the project would be an artefact.
    """
    s = spec("correlation_break", block=50)
    out = inject(base, s, np.random.default_rng(7))
    for col in s.affected:
        np.testing.assert_array_equal(
            np.sort(out[1000:, col]), np.sort(base[1000:, col])
        )


def test_correlation_break_actually_destroys_cross_correlation(base: np.ndarray) -> None:
    rng = np.random.default_rng(8)
    n = 4000
    shared = rng.normal(size=n)
    x = np.column_stack([shared + 0.1 * rng.normal(size=n) for _ in range(4)])
    s = DriftSpec(scenario="correlation_break", onset=2000, affected=(0, 1),
                  magnitude=0.0, params={"block": 20})
    out = inject(x, s, np.random.default_rng(9))
    before = np.corrcoef(x[2000:, 0], x[2000:, 3])[0, 1]
    after = np.corrcoef(out[2000:, 0], out[2000:, 3])[0, 1]
    assert before > 0.9
    assert after < 0.5


def test_block_permutation_is_a_permutation() -> None:
    idx = _block_permutation(1000, 60, np.random.default_rng(10))
    assert idx.shape == (1000,)
    np.testing.assert_array_equal(np.sort(idx), np.arange(1000))


def test_choose_features_skips_constant_columns() -> None:
    excluded = np.zeros(8, dtype=bool)
    excluded[[0, 2, 4]] = True
    picked = choose_features(8, 5, excluded, np.random.default_rng(11))
    assert set(picked).isdisjoint({0, 2, 4})
    assert len(picked) == 5


def test_choose_features_cannot_exceed_the_eligible_pool() -> None:
    excluded = np.ones(8, dtype=bool)
    excluded[3] = False
    picked = choose_features(8, 6, excluded, np.random.default_rng(12))
    assert picked == (3,)


def test_choose_features_rejects_a_fully_constant_matrix() -> None:
    with pytest.raises(ValueError, match="no non-constant features"):
        choose_features(4, 2, np.ones(4, dtype=bool), np.random.default_rng(13))


def test_make_spec_places_onset_at_the_configured_fraction() -> None:
    s = make_spec(
        "sudden_shift", n_rows=10_000, n_features=8, onset_fraction=0.6, n_affected=3,
        magnitude=1.0, ramp=100, block=10, excluded=np.zeros(8, dtype=bool),
        rng=np.random.default_rng(14),
    )
    assert s.onset == 6000
    assert len(s.affected) == 3
    assert not s.is_drift_free


def test_make_spec_for_none_is_marked_drift_free() -> None:
    s = make_spec(
        "none", n_rows=100, n_features=4, onset_fraction=0.5, n_affected=1, magnitude=1.0,
        ramp=1, block=1, excluded=np.zeros(4, dtype=bool), rng=np.random.default_rng(15),
    )
    assert s.is_drift_free
    assert s.onset == -1
    assert s.affected == ()


def test_injection_is_reproducible_for_a_fixed_seed(base: np.ndarray) -> None:
    s = spec("gradual_shift", ramp=400)
    a = inject(base, s, np.random.default_rng(99))
    b = inject(base, s, np.random.default_rng(99))
    np.testing.assert_array_equal(a, b)
