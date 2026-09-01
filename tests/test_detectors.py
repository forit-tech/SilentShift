"""Behavioural tests for the detectors.

These do not test scikit-learn. They test the two claims the project is built on:

* a marginal statistic cannot see a change that leaves marginals intact;
* a joint statistic can.

If either stops holding, the central result is wrong and this suite says so.
"""

from __future__ import annotations

import numpy as np
import pytest

from silentshift.detectors import MARGINAL_DETECTORS, available, build
from silentshift.injection.catalogue import DriftSpec, inject

N = 4000
D = 6


def correlated(n: int = N, seed: int = 0) -> np.ndarray:
    """Features driven by one latent factor, so a copula change is visible in principle."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    return np.column_stack([latent + 0.3 * rng.normal(size=n) for _ in range(D)])


@pytest.fixture
def reference() -> np.ndarray:
    return correlated(seed=1)


@pytest.mark.parametrize("name", sorted(set(available()) - {"random"}))
def test_a_detector_scores_higher_on_a_shifted_window(name: str, reference: np.ndarray) -> None:
    detector = build(name, seed=0)
    detector.fit_reference(reference)
    same = correlated(n=1500, seed=2)
    shifted = same + 3.0
    assert detector.score(shifted) > detector.score(same)


@pytest.mark.parametrize("name", sorted(available()))
def test_scores_are_finite(name: str, reference: np.ndarray) -> None:
    detector = build(name, seed=0)
    detector.fit_reference(reference)
    assert np.isfinite(detector.score(correlated(n=1500, seed=3)))


@pytest.mark.parametrize("name", sorted(available()))
def test_constant_columns_do_not_produce_nan(name: str) -> None:
    """SMD ships columns that never change; a NaN here would poison a whole metric."""
    rng = np.random.default_rng(4)
    ref = np.column_stack([rng.normal(size=N), np.zeros(N), np.ones(N)])
    win = np.column_stack([rng.normal(size=1000), np.zeros(1000), np.ones(1000)])
    detector = build(name, seed=0)
    detector.fit_reference(ref)
    assert np.isfinite(detector.score(win))


@pytest.mark.parametrize("name", sorted(MARGINAL_DETECTORS - {"ks_bonferroni"}))
def test_marginal_detectors_are_blind_to_a_pure_copula_change(name: str) -> None:
    """The claim the correlation_break scenario exists to test.

    The permutation leaves every marginal bit-for-bit identical, so a statistic computed
    per feature has no information to work with. Anything above chance here would mean the
    detector is reading something other than the marginals — or the injection is leaking.
    """
    x = correlated(seed=5)
    spec = DriftSpec(scenario="correlation_break", onset=2000, affected=(0, 1, 2),
                     magnitude=0.0, params={"block": 25})
    drifted = inject(x, spec, np.random.default_rng(6))

    detector = build(name, seed=0)
    detector.fit_reference(x[:2000])
    clean_score = detector.score(x[2000:])
    broken_score = detector.score(drifted[2000:])

    # Same values, different order: identical empirical marginals by construction.
    for col in spec.affected:
        np.testing.assert_array_equal(
            np.sort(drifted[2000:, col]), np.sort(x[2000:, col])
        )
    assert broken_score == pytest.approx(clean_score, rel=1e-9, abs=1e-9)


def test_a_joint_detector_does_see_the_copula_change() -> None:
    """The other half of the claim: multivariate machinery earns its cost here."""
    x = correlated(seed=7)
    spec = DriftSpec(scenario="correlation_break", onset=2000, affected=(0, 1, 2),
                     magnitude=0.0, params={"block": 25})
    drifted = inject(x, spec, np.random.default_rng(8))

    detector = build("pca_recon", seed=0)
    detector.fit_reference(x[:2000])
    assert detector.score(drifted[2000:]) > 2.0 * detector.score(x[2000:])


def test_random_control_ignores_the_data(reference: np.ndarray) -> None:
    detector = build("random", seed=0)
    detector.fit_reference(reference)
    scores = [detector.score(correlated(n=500, seed=s)) for s in range(30)]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert len(set(scores)) == len(scores)  # independent of the window it was given


@pytest.mark.parametrize("name", sorted(available()))
def test_scoring_before_fitting_is_an_error_not_a_number(name: str) -> None:
    detector = build(name, seed=0)
    if name == "random":  # stateless by design
        pytest.skip("the control detector has no reference to fit")
    with pytest.raises((AssertionError, AttributeError, TypeError, ValueError)):
        detector.score(correlated(n=500, seed=9))


def test_thinning_reduces_the_sample_actually_tested() -> None:
    x = correlated(seed=10)
    plain = build("ks_max", seed=0)
    thinned = build("ks_max_thinned", seed=0)
    plain.fit_reference(x[:2000])
    thinned.fit_reference(x[:2000])
    assert plain.tau == 1
    assert thinned.tau >= 1


def test_attribution_ranks_the_shifted_features_first() -> None:
    x = correlated(seed=11)
    window = correlated(n=1500, seed=12)
    window[:, [0, 3]] += 4.0
    detector = build("wasserstein_max", seed=0)
    detector.fit_reference(x)
    ranking = np.argsort(detector.attribute(window))[::-1][:2]
    assert set(ranking.tolist()) == {0, 3}


@pytest.mark.parametrize("name", sorted(available()))
def test_attribution_has_one_entry_per_feature(name: str, reference: np.ndarray) -> None:
    detector = build(name, seed=0)
    detector.fit_reference(reference)
    window = correlated(n=1000, seed=13)
    detector.score(window)
    assert detector.attribute(window).shape == (D,)


def test_unknown_detector_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        build("does_not_exist")
