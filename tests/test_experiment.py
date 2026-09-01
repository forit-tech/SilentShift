"""Stream construction, seeding and the guarantees the evaluation depends on."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np

from silentshift.experiment import MAGNITUDE_FREE, stream_seed


def test_stream_seed_is_deterministic_within_a_process() -> None:
    assert stream_seed("machine-1-1|sudden_shift|2|0|evaluation") == stream_seed(
        "machine-1-1|sudden_shift|2|0|evaluation"
    )


def test_different_streams_get_different_seeds() -> None:
    seeds = {
        stream_seed(f"machine-1-1|sudden_shift|{m}|0|evaluation") for m in (0.5, 1.0, 2.0, 4.0)
    }
    assert len(seeds) == 4


def test_stream_seed_is_stable_across_processes() -> None:
    """Regression test for REVIEW.md R1-1.

    `hash()` on a str is randomised per interpreter unless PYTHONHASHSEED is pinned. Seeding a
    stream with it made the choice of clean slice differ between runs, so `make all` did not
    reproduce. Two fresh interpreters must agree.
    """
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, "src")
        from silentshift.experiment import stream_seed
        print(stream_seed("machine-2-3|gradual_shift|1|7|evaluation"))
        """
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1, f"seed differed between processes: {runs}"


def test_seed_stays_inside_the_numpy_range() -> None:
    for i in range(200):
        value = stream_seed(f"machine-{i}|sudden_shift|2|0|evaluation")
        assert 0 <= value < 2**31
        np.random.default_rng(value)  # must be accepted as a seed


def test_scenarios_without_a_magnitude_are_declared() -> None:
    # These collapse to a single grid entry; duplicating them across magnitudes would produce
    # identical streams under different labels and silently inflate every sample size.
    assert frozenset({"none", "correlation_break"}) == MAGNITUDE_FREE
