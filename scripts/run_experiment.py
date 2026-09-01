"""Run the SilentShift experiment end to end.

    python scripts/run_experiment.py --stage calibrate
    python scripts/run_experiment.py --stage main --split development
    python scripts/run_experiment.py --stage policy
    python scripts/run_experiment.py --stage attribution

Stages are separate commands on purpose. Calibration must finish, and its thresholds must be
frozen, before any evaluation stage runs — running them together invites the exact mistake
the split exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from silentshift.config import ExperimentConfig, load_config  # noqa: E402
from silentshift.data.smd import dataset_digest, load_machine  # noqa: E402
from silentshift.evaluation.metrics import (  # noqa: E402
    attribution_scores,
    calibrate_thresholds,
)
from silentshift.experiment import (  # noqa: E402
    attribution_for_stream,
    iter_streams,
    score_stream,
)
from silentshift.injection.catalogue import ScenarioName  # noqa: E402
from silentshift.windows.policy import ReferencePolicy  # noqa: E402

log = logging.getLogger("silentshift.run")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path.relative_to(ROOT), len(df))


def stage_calibrate(cfg: ExperimentConfig) -> None:
    """Freeze one threshold per (machine, detector) on that machine's own clean history.

    Runs over *every* machine, not just the calibration split, because the threshold is part
    of how the method is deployed rather than a hyper-parameter being tuned. The machine
    split governs design decisions; this governs operating point. The two are kept apart by
    time: calibration reads the first 35% of a machine's clean history and never touches the
    65% the evaluation streams are drawn from.
    """
    length = int(cfg.extra["calibration_stream_length"])
    seeds = list(cfg.extra["calibration_seeds"])
    machines = (
        list(cfg.data.calibration_machines)
        + list(cfg.data.development_machines)
        + list(cfg.data.heldout_machines)
    )
    scenarios: list[tuple[ScenarioName, float]] = [("none", 0.0)]
    frames = []
    t0 = time.time()
    for stream in iter_streams(
        cfg, "calibration", length, scenarios, machines=machines, part="calibration", seeds=seeds
    ):
        for detector in cfg.detectors:
            frames.append(score_stream(stream, detector, ReferencePolicy.FIXED, cfg))
    scores = pd.concat(frames, ignore_index=True)
    _write(scores, cfg.artifacts_dir / "calibration_scores.csv")

    thresholds = calibrate_thresholds(scores, cfg.evaluation.target_false_alarms_per_window)
    _write(thresholds, cfg.artifacts_dir / "thresholds.csv")
    log.info("calibration finished in %.1fs", time.time() - t0)


def stage_main(cfg: ExperimentConfig, split: str) -> None:
    length = int(cfg.extra["stream_length"])
    frames = []
    t0 = time.time()
    n = 0
    for stream in iter_streams(cfg, split, length):
        for detector in cfg.detectors:
            frames.append(score_stream(stream, detector, ReferencePolicy.FIXED, cfg))
        n += 1
        if n % 25 == 0:
            log.info("%s: %d streams scored (%.0fs)", split, n, time.time() - t0)
    scores = pd.concat(frames, ignore_index=True)
    _write(scores, cfg.artifacts_dir / f"scores_{split}.csv")
    log.info("main stage (%s) finished in %.1fs over %d streams", split, time.time() - t0, n)


def stage_policy(cfg: ExperimentConfig) -> None:
    """Reference-window policy study: policy x drift type, on a narrow detector set."""
    length = int(cfg.extra["stream_length"])
    magnitude = float(cfg.extra["policy_study_magnitude"])
    detectors = list(cfg.extra["policy_study_detectors"])
    scenarios: list[tuple[ScenarioName, float]] = []
    for scenario in cfg.injection.scenarios:
        scenarios.append((scenario, 0.0 if scenario in {"none", "correlation_break"} else magnitude))

    frames = []
    t0 = time.time()
    for stream in iter_streams(cfg, "development", length, scenarios):
        for policy in ReferencePolicy:
            for detector in detectors:
                frames.append(score_stream(stream, detector, policy, cfg))
    scores = pd.concat(frames, ignore_index=True)
    _write(scores, cfg.artifacts_dir / "policy_scores.csv")
    log.info("policy stage finished in %.1fs", time.time() - t0)


def stage_attribution(cfg: ExperimentConfig) -> None:
    """Score per-feature attribution against injected ground truth."""
    length = int(cfg.extra["stream_length"])
    detectors = ["ks_max", "psi_max", "wasserstein_max", "js_max", "c2st", "pca_recon"]
    scenarios: list[tuple[ScenarioName, float]] = []
    for scenario in cfg.injection.scenarios:
        if scenario == "none":
            continue
        if scenario == "correlation_break":
            scenarios.append((scenario, 0.0))
        else:
            scenarios.extend((scenario, m) for m in (1.0, 4.0))

    rows: list[dict[str, object]] = []
    for stream in iter_streams(cfg, "development", length, scenarios):
        # Attribute at the first fully post-onset window: the earliest point where an
        # operator would actually be reading the explanation.
        first_post = (stream.spec.onset - cfg.window.reference_size) // cfg.window.stride + 1
        for detector in detectors:
            vector = attribution_for_stream(stream, detector, cfg, max(0, first_post))
            if vector is None:
                continue
            scored = attribution_scores(vector, stream.spec.affected)
            scored["n_features"] = float(stream.data.shape[1])
            rows.append(
                {
                    "stream_id": stream.stream_id,
                    "scenario": stream.spec.scenario,
                    "magnitude": stream.spec.magnitude,
                    "detector": detector,
                    **scored,
                }
            )
    _write(pd.DataFrame(rows), cfg.artifacts_dir / "attribution.csv")


def stage_negative_control(cfg: ExperimentConfig) -> None:
    """Do the detectors simply re-find SMD's labelled point anomalies?

    Runs the detectors over the *test* half, which contains real labelled anomalies and no
    injected drift, and records how often an alarm coincides with a labelled segment. A
    detector that fires mostly on those segments is doing anomaly detection, not drift
    detection, and the distinction is the premise of the whole project.
    """
    detectors = ["ks_max", "c2st", "pca_recon", "iforest_mean"]
    rows: list[dict[str, object]] = []
    for name in cfg.data.machines_for("development"):
        machine = load_machine(cfg.data.smd_root, name)
        length = min(int(cfg.extra["stream_length"]), machine.test.shape[0])
        data = machine.test[:length]
        labels = machine.test_labels[:length]
        from silentshift.detectors import build as build_detector
        from silentshift.windows.policy import enumerate_windows

        windows = enumerate_windows(
            length, cfg.window.reference_size, cfg.window.size, cfg.window.stride
        )
        for detector_name in detectors:
            detector = build_detector(detector_name, seed=0)
            detector.fit_reference(machine.train[: cfg.window.reference_size])
            for window in windows:
                chunk = data[window.start : window.end]
                anomaly_fraction = float(np.mean(labels[window.start : window.end]))
                rows.append(
                    {
                        "machine": name,
                        "detector": detector_name,
                        "window": window.index,
                        "score": detector.score(chunk),
                        "anomaly_fraction": anomaly_fraction,
                    }
                )
    _write(pd.DataFrame(rows), cfg.artifacts_dir / "negative_control.csv")


def stage_provenance(cfg: ExperimentConfig) -> None:
    record = {
        "smd_root": str(cfg.data.smd_root),
        "dataset_sha256": dataset_digest(cfg.data.smd_root),
        "n_machines": len(list(cfg.data.smd_root.glob("train/machine-*.txt"))),
        "config": {
            "window": cfg.window.__dict__,
            "injection": {k: list(v) if isinstance(v, tuple) else v
                          for k, v in cfg.injection.__dict__.items()},
            "evaluation": cfg.evaluation.__dict__,
            "detectors": list(cfg.detectors),
        },
    }
    path = cfg.artifacts_dir / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", path.relative_to(ROOT))


STAGES = {
    "calibrate": lambda cfg, args: stage_calibrate(cfg),
    "main": lambda cfg, args: stage_main(cfg, args.split),
    "policy": lambda cfg, args: stage_policy(cfg),
    "attribution": lambda cfg, args: stage_attribution(cfg),
    "negative_control": lambda cfg, args: stage_negative_control(cfg),
    "provenance": lambda cfg, args: stage_provenance(cfg),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--split", default="development", choices=["development", "heldout"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    STAGES[args.stage](cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
