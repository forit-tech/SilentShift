"""Turn raw window scores into the tables and figures the README reports.

    python scripts/analyse.py --split development

Reads only artefacts produced by run_experiment.py, so nothing here can accidentally
recompute a score with a different setting than the one that was evaluated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from silentshift.config import load_config  # noqa: E402
from silentshift.evaluation.discriminability import (  # noqa: E402
    calibration_transfer,
    matched_far_recall,
    window_auc,
)
from silentshift.evaluation.metrics import (  # noqa: E402
    detectability_table,
    intervals_overlap,
    summarise_streams,
)
from silentshift.reporting import plots  # noqa: E402

log = logging.getLogger("silentshift.analyse")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path.name, len(df))


def superiority_report(table: pd.DataFrame, baseline: str = "ks_max") -> pd.DataFrame:
    """Which detectors beat the baseline with non-overlapping bootstrap intervals.

    Overlapping intervals produce `inconclusive`, not a winner. Refusing to rank on
    differences the data cannot support is the whole point of computing the intervals.
    """
    rows: list[dict[str, object]] = []
    for (scenario, magnitude), group in table.groupby(["scenario", "magnitude"]):
        base = group[group["detector"] == baseline]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in group.iterrows():
            if r["detector"] == baseline:
                continue
            overlap = intervals_overlap(
                r["recall_lo"], r["recall_hi"], b["recall_lo"], b["recall_hi"]
            )
            if overlap:
                verdict = "inconclusive"
            else:
                verdict = "better" if r["recall"] > b["recall"] else "worse"
            rows.append(
                {
                    "scenario": scenario,
                    "magnitude": magnitude,
                    "detector": r["detector"],
                    "baseline": baseline,
                    "recall": round(float(r["recall"]), 3),
                    "baseline_recall": round(float(b["recall"]), 3),
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def negative_control_report(path: Path, thresholds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Correlation between a detector's score and the density of labelled point anomalies.

    A high correlation means the detector is re-finding SMD's anomalies rather than
    detecting distributional change — the failure mode the project is defined against.

    The alarm rate is reported next to it because a weak correlation on its own is ambiguous:
    it is equally consistent with "ignores point anomalies" and with "alarms on everything".
    """
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for detector, group in df.groupby("detector"):
        if group["anomaly_fraction"].std() < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(group["score"], group["anomaly_fraction"])[0, 1])
        alarm_rate = float("nan")
        if thresholds is not None:
            key = thresholds.set_index(["machine", "detector", "policy"])["value"].to_dict()
            fired = total = 0
            for machine, sub in group.groupby("machine"):
                t = key.get((machine, detector, "fixed"))
                if t is None:
                    continue
                fired += int(np.count_nonzero(sub["score"].to_numpy() > t))
                total += len(sub)
            if total:
                alarm_rate = fired / total

        rows.append(
            {
                "detector": detector,
                "n_windows": len(group),
                "corr_score_vs_anomaly_density": round(corr, 3),
                "alarm_rate_on_test_half": round(alarm_rate, 3),
                "mean_anomaly_fraction": round(float(group["anomaly_fraction"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--split", default="development")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    cfg = load_config(args.config)
    art = cfg.artifacts_dir
    tol = cfg.evaluation.tolerance_windows
    thresholds = pd.read_csv(art / "thresholds.csv")

    scores = pd.read_csv(art / f"scores_{args.split}.csv")
    summary = summarise_streams(scores, thresholds)
    _write(summary, art / f"summary_{args.split}.csv")

    table = detectability_table(
        summary, tolerance=tol,
        n_bootstrap=cfg.evaluation.bootstrap_samples, seed=cfg.evaluation.bootstrap_seed,
    )
    _write(table, art / f"detectability_{args.split}.csv")
    _write(superiority_report(table), art / f"superiority_{args.split}.csv")

    # False-alarm rate measured on the drift-free streams only.
    drift_free = summary[summary["scenario"] == "none"]
    far = (
        drift_free.groupby("detector")
        .agg(false_alarms_per_stream=("false_alarms", "mean"),
             clean_windows=("clean_windows", "mean"),
             n_streams=("stream_id", "nunique"))
        .reset_index()
    )
    _write(far, art / f"false_alarm_rate_{args.split}.csv")

    # Threshold-free discriminability, and the two diagnostics that explain why the
    # recall-at-calibrated-threshold numbers above cannot be taken at face value.
    auc = window_auc(scores, n_bootstrap=cfg.evaluation.bootstrap_samples, seed=cfg.evaluation.bootstrap_seed)
    _write(auc, art / f"discriminability_{args.split}.csv")

    transfer = calibration_transfer(
        scores, thresholds, cfg.evaluation.target_false_alarms_per_window
    )
    _write(transfer, art / "calibration_transfer.csv")

    # Per-stream operating point over the same horizon the recall uses, so the chance floor
    # equals the target instead of being ~4x higher.
    oracle = matched_far_recall(
        scores, cfg.extra.get("target_stream_false_positive_rate", 0.1), tol,
        n_bootstrap=cfg.evaluation.bootstrap_samples, seed=cfg.evaluation.bootstrap_seed,
    )
    _write(oracle, art / f"oracle_recall_{args.split}.csv")

    nc = negative_control_report(art / "negative_control.csv", thresholds)
    if not nc.empty:
        _write(nc, art / "negative_control_report.csv")

    figs = art / "figures"
    plots.threshold_spread(thresholds, figs / "threshold_spread.png")
    plots.auc_heatmap(auc, figs / "auc_heatmap.png", magnitude=2.0)
    plots.calibration_transfer_plot(transfer, figs / "calibration_transfer.png")
    plots.recall_heatmap(table, figs / "recall_heatmap.png", magnitude=2.0)
    plots.delay_distribution(summary, figs / "delay_sudden_shift.png")
    for scenario in ("sudden_shift", "gradual_shift", "incremental_shift", "variance_shift"):
        if (table["scenario"] == scenario).any():
            plots.detectability_curve(table, scenario, figs / f"detectability_{scenario}.png")

    policy_path = art / "policy_scores.csv"
    if policy_path.exists():
        # Threshold-free, like every other comparison here. Thresholds are calibrated under the
        # fixed policy only, so a recall table would silently drop the other policies rather
        # than compare them.
        policy_scores = pd.read_csv(policy_path)
        frames = []
        for policy, group in policy_scores.groupby("policy"):
            part = window_auc(group, n_bootstrap=cfg.evaluation.bootstrap_samples, seed=cfg.evaluation.bootstrap_seed)
            part["policy"] = policy
            frames.append(part)
        policy_table = pd.concat(frames, ignore_index=True)
        _write(policy_table, art / "policy_auc.csv")

        # `reset_on_alarm` is currently identical to `fixed`: the alarm feedback loop is not
        # wired -- see the README section on defects found in this code. Plotting it as a third row
        # would imply a comparison that was never made.
        exercised = policy_table[policy_table["policy"] != "reset_on_alarm"].rename(
            columns={"auc": "recall"}
        )
        plots.policy_interaction(exercised, figs / "policy_interaction.png")

    log.info("figures written to %s", figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
