"""Figures.

Four plots, each answering one question. There is no gallery of decorative charts here:
a figure that does not change what the reader believes is noise with axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGSIZE = (9, 5.5)
DPI = 140


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def detectability_curve(table: pd.DataFrame, scenario: str, out: Path) -> Path:
    """Recall against injected magnitude — how big must a change be before this is noticed."""
    sub = table[table["scenario"] == scenario].sort_values("magnitude")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for detector, group in sub.groupby("detector"):
        ax.plot(group["magnitude"], group["recall"], marker="o", label=detector, linewidth=1.6)
        ax.fill_between(group["magnitude"], group["recall_lo"], group["recall_hi"], alpha=0.12)
    ax.set_xlabel("injected shift (multiples of pre-onset per-feature sigma)")
    ax.set_ylabel("recall within tolerance\n(fraction of streams)")
    ax.set_title(f"Detectability: {scenario}")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    return _finish(fig, out)


def recall_heatmap(table: pd.DataFrame, out: Path, magnitude: float | None = None) -> Path:
    """Detector x scenario recall. The correlation_break column is the point of the figure."""
    sub = table if magnitude is None else table[
        (table["magnitude"] == magnitude) | (table["scenario"] == "correlation_break")
    ]
    grid = sub.pivot_table(index="detector", columns="scenario", values="recall", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.4 * len(grid.columns) + 4, 0.42 * len(grid) + 2.5))
    im = ax.imshow(grid.to_numpy(), cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.to_numpy()[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value < 0.6 else "black")
    ax.set_title("Recall by detector and drift type" +
                 ("" if magnitude is None else f" (shift = {magnitude:g} sigma)"))
    fig.colorbar(im, ax=ax, label="recall", fraction=0.03)
    return _finish(fig, out)


def delay_distribution(summary: pd.DataFrame, out: Path, scenario: str = "sudden_shift") -> Path:
    """Detection delay is right-skewed and censored, so it gets a distribution, not a mean."""
    sub = summary[(summary["scenario"] == scenario) & np.isfinite(summary["delay"])]
    order = sub.groupby("detector")["delay"].median().sort_values().index
    data = [sub.loc[sub["detector"] == d, "delay"].to_numpy() for d in order]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(data, tick_labels=list(order), vert=False, widths=0.6, showfliers=False)
    ax.set_xlabel("detection delay (windows after the first fully post-onset window)")
    ax.set_title(f"Detection delay: {scenario} (detected streams only)")
    ax.grid(alpha=0.25, axis="x", linewidth=0.6)
    return _finish(fig, out)


def policy_interaction(table: pd.DataFrame, out: Path) -> Path:
    """Reference policy against drift type: the trade the project exists to measure."""
    grid = table.pivot_table(index="policy", columns="scenario", values="recall", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.5 * len(grid.columns) + 3.5, 3.2))
    im = ax.imshow(grid.to_numpy(), cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=9)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.to_numpy()[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if value < 0.55 else "black")
    ax.set_title("Discriminability (AUC) by reference-window policy and drift type")
    fig.colorbar(im, ax=ax, label="AUC", fraction=0.04)
    return _finish(fig, out)


def threshold_spread(thresholds: pd.DataFrame, out: Path) -> Path:
    """Why thresholds are per machine: the null level spans decades across the fleet."""
    sub = thresholds[thresholds["value"] > 0].copy()
    order = sub.groupby("detector")["value"].median().sort_values().index
    data = [sub.loc[sub["detector"] == d, "value"].to_numpy() for d in order]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.boxplot(data, tick_labels=list(order), vert=False, widths=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("calibrated threshold (log scale), one point per machine")
    ax.set_title("A single global threshold is not available:\nthe drift-free null differs by orders of magnitude between machines")
    ax.grid(alpha=0.25, axis="x", which="both", linewidth=0.6)
    return _finish(fig, out)


def auc_heatmap(table: pd.DataFrame, out: Path, magnitude: float = 2.0) -> Path:
    """Threshold-free separability. The `correlation_break` column is the result to read.

    A data-independent control sits at 0.5 by construction, so any cell near 0.5 means the
    statistic carries no information about that drift type -- not that the drift was small.
    """
    sub = table[(table["magnitude"] == magnitude) | (table["scenario"] == "correlation_break")]
    grid = sub.pivot_table(index="detector", columns="scenario", values="auc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.5 * len(grid.columns) + 4, 0.42 * len(grid) + 2.6))
    im = ax.imshow(grid.to_numpy(), cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title(
        f"Window-level AUC, drifted vs drift-free (shift = {magnitude:g} sigma)\n"
        "0.50 = no information; the random control measures 0.43-0.52"
    )
    fig.colorbar(im, ax=ax, label="AUC", fraction=0.03)
    return _finish(fig, out)


def calibration_transfer_plot(transfer: pd.DataFrame, out: Path) -> Path:
    """How badly a frozen threshold overshoots its false-alarm budget on later clean data."""
    sub = transfer.sort_values("inflation_factor")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colours = ["tab:grey" if d == "random" else "tab:red" for d in sub["detector"]]
    ax.barh(sub["detector"], sub["inflation_factor"], color=colours)
    ax.axvline(1.0, color="black", linewidth=1.2, label="budget met")
    control = sub.loc[sub["detector"] == "random", "inflation_factor"]
    if not control.empty:
        ax.axvline(float(control.iloc[0]), color="tab:grey", linestyle="--", linewidth=1.2,
                   label="control (estimator noise only)")
    ax.set_xlabel("achieved false alarms / target false alarms")
    ax.set_title("A threshold calibrated on earlier clean history does not hold later\n"
                 "(same machine, disjoint time periods)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, axis="x", linewidth=0.6)
    return _finish(fig, out)
