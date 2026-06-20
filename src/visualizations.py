"""Static publication-quality plots (not the dashboard)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from config import PLOTS_DIR

sns.set_theme(style="whitegrid", palette="muted")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_drift_signals(drift_df: pd.DataFrame) -> Path:
    methods = drift_df["method"].unique()
    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 4 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        sub = drift_df[drift_df["method"] == method]
        ax.plot(sub["batch_id"], sub["score"], linewidth=1.5, label=method)
        drift_pts = sub[sub["is_drift"]]
        ax.scatter(drift_pts["batch_id"], drift_pts["score"],
                   color="red", zorder=5, s=60, label="Drift detected")
        ax.set_ylabel("Score")
        ax.set_title(method, fontsize=12)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Batch ID")
    fig.suptitle("Drift Detection Signals Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = PLOTS_DIR / "drift_signals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_strategy_f1(per_batch_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5))
    for strategy, grp in per_batch_df.groupby("strategy"):
        ax.plot(grp["batch_id"], grp["f1"], label=strategy, linewidth=1.4)

    drift_batches = per_batch_df[per_batch_df["drift_detected"]]["batch_id"].unique()
    for b in drift_batches:
        ax.axvline(b, color="red", alpha=0.15, linewidth=0.8)

    ax.set_xlabel("Batch ID")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score per Strategy Over Time (red = drift event)", fontsize=13)
    ax.legend()
    plt.tight_layout()
    path = PLOTS_DIR / "strategy_f1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_strategy_radar(summary_df: pd.DataFrame) -> Path:
    criteria = ["mean_f1", "drift_events", "mean_inference_s", "std_f1"]
    labels = ["Avg F1", "Drift Events Caught", "Inference Speed (inv)", "Stability (inv)"]

    # Normalise 0-1, invert cost criteria
    dm = summary_df[criteria].copy().values.astype(float)
    dm[:, 2] = 1 - (dm[:, 2] - dm[:, 2].min()) / (dm[:, 2].ptp() + 1e-10)  # speed → benefit
    dm[:, 3] = 1 - (dm[:, 3] - dm[:, 3].min()) / (dm[:, 3].ptp() + 1e-10)  # std   → stability
    for i in [0, 1]:
        dm[:, i] = (dm[:, i] - dm[:, i].min()) / (dm[:, i].ptp() + 1e-10)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    palette = ["#e05c5c", "#5c9ee0", "#5ce08a", "#e0c05c", "#c05ce0", "#5ce0d8"]

    for i, (_, row) in enumerate(summary_df.iterrows()):
        color = palette[i % len(palette)]
        vals = dm[i].tolist() + [dm[i][0]]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=row["strategy"])
        ax.fill(angles, vals, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Strategy Comparison — Radar Chart", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    path = PLOTS_DIR / "strategy_radar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_vikor_scores(vikor_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, col, color, title in zip(
        axes,
        ["S", "R", "Q"],
        ["#5c9ee0", "#e05c5c", "#5ce08a"],
        ["S (Group Utility)", "R (Individual Regret)", "Q (VIKOR Score — lower = better)"],
    ):
        ax.bar(vikor_df["strategy"], vikor_df[col], color=color)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("VIKOR Decision Scores by Strategy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = PLOTS_DIR / "vikor_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_window_size_over_time(per_batch_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(14, 4))
    for strategy, grp in per_batch_df.groupby("strategy"):
        ax.plot(grp["batch_id"], grp["window_size"], label=strategy, linewidth=1.4)
    ax.set_xlabel("Batch ID")
    ax.set_ylabel("Retraining Window Size")
    ax.set_title("ADWIN Retraining Window Size Over Time", fontsize=13)
    ax.legend()
    plt.tight_layout()
    path = PLOTS_DIR / "window_sizes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
