"""
plot_utils.py
-------------
Reusable plotting functions for the LoL capstone project.
Consistent style, annotations, and save behaviour across all notebooks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import COLORS, PLOTS_DIR


# ── Style setup ───────────────────────────────────────────────────────────────

def set_style():
    """Apply consistent professional chart style across all notebooks."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "axes.titlepad":      12,
        "axes.labelsize":     11,
        "axes.labelweight":   "bold",
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "legend.framealpha":  0.9,
        "figure.dpi":         120,
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
        "font.family":        "sans-serif",
    })


def save_plot(filename: str, tight: bool = True):
    """Save the current figure to plots/ directory."""
    PLOTS_DIR.mkdir(exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved -> plots/{filename}")


# ── Annotation helpers ────────────────────────────────────────────────────────

def add_significance_stars(ax, p_value: float, x: float, y: float,
                            fontsize: int = 11):
    """Add significance stars annotation to a chart."""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns"
    ax.text(x, y, stars, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color="black")


def add_bar_labels(ax, bars, values, fmt="{:.1f}%", offset=0.3,
                   fontsize=10, fontweight="bold"):
    """Add value labels to the end of horizontal bars."""
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(val),
            va="center", fontsize=fontsize, fontweight=fontweight
        )


def add_confidence_interval(ax, y_pos: float, ci_low: float, ci_high: float,
                             color: str = "black", linewidth: float = 2.0):
    """Add a horizontal confidence interval line to a bar chart."""
    ax.plot([ci_low, ci_high], [y_pos, y_pos],
            color=color, linewidth=linewidth, solid_capstyle="round")
    ax.plot(ci_low,  y_pos, "|", color=color, markersize=8, markeredgewidth=2)
    ax.plot(ci_high, y_pos, "|", color=color, markersize=8, markeredgewidth=2)


# ── Chart builders ────────────────────────────────────────────────────────────

def plot_win_rates_with_ci(data: pd.DataFrame, label_col: str, rate_col: str,
                           ci_low_col: str, ci_high_col: str,
                           title: str, filename: str,
                           baseline: float = 50.0, figsize=(12, 7)):
    """
    Horizontal bar chart of win rates with confidence intervals.
    Colors bars green if above baseline, red if below.
    """
    fig, ax = plt.subplots(figsize=figsize)
    data = data.sort_values(rate_col, ascending=True).reset_index(drop=True)

    colors = [COLORS["green"] if r >= baseline else COLORS["red"]
              for r in data[rate_col]]
    bars = ax.barh(data[label_col], data[rate_col],
                   color=colors, edgecolor="white", height=0.65, alpha=0.85)

    # Confidence intervals
    for i, (_, row) in enumerate(data.iterrows()):
        add_confidence_interval(ax, i, row[ci_low_col], row[ci_high_col])

    # Value labels
    add_bar_labels(ax, bars, data[rate_col])

    ax.axvline(baseline, color=COLORS["gray"], linestyle="--",
               linewidth=1.5, alpha=0.8, label=f"{baseline}% baseline")
    ax.set_xlabel("Win Rate (%)")
    ax.set_title(title)
    ax.legend()

    legend_patches = [
        mpatches.Patch(color=COLORS["green"], label=f"Above {baseline}%"),
        mpatches.Patch(color=COLORS["red"],   label=f"Below {baseline}%"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color=COLORS["gray"], linestyle="--", label=f"{baseline}% baseline")
    ])

    save_plot(filename)
    return fig, ax


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str,
                             filename: str, figsize=(12, 10)):
    """Seaborn correlation heatmap with annotations."""
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title(title)
    save_plot(filename)
    return fig, ax


def plot_roc_curve(fpr, tpr, auc_score: float, model_name: str,
                   filename: str, figsize=(8, 7)):
    """ROC curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color=COLORS["blue"], linewidth=2.5,
            label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--",
            linewidth=1.5, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS["blue"])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    save_plot(filename)
    return fig, ax


def plot_feature_importance(importances: pd.Series, title: str,
                             filename: str, top_n: int = 15,
                             figsize=(12, 7)):
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=figsize)
    top = importances.nlargest(top_n).sort_values(ascending=True)
    colors = [COLORS["green"] if v > importances.median()
              else COLORS["blue"] for v in top.values]
    bars = ax.barh(top.index, top.values, color=colors,
                   edgecolor="white", height=0.65)
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    save_plot(filename)
    return fig, ax
