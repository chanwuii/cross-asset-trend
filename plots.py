"""Charts saved as PNG files in output/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from metrics import drawdown_series


def plot_equity(curves: dict[str, pd.Series], path: str, split_date: str | None = None) -> None:
    """Growth of $1 for each strategy, log scale (equal % moves look equal)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, ret in curves.items():
        ax.plot((1 + ret.fillna(0)).cumprod(), label=name, linewidth=1.4)
    if split_date:
        ax.axvline(pd.Timestamp(split_date), color="grey", linestyle="--", linewidth=1)
        ax.text(pd.Timestamp(split_date), ax.get_ylim()[1], "  out-of-sample →", va="top", color="grey")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (log scale, after costs)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_drawdown(curves: dict[str, pd.Series], path: str) -> None:
    """How far below its previous peak each strategy was on every day."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, ret in curves.items():
        ax.plot(drawdown_series(ret) * 100, label=name, linewidth=1.1)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_weights(weights: pd.DataFrame, path: str) -> None:
    """Stacked area: what the strategy held over time (white space = cash)."""
    monthly = weights.resample("ME").last()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(monthly.index, [monthly[c] for c in monthly.columns], labels=monthly.columns, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title("Portfolio weights over time (rest = cash / T-bills)")
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_speed_contribution(pieces: dict[int, pd.DataFrame], path: str) -> None:
    """How much of the ensemble's total exposure came from each speed (stacked)."""
    gross = pd.DataFrame({f"{m}m": t.sum(axis=1) for m, t in pieces.items()})
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(gross.index, [gross[c] for c in gross.columns], labels=gross.columns, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title("Ensemble exposure by trend speed (each speed max 25%; rest = cash)")
    ax.legend(loc="upper left", ncol=4)
    ax.grid(alpha=0.3)
    _save(fig, path)


def _save(fig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
