"""v2 core: a multi-speed trend ensemble.

Instead of betting everything on ONE lookback (12 months), we run four
small trend strategies at different speeds and average their portfolios:

    speed 1m  -> fast, reacts quickly, more false alarms
    speed 3m
    speed 6m
    speed 12m -> slow, steady, reacts late

Each speed is a complete long/cash strategy (time-series momentum signal
-> the same inverse-vol sizing + 10% vol target + 30% cap as v1). The
ensemble holds the simple AVERAGE of the four target portfolios:

    w_ensemble = (w_1m + w_3m + w_6m + w_12m) / 4

Why this way?
  * Equal weights (1/4 each) are fixed in advance - nothing is tuned.
  * If only 1 of 4 speeds likes an asset, it gets ~1/4 of a full position:
    conviction is built in.
  * Every speed portfolio respects the caps and has gross <= 100%, so their
    average does too (an average can't exceed its largest member).

Skip-month choice: we do NOT skip the most recent month (skip = 0). A
1-month lookback with a 1-month skip would be empty, and using the same
rule at every speed keeps the ensemble simple to explain.
"""

from __future__ import annotations

import pandas as pd

from backtest import compute_targets
from signals import momentum


def tsmom_signal(prices: pd.DataFrame, months: int, skip_months: int = 0) -> pd.DataFrame:
    """Time-series momentum: 1.0 if the asset's own return over `months` is > 0, else 0.0."""
    return (momentum(prices, months, skip_months) > 0).astype(float)


def ensemble_score(prices: pd.DataFrame, speeds: list[int], skip_months: int = 0) -> pd.DataFrame:
    """Share of speeds that are long, per asset and day: 0, 0.25, 0.5, 0.75 or 1.0 (for 4 speeds)."""
    signals = [tsmom_signal(prices, m, skip_months) for m in speeds]
    return sum(signals) / len(signals)


def ensemble_targets(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    sizing_cfg: dict,
    speeds: list[int],
    skip_months: int = 0,
    multipliers: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Average of one target-weight table per speed.

    Returns (ensemble_targets, {speed: that speed's targets / len(speeds)}).
    The second item shows how much of each weight came from each speed;
    the pieces add up exactly to the ensemble.
    """
    pieces: dict[int, pd.DataFrame] = {}
    for m in speeds:
        sig = tsmom_signal(prices, m, skip_months)
        t = compute_targets(returns, sig, signal_dates, sizing_cfg, multipliers=multipliers)
        pieces[m] = t / len(speeds)
    total = sum(pieces.values())
    return total, pieces


def your_turn_inverse_speed_weights(speeds: list[int]) -> dict[int, float]:
    """YOUR TURN (optional exercise #6 in the study guide): unequal speed weights, fixed IN ADVANCE.

    Example idea: give slower speeds more weight because they trade less,
    e.g. weight proportional to sqrt(months), normalised to sum to 1.
    Return {speed: weight}. Then change `ensemble_targets` to use it and
    compare turnover and costs. (Don't pick weights by looking at Sharpe!)
    """
    raise NotImplementedError("YOUR TURN: choose fixed speed weights")
