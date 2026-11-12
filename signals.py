"""Trend signals: is each asset currently in an up-trend?

Every function here only looks at data *up to and including* row t.
Nothing reads the future. tests/test_no_lookahead.py checks this.
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS_PER_MONTH = 21


def momentum(prices: pd.DataFrame, lookback_months: int, skip_months: int = 0) -> pd.DataFrame:
    """Past return over `lookback_months`, optionally skipping the latest month(s).

    momentum_t = P_{t - skip} / P_{t - lookback} - 1

    Example: lookback 12, skip 1 -> return from 12 months ago to 1 month ago.
    """
    if skip_months >= lookback_months:
        raise ValueError("skip_months must be smaller than lookback_months")
    lb = lookback_months * TRADING_DAYS_PER_MONTH
    sk = skip_months * TRADING_DAYS_PER_MONTH
    return prices.shift(sk) / prices.shift(lb) - 1.0


def sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average of the last `window` closes (NaN until enough data)."""
    return prices.rolling(window=window, min_periods=window).mean()


def trend_signal(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 0,
    sma_window: int = 200,
) -> pd.DataFrame:
    """1.0 if the asset is in an up-trend, else 0.0.

    Up-trend = BOTH conditions true:
      1. momentum over the lookback is positive (the asset went up), AND
      2. today's price is above its `sma_window`-day moving average.

    Requiring both is a simple "two witnesses" rule: fewer false signals,
    at the cost of reacting a bit later.
    """
    mom = momentum(prices, lookback_months, skip_months)
    above_sma = prices > sma(prices, sma_window)
    signal = (mom > 0) & above_sma
    # Where we don't have enough history the comparisons involve NaN and
    # evaluate to False -> the asset is simply not held. That is safe.
    return signal.astype(float)


def your_turn_signal(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """YOUR TURN (optional exercise #1): a moving-average crossover signal.

    Goal: return 1.0 where the `fast`-day SMA is above the `slow`-day SMA,
    else 0.0 - same shape as `prices`.

    Hints:
      - you already have the `sma()` helper above
      - compare two DataFrames with `>` and convert with `.astype(float)`
    When done, set `signal.kind: your_turn` in config.yaml and run run.py.
    """
    raise NotImplementedError("YOUR TURN: implement the moving-average crossover")


def build_signal(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pick which signal to use based on config.yaml -> signal.kind."""
    kind = cfg.get("kind", "default")
    if kind == "default":
        return trend_signal(
            prices,
            lookback_months=cfg["momentum_lookback_months"],
            skip_months=cfg.get("skip_months", 0),
            sma_window=cfg["sma_window"],
        )
    if kind == "your_turn":
        return your_turn_signal(prices)
    raise ValueError(f"unknown signal kind: {kind}")
