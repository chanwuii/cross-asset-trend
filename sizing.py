"""Position sizing: how much of the portfolio goes into each active asset.

Three steps, each easy to explain:
  1. Inverse-volatility: calmer assets get bigger weights, wild ones smaller,
     so each asset contributes roughly similar risk.
  2. Volatility target: scale the whole basket so its expected annual
     volatility is about `target_vol` (e.g. 10%), never using leverage.
  3. Cap: no single asset above `max_weight`. Anything not invested is cash.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def inverse_vol_weights(vols: pd.Series) -> pd.Series:
    """w_i = (1/sigma_i) / sum_j (1/sigma_j). Assets with NaN/0 vol get 0."""
    inv = 1.0 / vols.replace(0.0, np.nan)
    inv = inv.fillna(0.0)
    total = inv.sum()
    if total <= 0:
        return inv * 0.0
    return inv / total


def portfolio_vol(weights: pd.Series, cov_annual: pd.DataFrame) -> float:
    """Annualized portfolio volatility sigma_p = sqrt(w^T * Sigma * w)."""
    w = weights.reindex(cov_annual.index).fillna(0.0).to_numpy()
    variance = float(w @ cov_annual.to_numpy() @ w)
    return math.sqrt(max(variance, 0.0))


def target_weights(
    signal_row: pd.Series,
    recent_returns: pd.DataFrame,
    target_vol: float = 0.10,
    max_weight: float = 0.30,
    max_gross: float = 1.0,
    multipliers: pd.Series | None = None,
) -> pd.Series:
    """Turn one day's signals into portfolio weights (cash = 1 - sum).

    Parameters
    ----------
    signal_row : 1.0/0.0 per asset for this rebalance date.
    recent_returns : the last N days of daily returns (rows) per asset (cols),
        ending ON the rebalance date - never after it.
    multipliers : optional per-asset tilt (e.g. 1.25 or 0.75) from the
        Google Trends overlay. None = no tilt (Stage 1).
    """
    assets = signal_row.index
    weights = pd.Series(0.0, index=assets)

    active = [a for a in assets if signal_row.get(a, 0.0) > 0]
    if not active:
        return weights  # everything in cash

    window = recent_returns[active].dropna()
    if len(window) < 20:  # not enough data to estimate risk
        return weights

    daily_vol = window.std(ddof=1)
    w = inverse_vol_weights(daily_vol)

    if multipliers is not None:
        w = w * multipliers.reindex(w.index).fillna(1.0)
        w = w / w.sum()

    cov_annual = window.cov() * TRADING_DAYS_PER_YEAR
    sigma_p = portfolio_vol(w, cov_annual)
    if sigma_p <= 0:
        return weights

    # Scale up/down toward the vol target, but never above max_gross (no leverage).
    scale = min(target_vol / sigma_p, max_gross / w.sum())
    w = w * scale

    # Cap each position. The trimmed amount simply stays in cash.
    w = w.clip(upper=max_weight)

    weights.loc[w.index] = w
    return weights
