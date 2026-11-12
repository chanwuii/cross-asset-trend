"""Performance statistics. Pure functions: numbers in, numbers out.

Risk-free rate is assumed to be 0 in Sharpe/Sortino to keep formulas
simple; this is stated in the README.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25


def years_between(index: pd.DatetimeIndex) -> float:
    """Calendar years covered by a date index."""
    if len(index) < 2:
        return float("nan")
    return (index[-1] - index[0]).days / DAYS_PER_YEAR


def cagr_from_returns(returns: pd.Series) -> float:
    """CAGR = (growth of $1) ^ (1 / years) - 1, from daily returns."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    growth = float((1.0 + r).prod())
    yrs = years_between(r.index)
    if growth <= 0 or not yrs > 0:
        return float("nan")
    return growth ** (1.0 / yrs) - 1.0


def annual_vol(returns: pd.Series) -> float:
    """sigma_annual = std(daily returns) * sqrt(252)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe(returns: pd.Series) -> float:
    """Sharpe = mean(daily) / std(daily) * sqrt(252) (risk-free = 0)."""
    r = returns.dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or sd == 0:
        return float("nan")
    return float(r.mean() / sd) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sortino(returns: pd.Series) -> float:
    """Like Sharpe, but only *downside* moves count as risk.

    downside deviation = sqrt(mean(min(r, 0)^2)) * sqrt(252)
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    downside = np.minimum(r.to_numpy(), 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    if dd == 0:
        return float("nan")
    return float(r.mean()) / dd * math.sqrt(TRADING_DAYS_PER_YEAR)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown_t = value_t / (highest value so far) - 1  (0 or negative)."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss, e.g. -0.25 = -25%."""
    if returns.dropna().empty:
        return float("nan")
    return float(drawdown_series(returns).min())


def calmar(returns: pd.Series) -> float:
    """Calmar = CAGR / |max drawdown|. 'How much return per unit of pain?'"""
    mdd = max_drawdown(returns)
    if not mdd < 0:
        return float("nan")
    return cagr_from_returns(returns) / abs(mdd)


def summarize(
    returns: pd.Series,
    weights: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> dict[str, float]:
    """All headline metrics for one return stream (optionally with trading stats)."""
    out = {
        "CAGR": cagr_from_returns(returns),
        "Vol": annual_vol(returns),
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "MaxDD": max_drawdown(returns),
        "Calmar": calmar(returns),
    }
    yrs = years_between(returns.dropna().index)
    if turnover is not None and yrs > 0:
        out["Turnover/yr"] = float(turnover.sum()) / yrs
    if costs is not None and yrs > 0:
        out["Costs/yr"] = float(costs.sum()) / yrs
    if weights is not None and len(weights):
        gross = weights.sum(axis=1)
        out["AvgInvested"] = float(gross.mean())
        out["%DaysInvested"] = float((gross > 1e-6).mean())
    return out


def difference_test(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Is strategy A really different from B? Simple t-test on daily return differences.

    d_t = a_t - b_t
    t-stat = mean(d) / std(d) * sqrt(n)       (|t| < ~2 -> could easily be luck)
    Also reports the annualised mean difference and tracking error.
    Caveat: daily returns are not perfectly independent, so treat it as rough.
    """
    d = (a - b).dropna()
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    t = float(d.mean()) / sd * math.sqrt(n) if sd and sd > 0 else float("nan")
    return {
        "MeanDiff/yr": float(d.mean()) * TRADING_DAYS_PER_YEAR,
        "TrackingErr": sd * math.sqrt(TRADING_DAYS_PER_YEAR),
        "t_stat": t,
        "n_days": float(n),
    }


def your_turn_monthly_win_rate(returns: pd.Series) -> float:
    """YOUR TURN (optional exercise #3): share of calendar months with a gain.

    Steps: compound daily returns into monthly returns
    ( (1 + r).resample("ME").prod() - 1 ), then return the fraction > 0.
    Not used by run.py yet - add it to `summarize()` when it works.
    """
    raise NotImplementedError("YOUR TURN: implement the monthly win rate")
