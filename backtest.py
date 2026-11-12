"""The backtest engine: replay history day by day.

Timing rule (the most important line in any backtest):
  * On the LAST trading day of each month we look at closes up to that day
    and decide target weights ("signal day", t).
  * We trade at the close of the NEXT trading day (t+1).
  * So the new weights only start earning returns from day t+2.
There is no way for tomorrow's price to sneak into today's decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sizing import target_weights


@dataclass
class BacktestResult:
    equity: pd.Series        # portfolio value each day
    returns: pd.Series       # daily portfolio return (after costs)
    weights: pd.DataFrame    # risky weights held at the end of each day
    turnover: pd.Series      # sum |trade| on each trade day (0 otherwise)
    costs: pd.Series         # cost paid each day, as a fraction of equity
    targets: pd.DataFrame    # target weights decided on each signal day


def month_end_dates(index: pd.DatetimeIndex, start: str | pd.Timestamp) -> list[pd.Timestamp]:
    """Last trading day of every month in `index` that is on/after `start`."""
    dates = pd.Series(index, index=index)
    last_per_month = dates.groupby([index.year, index.month]).max()
    start = pd.Timestamp(start)
    return [d for d in last_per_month if d >= start]


def compute_targets(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    sizing_cfg: dict,
    multipliers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Target weights for each signal date (rows) and asset (cols).

    For date t we only pass `returns.loc[:t]` and `signal.loc[t]`
    into the sizing function - data up to t, nothing after.
    """
    lookback = sizing_cfg["vol_lookback_days"]
    rows = {}
    for t in signal_dates:
        recent = returns.loc[:t].tail(lookback)
        mult = None
        if multipliers is not None and t in multipliers.index:
            mult = multipliers.loc[t]
        rows[t] = target_weights(
            signal.loc[t],
            recent,
            target_vol=sizing_cfg["target_vol"],
            max_weight=sizing_cfg["max_weight"],
            max_gross=sizing_cfg["max_gross"],
            multipliers=mult,
        )
    return pd.DataFrame(rows).T.reindex(columns=returns.columns).fillna(0.0)


def fixed_targets(weights: dict[str, float], columns: list[str], signal_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Same weights every month - used for SPY buy&hold and 60/40 benchmarks."""
    row = pd.Series(0.0, index=columns)
    for k, v in weights.items():
        row[k] = v
    return pd.DataFrame([row] * len(signal_dates), index=signal_dates)


def simulate(
    returns: pd.DataFrame,
    targets: pd.DataFrame,
    start: str | pd.Timestamp,
    cost_bps: float = 5.0,
    initial_capital: float = 100_000.0,
    cash_returns: pd.Series | None = None,
) -> BacktestResult:
    """Run the day-by-day simulation.

    Each day:
      1. Apply today's asset returns to yesterday's holdings.
      2. Let weights drift (winners grow, losers shrink).
      3. If today is the day after a signal day, trade to the new target
         and pay cost = cost_bps/10,000 * turnover.
    """
    days = returns.index[returns.index >= pd.Timestamp(start)]
    assets = list(returns.columns)
    r = returns.loc[days, assets].fillna(0.0).to_numpy()
    if cash_returns is None:
        rc = np.zeros(len(days))
    else:
        rc = cash_returns.reindex(days).fillna(0.0).to_numpy()

    # Map: execution day -> the signal day whose targets we trade to.
    all_days = list(returns.index)
    execute_on: dict[pd.Timestamp, pd.Timestamp] = {}
    for t in targets.index:
        i = all_days.index(t)
        if i + 1 < len(all_days):
            execute_on[all_days[i + 1]] = t

    cost_rate = cost_bps / 10_000.0
    w = np.zeros(len(assets))          # risky weights; cash = 1 - w.sum()
    equity = initial_capital

    eq_hist, ret_hist, w_hist, to_hist, cost_hist = [], [], [], [], []
    for i, day in enumerate(days):
        cash_w = 1.0 - w.sum()
        gross = float(w @ r[i] + cash_w * rc[i])
        equity_before = equity
        equity *= 1.0 + gross
        # Drift: each position grows with its own return.
        if 1.0 + gross != 0:
            w = w * (1.0 + r[i]) / (1.0 + gross)

        turnover = 0.0
        cost = 0.0
        if day in execute_on:
            new_w = targets.loc[execute_on[day], assets].to_numpy(dtype=float)
            turnover = float(np.abs(new_w - w).sum())
            cost = cost_rate * turnover
            equity *= 1.0 - cost
            w = new_w

        eq_hist.append(equity)
        ret_hist.append(equity / equity_before - 1.0)
        w_hist.append(w.copy())
        to_hist.append(turnover)
        cost_hist.append(cost)

    return BacktestResult(
        equity=pd.Series(eq_hist, index=days, name="equity"),
        returns=pd.Series(ret_hist, index=days, name="returns"),
        weights=pd.DataFrame(w_hist, index=days, columns=assets),
        turnover=pd.Series(to_hist, index=days, name="turnover"),
        costs=pd.Series(cost_hist, index=days, name="costs"),
        targets=targets,
    )
