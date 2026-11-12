"""If we change prices AFTER day T, nothing decided on or before T may change."""

import numpy as np
import pandas as pd

from backtest import compute_targets, month_end_dates, simulate
from signals import trend_signal

SIZING = {"vol_lookback_days": 63, "target_vol": 0.10, "max_weight": 0.5, "max_gross": 1.0}


def _toy_prices(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=700)
    rets = rng.normal(0.0004, 0.01, size=(len(idx), 3))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=["A", "B", "C"])


def _run(prices: pd.DataFrame):
    rets = prices.pct_change(fill_method=None)
    sig = trend_signal(prices, lookback_months=6, sma_window=100)
    dates = [d for d in month_end_dates(rets.index, "2016-01-01") if d < rets.index[-1]]
    targets = compute_targets(rets, sig, dates, SIZING)
    return targets, simulate(rets, targets, start="2016-01-01", cost_bps=5)


def test_future_prices_do_not_change_past_decisions():
    prices = _toy_prices()
    cut = pd.Timestamp("2016-09-30")
    shocked = prices.copy()
    shocked.loc[shocked.index > cut] *= 3.0  # wild change in the "future"

    t1, r1 = _run(prices)
    t2, r2 = _run(shocked)

    past = t1.index[t1.index <= cut]
    pd.testing.assert_frame_equal(t1.loc[past], t2.loc[past])
    # Equity up to and including the cut day must be identical too.
    pd.testing.assert_series_equal(r1.equity.loc[:cut], r2.equity.loc[:cut])


def test_trade_happens_day_after_signal():
    prices = _toy_prices(1)
    targets, res = _run(prices)
    first_signal = targets.index[0]
    days = list(res.equity.index)
    nonzero_days = res.turnover[res.turnover > 0].index
    if targets.iloc[0].sum() > 0:
        assert nonzero_days[0] == days[days.index(first_signal) + 1]


def test_signal_uses_only_past_data():
    prices = _toy_prices(2)
    t = prices.index[400]
    full = trend_signal(prices, 6, 0, 100).loc[t]
    truncated = trend_signal(prices.loc[:t], 6, 0, 100).loc[t]
    pd.testing.assert_series_equal(full, truncated)
