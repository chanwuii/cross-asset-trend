import numpy as np
import pandas as pd

from backtest import month_end_dates
from ensemble import ensemble_score, ensemble_targets

SIZING = {"vol_lookback_days": 63, "target_vol": 0.10, "max_weight": 0.3, "max_gross": 1.0}
SPEEDS = [1, 3, 6, 12]


def _prices(seed=0, n=800):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n)
    rets = rng.normal(0.0003, 0.01, size=(n, 4))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list("ABCD"))


def _targets(prices):
    rets = prices.pct_change(fill_method=None)
    dates = [d for d in month_end_dates(rets.index, "2015-03-01") if d < rets.index[-1]]
    return ensemble_targets(prices, rets, dates, SIZING, SPEEDS)


def test_score_takes_only_quarter_steps():
    score = ensemble_score(_prices(), SPEEDS)
    assert set(np.unique(score.to_numpy())) <= {0.0, 0.25, 0.5, 0.75, 1.0}


def test_ensemble_is_average_of_speeds_and_respects_limits():
    total, pieces = _targets(_prices(1))
    pd.testing.assert_frame_equal(total, sum(pieces.values()))
    assert (total.sum(axis=1) <= 1.0 + 1e-9).all()      # no leverage
    assert (total.to_numpy() <= 0.3 + 1e-9).all()        # caps hold after averaging
    assert (total.to_numpy() >= 0).all()                 # long / cash only
    for p in pieces.values():                            # each speed is at most 1/4 of the book
        assert (p.sum(axis=1) <= 0.25 + 1e-9).all()


def test_ensemble_has_no_lookahead():
    prices = _prices(2)
    cut = pd.Timestamp("2016-06-30")
    shocked = prices.copy()
    shocked.loc[shocked.index > cut] *= 0.2
    t1, _ = _targets(prices)
    t2, _ = _targets(shocked)
    past = t1.index[t1.index <= cut]
    pd.testing.assert_frame_equal(t1.loc[past], t2.loc[past])
