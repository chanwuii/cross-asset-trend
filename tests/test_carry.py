import numpy as np
import pandas as pd

from carry import carry_multipliers, monthly_to_available_daily, trailing_dividend_yield
from metrics import difference_test


def _idx():
    return pd.bdate_range("2018-01-01", "2020-12-31")


def test_dividend_yield_basic_and_first_year_nan():
    idx = _idx()
    close = pd.Series(100.0, index=idx)
    divs = pd.Series([0.5, 0.5, 0.5, 0.5], index=pd.to_datetime(["2018-03-15", "2018-06-15", "2018-09-14", "2018-12-14"]))
    dy = trailing_dividend_yield(close, divs)
    assert dy.loc[:"2018-12-31"].isna().all()                  # not a full year yet
    assert abs(dy.loc["2019-01-02"] - 0.02) < 1e-12            # 4 x 0.5 / 100


def test_dividend_yield_ignores_future_dividends():
    idx = _idx()
    close = pd.Series(100.0, index=idx)
    divs = pd.Series([1.0], index=pd.to_datetime(["2019-06-14"]))
    more = pd.concat([divs, pd.Series([50.0], index=pd.to_datetime(["2020-03-16"]))])
    t = pd.Timestamp("2019-12-31")
    assert trailing_dividend_yield(close, divs).loc[t] == trailing_dividend_yield(close, more).loc[t]


def test_monthly_rate_is_lagged_to_publication():
    monthly = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    daily = monthly_to_available_daily(monthly, pd.bdate_range("2020-01-01", "2020-06-30"), publication_lag_months=1)
    assert np.isnan(daily.loc["2020-02-28"])         # January not yet published
    assert daily.loc["2020-03-02"] == 1.0            # January known from March
    assert daily.loc["2020-04-30"] == 2.0            # February known from April, March not yet


def test_carry_multiplier_uses_only_known_values():
    idx = _idx()
    carry = pd.DataFrame({"A": 0.01, "B": -0.01, "C": np.nan}, index=idx)
    t = pd.Timestamp("2019-06-28")
    m1 = carry_multipliers(carry, [t])
    changed = carry.copy()
    changed.loc[changed.index > t] = -1.0           # flip everything AFTER t
    m2 = carry_multipliers(changed, [t])
    pd.testing.assert_frame_equal(m1, m2)
    assert list(m1.loc[t]) == [1.25, 0.75, 1.0]


def test_difference_test_detects_a_clear_edge():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=1000)
    b = pd.Series(rng.normal(0, 0.01, 1000), index=idx)
    a = b + 0.001 + pd.Series(rng.normal(0, 0.001, 1000), index=idx)  # +0.1%/day edge, small noise
    out = difference_test(a, b)
    assert abs(out["MeanDiff/yr"] - 0.252) < 0.03
    assert out["t_stat"] > 10
