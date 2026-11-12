import pandas as pd

from overlay_trends import attention_ratio, multipliers_on_signal_dates, ratio_to_multiplier


def test_ratio_is_scale_invariant():
    idx = pd.date_range("2015-01-01", periods=40, freq="MS")
    x = pd.DataFrame({"A": range(10, 50)}, index=idx, dtype=float)
    r1 = attention_ratio(x, 3, 12)
    r2 = attention_ratio(x * 0.5, 3, 12)  # Google re-scaling = constant factor
    pd.testing.assert_frame_equal(r1, r2)


def test_multiplier_levels():
    ratio = pd.DataFrame({"A": [1.5, 1.0, 0.5, float("nan")]})
    m = ratio_to_multiplier(ratio, threshold=0.2, tilt=0.25)
    assert list(m["A"]) == [1.25, 1.0, 0.75, 1.0]


def test_lag_uses_previous_month_only():
    idx = pd.date_range("2020-01-01", periods=4, freq="MS")  # Jan..Apr
    mult = pd.DataFrame({"A": [1.0, 1.25, 0.75, 1.25]}, index=idx)
    out = multipliers_on_signal_dates(mult, [pd.Timestamp("2020-03-31")], lag_months=1)
    assert out.loc[pd.Timestamp("2020-03-31"), "A"] == 1.25  # February's value, not March's
