import math

import numpy as np
import pandas as pd

from metrics import annual_vol, cagr_from_returns, calmar, max_drawdown, sharpe, sortino


def test_cagr_of_constant_growth():
    idx = pd.date_range("2020-01-01", "2022-01-01", freq="D")
    # grow exactly 21% over two years -> 10% per year
    daily = (1.21) ** (1 / (len(idx) - 1)) - 1
    r = pd.Series(daily, index=idx)
    r.iloc[0] = 0.0
    assert abs(cagr_from_returns(r) - 0.10) < 1e-3


def test_max_drawdown_toy():
    # 100 -> 120 -> 60 -> 90 : worst fall is 120 -> 60 = -50%
    r = pd.Series([0.0, 0.2, -0.5, 0.5], index=pd.bdate_range("2021-01-01", periods=4))
    assert abs(max_drawdown(r) - (-0.5)) < 1e-12


def test_vol_and_sharpe_scale():
    r = pd.Series([0.01, -0.01] * 50, index=pd.bdate_range("2021-01-01", periods=100))
    assert abs(annual_vol(r) - r.std() * math.sqrt(252)) < 1e-12
    assert abs(sharpe(r)) < 1e-9  # mean is zero


def test_sortino_ignores_upside():
    r = pd.Series([0.02, -0.01, 0.02, -0.01], index=pd.bdate_range("2021-01-01", periods=4))
    expected = r.mean() / math.sqrt(np.mean(np.minimum(r, 0) ** 2)) * math.sqrt(252)
    assert abs(sortino(r) - expected) < 1e-12


def test_calmar_positive_when_growing_with_dip():
    r = pd.Series([0.01] * 300 + [-0.05] + [0.01] * 300, index=pd.bdate_range("2020-01-01", periods=601))
    assert calmar(r) > 0
