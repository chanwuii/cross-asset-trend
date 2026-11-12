import numpy as np
import pandas as pd

from sizing import inverse_vol_weights, target_weights


def _returns(vols, n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {k: rng.normal(0, v, n) for k, v in vols.items()}
    return pd.DataFrame(data, index=pd.bdate_range("2020-01-01", periods=n))


def test_inverse_vol_sums_to_one_and_favors_calm_assets():
    w = inverse_vol_weights(pd.Series({"calm": 0.01, "wild": 0.03}))
    assert abs(w.sum() - 1) < 1e-12
    assert w["calm"] > w["wild"]
    assert abs(w["calm"] - 0.75) < 1e-12  # (1/0.01)/(1/0.01+1/0.03)


def test_weights_respect_caps_and_no_leverage():
    rets = _returns({"A": 0.002, "B": 0.003, "C": 0.004})  # very calm -> would want leverage
    sig = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
    w = target_weights(sig, rets, target_vol=0.10, max_weight=0.4, max_gross=1.0)
    assert (w <= 0.4 + 1e-12).all()
    assert w.sum() <= 1.0 + 1e-12
    assert (w >= 0).all()


def test_inactive_assets_get_zero_and_all_off_means_cash():
    rets = _returns({"A": 0.01, "B": 0.01})
    w = target_weights(pd.Series({"A": 1.0, "B": 0.0}), rets)
    assert w["B"] == 0.0 and w["A"] > 0
    w_off = target_weights(pd.Series({"A": 0.0, "B": 0.0}), rets)
    assert w_off.sum() == 0.0


def test_vol_target_scales_down_risky_basket():
    rets = _returns({"A": 0.03, "B": 0.03}, n=500)  # ~47% annual vol each
    w = target_weights(pd.Series({"A": 1.0, "B": 1.0}), rets, target_vol=0.10, max_weight=1.0)
    port = rets @ w
    realized = port.std() * np.sqrt(252)
    assert abs(realized - 0.10) < 0.01
