import pandas as pd

from alpaca_paper import plan_orders


def test_plan_orders_whole_shares_and_sides():
    w = pd.Series({"SPY": 0.5, "GLD": 0.0})
    px = pd.Series({"SPY": 100.0, "GLD": 200.0})
    orders = plan_orders(w, px, equity=10_000, current_qty={"GLD": 10})
    o = orders.set_index("symbol")
    assert o.loc["SPY", "side"] == "buy" and o.loc["SPY", "qty"] == 50
    assert o.loc["GLD", "side"] == "sell" and o.loc["GLD", "qty"] == 10
