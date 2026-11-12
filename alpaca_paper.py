"""Stage 3 (stub): turn the latest target weights into a paper-trading order list.

SAFETY
  * Default is DRY-RUN: it only PRINTS orders. Nothing is sent anywhere.
  * Orders are only sent with the explicit `--submit` flag, and only to
    Alpaca's PAPER endpoint (fake money). There is no live-trading path.
  * Keys come from environment variables and are never printed:
        ALPACA_PAPER_API_KEY_ID, ALPACA_PAPER_API_SECRET

Usage:
    python run.py                       # writes output/v2/latest_target_weights.csv (ensemble)
    python alpaca_paper.py              # dry-run with a pretend $100,000 account
    python alpaca_paper.py --equity 25000
    python alpaca_paper.py --use-account  # dry-run, but read equity/positions (read-only)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import pandas as pd

PAPER_URL = "https://paper-api.alpaca.markets"


def _headers() -> dict[str, str]:
    key = os.environ.get("ALPACA_PAPER_API_KEY_ID")
    secret = os.environ.get("ALPACA_PAPER_API_SECRET")
    if not key or not secret:
        sys.exit("Set ALPACA_PAPER_API_KEY_ID and ALPACA_PAPER_API_SECRET first.")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def read_account() -> tuple[float, dict[str, float]]:
    """Read-only: account equity and current share quantities (paper account)."""
    import requests

    h = _headers()
    acct = requests.get(f"{PAPER_URL}/v2/account", headers=h, timeout=15)
    acct.raise_for_status()
    pos = requests.get(f"{PAPER_URL}/v2/positions", headers=h, timeout=15)
    pos.raise_for_status()
    holdings = {p["symbol"]: float(p["qty"]) for p in pos.json()}
    return float(acct.json()["equity"]), holdings


def plan_orders(
    target_weights: pd.Series,
    prices: pd.Series,
    equity: float,
    current_qty: dict[str, float] | None = None,
    min_trade_value: float = 50.0,
) -> pd.DataFrame:
    """Whole-share orders that move current holdings toward the targets.

    target_qty = floor(weight * equity / price); order = target - current.
    Tiny trades below `min_trade_value` dollars are skipped (not worth the cost).
    """
    current_qty = current_qty or {}
    rows = []
    for sym, w in target_weights.items():
        price = float(prices[sym])
        target_qty = math.floor(max(w, 0.0) * equity / price)
        delta = target_qty - current_qty.get(sym, 0.0)
        if abs(delta) * price < min_trade_value:
            continue
        rows.append({"symbol": sym, "side": "buy" if delta > 0 else "sell",
                     "qty": abs(int(delta)), "approx_value": round(abs(delta) * price, 2),
                     "target_weight": round(float(w), 4)})
    return pd.DataFrame(rows, columns=["symbol", "side", "qty", "approx_value", "target_weight"])


def submit_orders(orders: pd.DataFrame) -> None:
    """Send market-on-open style day orders to the PAPER account. Only with --submit."""
    import requests

    h = _headers()
    for o in orders.itertuples():
        body = {"symbol": o.symbol, "qty": str(o.qty), "side": o.side, "type": "market", "time_in_force": "day"}
        resp = requests.post(f"{PAPER_URL}/v2/orders", headers=h, json=body, timeout=15)
        print(o.symbol, o.side, o.qty, "->", resp.status_code)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="output/v2/latest_target_weights.csv")
    p.add_argument("--prices", default="data/prices.csv")
    p.add_argument("--equity", type=float, default=100_000.0, help="pretend account size for dry-run")
    p.add_argument("--use-account", action="store_true", help="read equity/positions from the paper account")
    p.add_argument("--submit", action="store_true", help="actually send orders to the PAPER account")
    args = p.parse_args()

    weights = pd.read_csv(args.weights, index_col=0).iloc[:, 0]
    last_prices = pd.read_csv(args.prices, index_col=0, parse_dates=True).ffill().iloc[-1]

    equity, current = args.equity, {}
    if args.use_account or args.submit:
        equity, current = read_account()

    orders = plan_orders(weights, last_prices, equity, current)
    mode = "SUBMIT (paper)" if args.submit else "DRY-RUN (nothing sent)"
    print(f"Mode: {mode} | equity used: ${equity:,.0f}")
    print(orders.to_string(index=False) if len(orders) else "No orders needed.")

    if args.submit:
        if input("Type 'yes' to send these PAPER orders: ").strip().lower() != "yes":
            print("Cancelled.")
            return
        submit_orders(orders)


if __name__ == "__main__":
    main()
