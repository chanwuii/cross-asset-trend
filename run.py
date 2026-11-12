"""End-to-end run: data -> signals -> sizing -> backtest -> metrics -> charts.

Usage:
    python run.py              # v2: v1 baseline vs multi-speed ensemble vs ensemble+carry
    python run.py --v1         # reproduce the original v1 report (incl. Google Trends) in output/v1/
    python run.py --refresh    # re-download all data (prices, carry inputs, Trends)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from backtest import compute_targets, fixed_targets, month_end_dates, simulate
from carry import build_carry, carry_multipliers
from data import daily_returns, download_prices
from ensemble import ensemble_targets
from metrics import difference_test, summarize
from overlay_trends import (
    attention_ratio,
    fetch_trends,
    multipliers_on_signal_dates,
    ratio_to_multiplier,
)
from plots import plot_drawdown, plot_equity, plot_speed_contribution, plot_weights
from signals import build_signal, trend_signal


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_metrics(name: str, res, split: str) -> list[dict]:
    """Metrics for full period, in-sample (<= split) and out-of-sample (> split)."""
    rows = []
    split_ts = pd.Timestamp(split)
    periods = {
        "Full": res.returns.index >= res.returns.index[0],
        "In-sample": res.returns.index <= split_ts,
        "Out-of-sample": res.returns.index > split_ts,
    }
    for period, mask in periods.items():
        m = summarize(res.returns[mask], res.weights[mask], res.turnover[mask], res.costs[mask])
        rows.append({"Strategy": name, "Period": period, **m})
    return rows


def ab_test(name: str, a, b, split: str) -> list[dict]:
    """t-test of (a - b) daily returns in each period."""
    split_ts = pd.Timestamp(split)
    idx = a.returns.index
    out = []
    for period, mask in {"Full": idx >= idx[0], "In-sample": idx <= split_ts, "Out-of-sample": idx > split_ts}.items():
        out.append({"Comparison": name, "Period": period, **difference_test(a.returns[mask], b.returns[mask])})
    return out


def format_table(table: pd.DataFrame) -> str:
    pct = ["CAGR", "Vol", "MaxDD", "Turnover/yr", "Costs/yr", "AvgInvested", "%DaysInvested"]
    shown = table.copy()
    for c in pct:
        if c in shown:
            shown[c] = shown[c].map(lambda x: f"{x:.2%}")
    for c in ["Sharpe", "Sortino", "Calmar"]:
        shown[c] = shown[c].map(lambda x: f"{x:.2f}")
    return shown.to_string(index=False)


def load_market(cfg: dict, refresh: bool):
    """Prices, returns and month-end signal dates shared by v1 and v2."""
    assets = list(cfg["universe"])
    cash = cfg["cash_ticker"]
    prices = download_prices(assets + [cash], cfg["data"]["start"], cfg["data"]["end"],
                             cfg["data"]["cache_path"], refresh=refresh)
    rets = daily_returns(prices)
    print(f"Data: {prices.index[0].date()} -> {prices.index[-1].date()}, {len(prices)} days")
    bt = cfg["backtest"]
    signal_dates = month_end_dates(rets.index, bt["start"])
    # Do not "rebalance" on the very last row: nothing comes after it yet.
    signal_dates = [d for d in signal_dates if d < rets.index[-1]]
    common = dict(start=bt["start"], cost_bps=bt["cost_bps"],
                  initial_capital=bt["initial_capital"], cash_returns=rets[cash])
    return assets, prices[assets], rets[assets], signal_dates, common


def run_v1(cfg: dict, refresh: bool) -> None:
    """The original v1 pipeline, unchanged: 12m momentum + 200d SMA, Trends overlay, sensitivity."""
    out = Path("output/v1")
    out.mkdir(parents=True, exist_ok=True)
    assets, asset_prices, asset_rets, signal_dates, common = load_market(cfg, refresh)
    split = cfg["backtest"]["in_sample_end"]

    signal = build_signal(asset_prices, cfg["signal"])
    targets = compute_targets(asset_rets, signal, signal_dates, cfg["sizing"])
    trend = simulate(asset_rets, targets, **common)
    spy = simulate(asset_rets, fixed_targets({cfg["benchmarks"]["spy_ticker"]: 1.0}, assets, signal_dates), **common)
    sixty = simulate(asset_rets, fixed_targets(cfg["benchmarks"]["sixty_forty"], assets, signal_dates), **common)
    rows = (split_metrics("Trend (Stage 1)", trend, split)
            + split_metrics("SPY buy&hold", spy, split)
            + split_metrics("60/40 SPY/IEF", sixty, split))

    tcfg = cfg.get("trends", {})
    overlay = None
    if tcfg.get("enabled"):
        monthly = fetch_trends(tcfg["keywords"], tcfg["timeframe"], tcfg["cache_path"], refresh=refresh)
        if monthly is not None:
            ratio = attention_ratio(monthly, tcfg["short_months"], tcfg["long_months"])
            mult_m = ratio_to_multiplier(ratio, tcfg["threshold"], tcfg["tilt"])
            mult = multipliers_on_signal_dates(mult_m, signal_dates, tcfg["lag_months"])
            ov_targets = compute_targets(asset_rets, signal, signal_dates, cfg["sizing"], multipliers=mult)
            overlay = simulate(asset_rets, ov_targets, **common)
            rows += split_metrics("Trend + Trends overlay", overlay, split)
            mult.to_csv(out / "trends_multipliers.csv")
            print(f"[trends] share of asset-months tilted: {float((mult != 1.0).to_numpy().mean()):.1%}")
        else:
            print("[trends] overlay SKIPPED: no real Google Trends data (no synthetic fallback).")

    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)

    sens_rows = []
    s = cfg["sensitivity"]
    for lb in s["momentum_lookback_months"]:
        for win in s["sma_window"]:
            sig = trend_signal(asset_prices, lookback_months=lb, sma_window=win,
                               skip_months=cfg["signal"].get("skip_months", 0))
            res = simulate(asset_rets, compute_targets(asset_rets, sig, signal_dates, cfg["sizing"]), **common)
            for r in split_metrics("trend", res, split):
                if r["Period"] != "Full":
                    sens_rows.append({"lookback_m": lb, "sma": win, "Period": r["Period"],
                                      "CAGR": r["CAGR"], "Sharpe": r["Sharpe"], "MaxDD": r["MaxDD"]})
    pd.DataFrame(sens_rows).to_csv(out / "sensitivity.csv", index=False)

    curves = {"Trend (Stage 1)": trend.returns, "SPY buy&hold": spy.returns, "60/40": sixty.returns}
    if overlay is not None:
        curves["Trend + Trends overlay"] = overlay.returns
    plot_equity(curves, str(out / "equity_curve.png"), split_date=split)
    plot_drawdown(curves, str(out / "drawdown.png"))
    plot_weights(trend.weights, str(out / "weights.png"))
    pd.DataFrame({k: (1 + v).cumprod() for k, v in curves.items()}).to_csv(out / "equity.csv")
    latest = compute_targets(asset_rets, signal, [asset_rets.index[-1]], cfg["sizing"]).iloc[0]
    latest.rename("weight").to_csv(out / "latest_target_weights.csv")

    print("\n=== v1 results (after costs) ===")
    print(format_table(table))
    print(f"\nSaved to {out}/")


def run_v2(cfg: dict, refresh: bool) -> None:
    """v2: v1 baseline vs multi-speed ensemble vs ensemble + carry tilt, same period and costs."""
    out = Path("output/v2")
    out.mkdir(parents=True, exist_ok=True)
    assets, asset_prices, asset_rets, signal_dates, common = load_market(cfg, refresh)
    split = cfg["backtest"]["in_sample_end"]
    ecfg = cfg["ensemble"]

    # Baseline: exactly the v1 core (12m momentum + 200d SMA).
    v1_signal = build_signal(asset_prices, cfg["signal"])
    v1 = simulate(asset_rets, compute_targets(asset_rets, v1_signal, signal_dates, cfg["sizing"]), **common)

    # New core: multi-speed ensemble.
    ens_t, pieces = ensemble_targets(asset_prices, asset_rets, signal_dates, cfg["sizing"],
                                     ecfg["speeds_months"], ecfg.get("skip_months", 0))
    ens = simulate(asset_rets, ens_t, **common)

    # Each speed on its own (for understanding only - NOT used to pick weights).
    speed_rows = []
    for m, piece in pieces.items():
        alone = simulate(asset_rets, piece * len(pieces), **common)
        for r in split_metrics(f"{m}m only", alone, split):
            if r["Period"] != "Full":
                speed_rows.append({k: r[k] for k in ["Strategy", "Period", "CAGR", "Sharpe", "MaxDD", "Turnover/yr"]})
    speeds_table = pd.DataFrame(speed_rows)
    speeds_table.to_csv(out / "speeds_standalone.csv", index=False)

    spy = simulate(asset_rets, fixed_targets({cfg["benchmarks"]["spy_ticker"]: 1.0}, assets, signal_dates), **common)
    sixty = simulate(asset_rets, fixed_targets(cfg["benchmarks"]["sixty_forty"], assets, signal_dates), **common)

    rows = (split_metrics("v1 12m+SMA200", v1, split)
            + split_metrics("Ensemble 1/3/6/12m", ens, split))
    ab_rows = ab_test("Ensemble - v1", ens, v1, split)

    # Carry A/B on top of the ensemble.
    ccfg = cfg.get("carry", {})
    ens_carry = None
    if ccfg.get("enabled"):
        carry = build_carry(ccfg, assets, asset_rets.index, refresh=refresh)
        carry.to_csv(out / "carry_daily.csv")
        mult = carry_multipliers(carry, signal_dates, ccfg["agree"], ccfg["disagree"])
        mult.to_csv(out / "carry_multipliers.csv")
        ec_t, _ = ensemble_targets(asset_prices, asset_rets, signal_dates, cfg["sizing"],
                                   ecfg["speeds_months"], ecfg.get("skip_months", 0), multipliers=mult)
        ens_carry = simulate(asset_rets, ec_t, **common)
        rows += split_metrics("Ensemble + carry", ens_carry, split)
        ab_rows += ab_test("Ensemble+carry - Ensemble", ens_carry, ens, split)
        # How often did the rule actually bite (on assets the ensemble held)?
        held = ens_t > 0
        bite = (mult.reindex(ens_t.index)[held] != 1.0).sum().sum() / max(held.sum().sum(), 1)
        coverage = carry.loc[carry.index >= pd.Timestamp(common["start"])].notna().mean()
        print(f"[carry] share of held asset-months tilted: {bite:.1%}")
        print("[carry] data coverage since backtest start:", coverage.round(2).to_dict())

    rows += split_metrics("SPY buy&hold", spy, split) + split_metrics("60/40 SPY/IEF", sixty, split)
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)
    ab = pd.DataFrame(ab_rows)
    ab.to_csv(out / "ab_tests.csv", index=False)

    curves = {"v1 12m+SMA200": v1.returns, "Ensemble": ens.returns}
    if ens_carry is not None:
        curves["Ensemble + carry"] = ens_carry.returns
    curves.update({"SPY buy&hold": spy.returns, "60/40": sixty.returns})
    plot_equity(curves, str(out / "equity_curve.png"), split_date=split)
    plot_drawdown(curves, str(out / "drawdown.png"))
    plot_weights(ens.weights, str(out / "weights_ensemble.png"))
    plot_speed_contribution(pieces, str(out / "speed_contribution.png"))
    pd.DataFrame({k: (1 + v).cumprod() for k, v in curves.items()}).to_csv(out / "equity.csv")

    # Latest ensemble target for the Stage 3 dry-run (decided at the last close).
    latest, _ = ensemble_targets(asset_prices, asset_rets, [asset_rets.index[-1]], cfg["sizing"],
                                 ecfg["speeds_months"], ecfg.get("skip_months", 0))
    latest.iloc[0].rename("weight").to_csv(out / "latest_target_weights.csv")

    pd.set_option("display.width", 220)
    print("\n=== v2 results (after costs) ===")
    print(format_table(table))
    print("\n=== Each speed alone (diagnostic only) ===")
    print(speeds_table.round(4).to_string(index=False))
    print("\n=== A/B difference tests (daily returns) ===")
    print(ab.round(4).to_string(index=False))
    w = latest.iloc[0]
    print("\nLatest ensemble target weights (cash = rest):")
    print(w[w > 0].round(3).to_string(), f"\ncash {1 - w.sum():.3f}")
    print(f"\nSaved to {out}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--refresh", action="store_true", help="re-download data")
    parser.add_argument("--v1", action="store_true", help="reproduce the original v1 report")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.v1:
        run_v1(cfg, args.refresh)
    else:
        run_v2(cfg, args.refresh)


if __name__ == "__main__":
    main()
