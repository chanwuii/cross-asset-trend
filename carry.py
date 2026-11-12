"""v2 experiment: carry - what an asset "pays you to hold it" if prices stay flat.

All carry numbers are measured relative to cash (the 3-month T-bill), so
positive carry = "holding this beats holding T-bills if nothing moves".

  Equities (SPY QQQ EFA EEM VNQ): trailing 12-month dividends / price - T-bill
  Bonds    (TLT, IEF):           Treasury yield of a matching tenor - T-bill
                                 TLT -> 30y (^TYX), IEF -> 10y (^TNX)
  Dollar   (UUP):                US short rate - weighted foreign short rate
                                 (euro via Germany, Japan, UK, Canada; DXY-style
                                 weights; OECD monthly data from FRED, lagged)
  Commodities (DBC):             SKIPPED. Roll yield needs historical prices of
                                 expired futures contracts; Yahoo does not keep
                                 them. TODO: add a futures-curve data source.
  Gold (GLD):                    NO TILT. GLD holds physical gold (no roll). A
                                 futures-style definition would be lease rate - T-bill,
                                 but no free historical lease-rate series. See YOUR TURN.
  Cash (BIL):                    cash itself - never tilted.

Point-in-time rules
  * Dividends count only once their ex-date is on/before the signal date.
  * Daily yields: we use the last close on/before the signal date.
  * Monthly OECD rates: month M is treated as known only from the first day
    of month M + 1 + publication_lag_months (default: 2 months later).
Nothing in this file is filled with a value from the future, and if a
download fails that asset simply gets no carry (NaN -> no tilt).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


# ---------------------------------------------------------------- downloads
def _cached(path: str | None, refresh: bool) -> pd.DataFrame | None:
    if path and Path(path).exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None


def _save(df: pd.DataFrame, path: str | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)


def fetch_dividends(tickers: list[str], cache_path: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Cash dividend per share on each ex-date (rows = ex-dates, cols = tickers)."""
    cached = _cached(cache_path, refresh)
    if cached is not None and set(tickers).issubset(cached.columns):
        return cached[tickers]
    series = {}
    for t in tickers:
        try:
            d = yf.Ticker(t).dividends
            d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
            series[t] = d
        except Exception as exc:
            print(f"[carry] dividends failed for {t}: {exc} -> no carry for {t}")
    out = pd.DataFrame(series).sort_index()
    _save(out, cache_path)
    return out


def fetch_raw_close(tickers: list[str], start: str, cache_path: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Un-adjusted closes (what the price really was that day), for yields."""
    cached = _cached(cache_path, refresh)
    if cached is not None and set(tickers).issubset(cached.columns):
        return cached[tickers]
    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw[tickers].dropna(how="all")
    _save(raw, cache_path)
    return raw


def fetch_fred_monthly(series_ids: dict[str, str], cache_path: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Monthly OECD short rates from FRED (percent). cols = our labels (e.g. 'EUR')."""
    cached = _cached(cache_path, refresh)
    if cached is not None and set(series_ids).issubset(cached.columns):
        return cached[list(series_ids)]
    cols = {}
    for label, sid in series_ids.items():
        try:
            df = pd.read_csv(FRED_CSV.format(sid=sid), index_col=0, parse_dates=True)
            cols[label] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        except Exception as exc:
            print(f"[carry] FRED {sid} failed: {exc}")
    out = pd.DataFrame(cols)
    _save(out, cache_path)
    return out


# ---------------------------------------------------------------- carry math
def trailing_dividend_yield(raw_close: pd.Series, dividends: pd.Series, window_days: int = 365) -> pd.Series:
    """D_t / P_t: dividends with ex-date in (t - 365 days, t] divided by today's price.

    NaN during the first year of price history (we can't know a full year yet).
    """
    idx = raw_close.dropna().index
    divs = dividends.dropna()
    divs = divs[divs > 0]
    # Put each dividend on its ex-date (or the next trading day) in our index.
    pos = idx.searchsorted(divs.index)
    keep = pos < len(idx)
    daily = pd.Series(0.0, index=idx)
    for p, amt in zip(pos[keep], divs.to_numpy()[keep]):
        daily.iloc[p] += amt
    d12 = daily.rolling(f"{window_days}D").sum()
    yld = d12 / raw_close.reindex(idx)
    first_full = idx[0] + pd.Timedelta(days=window_days)
    return yld.where(idx >= first_full)


def monthly_to_available_daily(monthly: pd.Series, daily_index: pd.DatetimeIndex, publication_lag_months: int = 1) -> pd.Series:
    """Shift a monthly series to the date it was actually known, then forward-fill.

    A value labelled 2020-03-01 (= March average) becomes usable from
    2020-03-01 + (1 + lag) months, i.e. 2020-05-01 with the default lag 1.
    """
    m = monthly.dropna()
    known_from = m.index.to_period("M") + 1 + publication_lag_months
    shifted = pd.Series(m.to_numpy(), index=known_from.to_timestamp())
    shifted = shifted[~shifted.index.duplicated(keep="last")]
    combined = shifted.reindex(shifted.index.union(daily_index)).sort_index().ffill()
    return combined.reindex(daily_index)


def build_carry(cfg: dict, assets: list[str], daily_index: pd.DatetimeIndex, refresh: bool = False) -> pd.DataFrame:
    """Daily carry (decimal, e.g. 0.012 = 1.2%/yr) per asset; NaN = not available."""
    carry = pd.DataFrame(index=daily_index, columns=assets, dtype=float)
    rates_tickers = [cfg["short_rate"]] + list(cfg["bonds"].values())
    rates = fetch_raw_close(rates_tickers, cfg["start"], cfg["rates_cache"], refresh)
    rates = rates.reindex(daily_index).ffill(limit=5) / 100.0  # yields are quoted in percent
    tbill = rates[cfg["short_rate"]]

    # Equities: dividend yield - T-bill
    eq = [a for a in cfg["equities"] if a in assets]
    raw = fetch_raw_close(eq, cfg["start"], cfg["raw_close_cache"], refresh)
    divs = fetch_dividends(eq, cfg["dividends_cache"], refresh)
    for a in eq:
        if a not in divs.columns or divs[a].dropna().empty:
            print(f"[carry] TODO: no dividend data for {a}, no carry tilt")
            continue
        dy = trailing_dividend_yield(raw[a], divs[a]).reindex(daily_index)
        carry[a] = dy - tbill

    # Bonds: matching-tenor yield - T-bill
    for a, yt in cfg["bonds"].items():
        if a in assets:
            carry[a] = rates[yt] - tbill

    # Dollar: US short rate - weighted foreign short rate
    fx = cfg.get("fx")
    if fx and fx["asset"] in assets:
        ids = {"US": fx["us_series"], **{k: v["series"] for k, v in fx["foreign"].items()}}
        monthly = fetch_fred_monthly(ids, cfg["fred_cache"], refresh)
        lag = fx.get("publication_lag_months", 1)
        if "US" in monthly and all(k in monthly for k in fx["foreign"]):
            us = monthly_to_available_daily(monthly["US"], daily_index, lag)
            w_total = sum(v["weight"] for v in fx["foreign"].values())
            foreign = sum(
                monthly_to_available_daily(monthly[k], daily_index, lag) * v["weight"] / w_total
                for k, v in fx["foreign"].items()
            )
            carry[fx["asset"]] = (us - foreign) / 100.0
        else:
            print("[carry] TODO: FRED short rates incomplete, no carry tilt for UUP")

    for a in cfg.get("skipped", {}):
        print(f"[carry] {a}: no carry tilt ({cfg['skipped'][a]})")
    return carry


def carry_multipliers(
    carry: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    agree: float = 1.25,
    disagree: float = 0.75,
) -> pd.DataFrame:
    """Pre-specified rule, applied only to assets the trend already holds (long):

        carry > 0  (agrees with a long trend)   -> weight x `agree`    (1.25)
        carry < 0  (disagrees)                   -> weight x `disagree` (0.75)
        carry NaN / 0 (unknown)                  -> x 1.0

    Uses carry.loc[:t] only - the last value known on the signal date.
    """
    rows = {}
    for t in signal_dates:
        known = carry.loc[:t]
        last = known.iloc[-1] if len(known) else pd.Series(float("nan"), index=carry.columns)
        m = pd.Series(1.0, index=carry.columns)
        m[last > 0] = agree
        m[last < 0] = disagree
        rows[t] = m
    return pd.DataFrame(rows).T


def your_turn_gold_carry(tbill: pd.Series) -> pd.Series:
    """YOUR TURN (optional exercise #7 in the study guide): give GLD a carry.

    Holding gold pays no income, so versus cash its carry is roughly
    -(T-bill rate) (ignoring the small gold lease rate). Return that series,
    plug it into build_carry for GLD, and see whether the A/B changes.
    Think: does this make GLD "disagree" almost always? Is that sensible?
    """
    raise NotImplementedError("YOUR TURN: gold carry = -T-bill")
