"""Stage 2: Google Trends "attention" overlay on top of the trend weights.

Idea
----
If people are searching for "gold price" much more than usual, attention
is flowing into gold. Trend followers might want a bit MORE of an asset
that is both trending up and gaining attention, and a bit LESS of one
whose attention is fading. This module turns search interest into a
per-asset multiplier (e.g. 1.25 / 1.00 / 0.75) for sizing.target_weights.

Data caveats (read before trusting any result!)
-----------------------------------------------
1. Re-normalization: Google returns values 0-100 scaled so the MAX of the
   requested window = 100. Download a different window and every number
   changes. A value from 2012 we download today was scaled using 2026's
   peak - information nobody had in 2012 (look-ahead!).
   Mitigation: we only use a RATIO (recent average / older average) of the
   same series. A constant scale factor cancels out in a ratio, so the
   ratio does not depend on the future peak. It is still not perfect:
   values are rounded to integers (small numbers get noisy) and Google
   samples its data, so repeat downloads can differ slightly.
2. Sampling / revisions: Google Trends is computed from a sample of
   searches and its methodology has changed (e.g. 2011, 2016, 2022).
   Today's history is not exactly what a trader would have seen back then.
3. Keyword choice: "gold price" for GLD is a judgement call made today.
4. Publication lag: a month's value is only complete after the month ends,
   so we lag by `lag_months` (default 1) before using it.
A true point-in-time test would require saving a snapshot every month going
forward - see the README roadmap.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pandas as pd


def fetch_trends(
    keywords: dict[str, str],
    timeframe: str,
    cache_path: str | None = None,
    refresh: bool = False,
    pause_seconds: float = 2.0,
) -> pd.DataFrame | None:
    """Download monthly Google Trends interest, one keyword at a time.

    Returns a DataFrame (rows = month start dates, cols = tickers) or None
    if the download fails (pytrends is unofficial and often rate-limited).
    We NEVER make up numbers when the download fails.
    """
    if cache_path and Path(cache_path).exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if set(keywords).issubset(cached.columns):
            print(f"[trends] using cached real data from {cache_path}")
            return cached[list(keywords)]

    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("[trends] pytrends not installed - skipping overlay")
        return None

    timeframe = timeframe.replace("{today}", date.today().isoformat())
    client = TrendReq(hl="en-US", tz=0, timeout=(10, 30))
    series = {}
    for ticker, kw in keywords.items():
        df = None
        for attempt in range(1, 4):  # a few polite retries
            try:
                # One keyword per request, so each series has its own 0-100 scale.
                client.build_payload([kw], timeframe=timeframe)
                df = client.interest_over_time()
                break
            except Exception as exc:  # rate limit (HTTP 429), network, API change
                print(f"[trends] attempt {attempt} failed for '{kw}': {exc}")
                time.sleep(pause_seconds * 5 * attempt)
        if df is None:
            print("[trends] giving up - no data will be invented")
            return None
        if df is None or df.empty:
            print(f"[trends] no data for '{kw}'")
            return None
        if "isPartial" in df.columns:
            df = df[~df["isPartial"].astype(bool)]  # drop the unfinished month
        series[ticker] = df[kw].astype(float)
        time.sleep(pause_seconds)

    out = pd.DataFrame(series)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path)
        print(f"[trends] saved real data to {cache_path}")
    return out


def attention_ratio(monthly: pd.DataFrame, short_months: int = 3, long_months: int = 12) -> pd.DataFrame:
    """ratio_m = mean(last `short` months) / mean(the `long` months before that).

    Uses only months <= m. Ratio > 1 means attention is rising.
    Baselines below 1 (almost no searches) give NaN -> no tilt.
    """
    recent = monthly.rolling(short_months, min_periods=short_months).mean()
    baseline = monthly.shift(short_months).rolling(long_months, min_periods=long_months).mean()
    baseline = baseline.where(baseline >= 1.0)
    return recent / baseline


def ratio_to_multiplier(ratio: pd.DataFrame, threshold: float = 0.2, tilt: float = 0.25) -> pd.DataFrame:
    """Rising attention -> 1 + tilt, fading -> 1 - tilt, otherwise 1 (incl. NaN)."""
    mult = pd.DataFrame(1.0, index=ratio.index, columns=ratio.columns)
    mult[ratio > 1.0 + threshold] = 1.0 + tilt
    mult[ratio < 1.0 - threshold] = 1.0 - tilt
    return mult


def your_turn_contrarian(ratio: pd.DataFrame, threshold: float = 0.2, tilt: float = 0.25) -> pd.DataFrame:
    """YOUR TURN (optional exercise #2): the opposite hypothesis.

    Maybe a search SPIKE means panic or a crowded trade, so we should hold
    LESS. Return 1 - tilt when ratio > 1 + threshold, 1 + tilt when
    ratio < 1 - threshold, else 1.0. Then try it in run.py and compare.
    """
    raise NotImplementedError("YOUR TURN: implement the contrarian tilt")


def multipliers_on_signal_dates(
    monthly_mult: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    lag_months: int = 1,
) -> pd.DataFrame:
    """For each month-end signal date, pick the multiplier from `lag_months` earlier.

    Google labels a month by its first day (2020-03-01 = all of March).
    On the signal date 2020-03-31 with lag 1 we may only use February's row.
    """
    rows = {}
    for t in signal_dates:
        usable_month = (t.to_period("M") - lag_months).to_timestamp()
        avail = monthly_mult.loc[:usable_month]
        rows[t] = avail.iloc[-1] if len(avail) else pd.Series(1.0, index=monthly_mult.columns)
    return pd.DataFrame(rows).T.fillna(1.0)
