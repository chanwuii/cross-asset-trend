"""Download and cache daily ETF prices.

We use *adjusted* close prices (dividends and splits folded in), so a
simple percentage change of the price series is the total return an
investor would have earned.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_path: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted closes: rows = dates, columns = tickers.

    If `cache_path` exists (and `refresh` is False) we read the CSV instead
    of hitting the internet again. This keeps runs fast and reproducible.
    """
    if cache_path and Path(cache_path).exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns):
            return cached[tickers]

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data - check internet / tickers")

    prices = raw["Close"]
    if isinstance(prices, pd.Series):  # happens when only one ticker
        prices = prices.to_frame(tickers[0])
    prices = prices[tickers].sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    # Drop days where *every* ticker is missing (e.g. holidays).
    prices = prices.dropna(how="all")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns r_t = P_t / P_{t-1} - 1.

    A ticker that has not started trading yet stays NaN. We do NOT
    forward-fill returns, because that would invent data.
    """
    return prices.pct_change(fill_method=None)
