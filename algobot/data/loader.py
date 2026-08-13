"""Load OHLCV bars from CSV, Yahoo Finance, or a synthetic generator.

Whatever the source, the loader returns the same shape: a ``DataFrame`` indexed
by a sorted, unique ``DatetimeIndex`` with float columns
``open, high, low, close, volume``. The rest of the framework may assume that
contract holds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Column spellings seen in the wild, mapped onto our canonical names.
_ALIASES = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
    "adj close": "close",
    "adj_close": "close",
    "adjclose": "close",
    "price": "close",
    "last": "close",
}

_INDEX_CANDIDATES = ("date", "datetime", "timestamp", "time", "unnamed: 0")


class DataError(Exception):
    """Raised when input data cannot be coerced into the OHLCV contract."""


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex when it groups by ticker; drop that level."""
    if isinstance(df.columns, pd.MultiIndex):
        levels = [df.columns.get_level_values(i) for i in range(df.columns.nlevels)]
        # Keep the level that actually looks like OHLCV field names.
        best = max(levels, key=lambda lv: sum(str(x).strip().lower() in _known() for x in lv))
        df = df.copy()
        df.columns = best
    return df


def _known() -> set[str]:
    return set(OHLCV_COLUMNS) | set(_ALIASES)


def normalize(df: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    """Coerce an arbitrary price table into the canonical OHLCV frame."""
    if df is None or len(df) == 0:
        raise DataError(f"no rows returned for {symbol or 'input data'}")

    df = _flatten_columns(df)
    df = df.rename(columns=lambda c: str(c).strip().lower())
    df = df.rename(columns=_ALIASES)
    # A rename can collide (e.g. both "close" and "adj close" present); keep the first.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if not isinstance(df.index, pd.DatetimeIndex):
        for candidate in _INDEX_CANDIDATES:
            if candidate in df.columns:
                df = df.set_index(candidate)
                break
        try:
            df.index = pd.to_datetime(df.index, utc=False, errors="raise")
        except (ValueError, TypeError) as exc:
            raise DataError(
                "could not build a DatetimeIndex; expected a date/datetime index or a "
                f"column named one of {_INDEX_CANDIDATES}"
            ) from exc

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    df.index.name = "timestamp"

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise DataError(f"missing required column(s) {missing}; got {sorted(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    dropped = before - len(df)
    if dropped:
        log.warning("dropped %d bar(s) with missing or non-positive prices", dropped)

    if df.empty:
        raise DataError(f"no usable bars left for {symbol or 'input data'} after cleaning")

    df["volume"] = df["volume"].fillna(0.0)

    # Some feeds publish a high/low that does not bracket the open/close.
    # Widen the bar rather than discarding it, so stop simulation stays sane.
    high = df[["high", "open", "close"]].max(axis=1)
    low = df[["low", "open", "close"]].min(axis=1)
    inconsistent = int(((high != df["high"]) | (low != df["low"])).sum())
    if inconsistent:
        log.warning("repaired %d bar(s) whose high/low did not bracket open/close", inconsistent)
    df["high"] = high
    df["low"] = low

    return df


def load_csv(path: str | Path, **_: object) -> pd.DataFrame:
    """Load bars from a CSV file with a date column or date index."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"csv file not found: {path}")
    return normalize(pd.read_csv(path), symbol=path.name)


def load_yahoo(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    **_: object,
) -> pd.DataFrame:
    """Download split/dividend-adjusted bars from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DataError(
            "yfinance is not installed; run `pip install yfinance` or use --source csv"
        ) from exc

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or len(raw) == 0:
        raise DataError(
            f"Yahoo returned no data for {symbol!r} between {start} and {end} at {interval}. "
            "Check the ticker, the date range and that the interval is supported. "
            "If the log above shows a connection error, this machine cannot reach "
            "Yahoo - export the bars to CSV elsewhere and use --source csv."
        )
    return normalize(raw, symbol=symbol)


def load_synthetic(
    symbol: str = "SYNTH",
    start: str | None = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
    *,
    periods: int | None = None,
    seed: int = 7,
    annual_drift: float = 0.08,
    annual_vol: float = 0.20,
    start_price: float = 100.0,
    **_: object,
) -> pd.DataFrame:
    """Generate reproducible geometric-Brownian-motion bars.

    Useful for offline demos and for tests that must not depend on a network
    call. The intrabar high/low are drawn around the open-to-close path so the
    bars behave plausibly for stop simulation.
    """
    freq = {"1d": "B", "1wk": "W-FRI", "1h": "h"}.get(interval, "B")
    if periods is not None:
        index = pd.date_range(start=start or "2015-01-01", periods=periods, freq=freq)
    else:
        index = pd.date_range(start=start or "2015-01-01", end=end or "2024-12-31", freq=freq)
    n = len(index)
    if n < 2:
        raise DataError("synthetic series needs at least 2 bars")

    ppy = {"B": 252, "W-FRI": 52, "h": 252 * 7}[freq]
    dt = 1.0 / ppy
    rng = np.random.default_rng(seed)

    shocks = rng.normal(
        (annual_drift - 0.5 * annual_vol**2) * dt, annual_vol * np.sqrt(dt), size=n
    )
    close = start_price * np.exp(np.cumsum(shocks))
    open_ = np.concatenate([[start_price], close[:-1]])

    bar_vol = annual_vol * np.sqrt(dt) * np.abs(rng.normal(0.0, 1.0, size=n))
    high = np.maximum(open_, close) * (1.0 + bar_vol)
    low = np.minimum(open_, close) * (1.0 - bar_vol)
    volume = rng.lognormal(mean=13.0, sigma=0.4, size=n).round()

    return normalize(
        pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        ),
        symbol=symbol,
    )


def load(
    source: str = "yahoo",
    symbol: str = "SPY",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    path: str | Path | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Dispatch to the loader named by *source*."""
    source = source.lower()
    if source == "csv":
        if not path:
            raise DataError("source 'csv' requires a path")
        return load_csv(path)
    if source in ("yahoo", "yfinance"):
        return load_yahoo(symbol, start=start, end=end, interval=interval)
    if source == "synthetic":
        return load_synthetic(symbol, start=start, end=end, interval=interval, **kwargs)
    raise DataError(f"unknown data source {source!r}; expected csv, yahoo, or synthetic")
