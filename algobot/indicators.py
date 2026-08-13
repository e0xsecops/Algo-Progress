"""Vectorised technical indicators.

Every function here is *causal*: the value at bar ``t`` is computed only from
bars ``<= t``. That property is what keeps the backtester honest, so any new
indicator added to this module must preserve it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with the conventional ``2/(n+1)`` smoothing."""
    if period < 1:
        raise ValueError("period must be >= 1")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    if period < 1:
        raise ValueError("period must be >= 1")
    return series.rolling(period, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's true range: the largest of the three classic bar ranges."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range, smoothed the way Wilder defined it (alpha = 1/n)."""
    if period < 1:
        raise ValueError("period must be >= 1")
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's relative strength index, expressed 0-100."""
    if period < 1:
        raise ValueError("period must be >= 1")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # A flat-to-up window has no losses at all, which is an RSI of 100.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def rolling_max(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).max()


def rolling_min(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).min()


def rolling_std(series: pd.Series, period: int) -> pd.Series:
    """Sample standard deviation over a rolling window."""
    if period < 2:
        raise ValueError("period must be >= 2")
    return series.rolling(period, min_periods=period).std(ddof=1)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, its signal line, and the histogram between them."""
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")
    line = ema(series, fast) - ema(series, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=slow + signal - 1).mean()
    return pd.DataFrame({"macd": line, "signal": signal_line, "histogram": line - signal_line})


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Middle band (SMA) with volatility-scaled upper and lower bands."""
    if num_std <= 0:
        raise ValueError("num_std must be positive")
    middle = sma(series, period)
    spread = rolling_std(series, period) * num_std
    return pd.DataFrame({"middle": middle, "upper": middle + spread, "lower": middle - spread})


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Highest high and lowest low of the *previous* ``period`` bars.

    Shifted by one bar deliberately: a breakout must be measured against the
    channel as it stood before the current bar, otherwise the current bar's own
    high defines the level it is supposed to be breaking.
    """
    return pd.DataFrame(
        {
            "upper": rolling_max(df["high"], period).shift(1),
            "lower": rolling_min(df["low"], period).shift(1),
        }
    )


def hold_between(entry: pd.Series, exit_: pd.Series, value: float = 1.0) -> pd.Series:
    """Turn entry/exit triggers into a held position.

    Many strategies are stateful — enter on one condition, hold until a
    different one fires. Forward-filling the trigger states expresses that
    without a Python loop, and stays causal because ``ffill`` only ever
    propagates a value forward in time.
    """
    state = pd.Series(np.nan, index=entry.index, dtype=float)
    state[exit_.fillna(False).astype(bool)] = 0.0
    state[entry.fillna(False).astype(bool)] = value  # entry wins a tie
    return state.ffill().fillna(0.0)
