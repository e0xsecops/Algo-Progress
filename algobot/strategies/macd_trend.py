"""MACD signal-line crossover.

A smoother trend follower than a raw EMA cross: the MACD line is already a
difference of two EMAs, so crossing its own signal line reacts to a change in
*momentum* rather than to price touching a level.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from algobot.indicators import ema, macd
from algobot.strategies.base import Strategy


class MacdTrendStrategy(Strategy):
    """Long while the MACD line is above its signal line."""

    name: ClassVar[str] = "macd"
    params_schema: ClassVar[dict[str, Any]] = {
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "allow_short": False,
        "trend_filter": 0,
        "min_histogram": 0.0,  # require this much separation before acting
    }

    def validate(self) -> None:
        if min(self.fast, self.slow, self.signal) < 1:
            raise ValueError("fast, slow and signal must be >= 1")
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be shorter than slow ({self.slow})")
        if self.min_histogram < 0:
            raise ValueError("min_histogram must be non-negative")

    @property
    def warmup(self) -> int:
        return max(self.slow + self.signal, self.trend_filter or 0)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = macd(df["close"], self.fast, self.slow, self.signal)
        if self.trend_filter:
            out["trend"] = ema(df["close"], self.trend_filter)
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ind = self.indicators(df)
        histogram = ind["histogram"]

        bullish = histogram > self.min_histogram
        bearish = histogram < -self.min_histogram

        signal = pd.Series(
            np.where(bullish, 1.0, np.where(bearish & self.allow_short, -1.0, 0.0)),
            index=df.index,
            dtype=float,
        )

        if self.trend_filter:
            above = df["close"] > ind["trend"]
            signal = signal.where(~((signal > 0) & ~above), 0.0)
            signal = signal.where(~((signal < 0) & above), 0.0)
            signal = signal.where(ind["trend"].notna(), 0.0)

        return signal.where(ind["signal"].notna(), 0.0)
