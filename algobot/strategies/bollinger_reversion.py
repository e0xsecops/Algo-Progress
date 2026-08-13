"""Bollinger band mean reversion.

Fade a stretch to the lower band and take the position off at the middle band.
Unlike a fixed-percentage rule the bands widen with volatility, so the strategy
demands a bigger dislocation from a noisy instrument than from a calm one.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from algobot.indicators import bollinger, hold_between
from algobot.strategies.base import Strategy


class BollingerReversionStrategy(Strategy):
    """Buy the lower band, exit at the middle band."""

    name: ClassVar[str] = "bollinger"
    params_schema: ClassVar[dict[str, Any]] = {
        "period": 20,
        "num_std": 2.0,
        "allow_short": False,
    }

    def validate(self) -> None:
        if self.period < 2:
            raise ValueError("period must be >= 2")
        if self.num_std <= 0:
            raise ValueError("num_std must be positive")

    @property
    def warmup(self) -> int:
        return self.period

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return bollinger(df["close"], self.period, self.num_std)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ind = self.indicators(df)
        close = df["close"]

        longs = hold_between(close < ind["lower"], close >= ind["middle"], 1.0)
        signal = longs
        if self.allow_short:
            shorts = hold_between(close > ind["upper"], close <= ind["middle"], -1.0)
            signal = longs + shorts

        return signal.where(ind["middle"].notna(), 0.0)
