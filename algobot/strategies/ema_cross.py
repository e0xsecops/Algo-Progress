"""EMA crossover — the reference strategy.

Long while the fast EMA sits above the slow EMA, optionally short (or flat)
when it does not, and optionally gated by a long-term trend filter so the
strategy stops fighting a bear market.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from algobot.indicators import ema
from algobot.strategies.base import Strategy


class EmaCrossStrategy(Strategy):
    """Long while the fast EMA is above the slow EMA, gated by a trend filter."""

    name: ClassVar[str] = "ema_cross"
    params_schema: ClassVar[dict[str, Any]] = {
        "fast": 20,
        "slow": 50,
        "allow_short": False,
        "trend_filter": 200,  # 0 disables
    }

    def validate(self) -> None:
        if self.fast < 1 or self.slow < 1:
            raise ValueError("fast and slow must be >= 1")
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be shorter than slow ({self.slow})")
        if self.trend_filter and self.trend_filter < 1:
            raise ValueError("trend_filter must be 0 (disabled) or >= 1")

    @property
    def warmup(self) -> int:
        return max(self.slow, self.trend_filter or 0)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["ema_fast"] = ema(df["close"], self.fast)
        out["ema_slow"] = ema(df["close"], self.slow)
        if self.trend_filter:
            out["ema_trend"] = ema(df["close"], self.trend_filter)
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ind = self.indicators(df)
        bullish = ind["ema_fast"] > ind["ema_slow"]

        signal = pd.Series(
            np.where(bullish, 1.0, -1.0 if self.allow_short else 0.0),
            index=df.index,
            dtype=float,
        )

        if self.trend_filter:
            # Only take longs above the trend line and shorts below it.
            above_trend = df["close"] > ind["ema_trend"]
            signal = signal.where(~((signal > 0) & ~above_trend), 0.0)
            signal = signal.where(~((signal < 0) & above_trend), 0.0)
            signal = signal.where(ind["ema_trend"].notna(), 0.0)

        # No position until every input indicator is warmed up.
        ready = ind["ema_fast"].notna() & ind["ema_slow"].notna()
        return signal.where(ready, 0.0)
