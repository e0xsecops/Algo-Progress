"""RSI mean reversion — buy panic, sell the recovery.

The opposite bet to a trend follower: it assumes a sharp move away from the
mean is an overreaction that will snap back. It wins often and loses big, which
is exactly the profile a stop-loss exists to cap.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from algobot.indicators import hold_between, rsi, sma
from algobot.strategies.base import Strategy


class RsiReversionStrategy(Strategy):
    """Buy oversold weakness, exit once RSI recovers to the midline."""

    name: ClassVar[str] = "rsi_reversion"
    params_schema: ClassVar[dict[str, Any]] = {
        "period": 14,
        "oversold": 30,
        "overbought": 70,
        "exit_level": 50,
        "allow_short": False,
        "trend_filter": 0,  # 0 disables; otherwise only buy above the SMA
    }

    def validate(self) -> None:
        if self.period < 2:
            raise ValueError("period must be >= 2")
        if not 0 < self.oversold < self.exit_level < self.overbought < 100:
            raise ValueError(
                "levels must satisfy 0 < oversold < exit_level < overbought < 100; "
                f"got {self.oversold}, {self.exit_level}, {self.overbought}"
            )

    @property
    def warmup(self) -> int:
        return max(self.period, self.trend_filter or 0)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["rsi"] = rsi(df["close"], self.period)
        if self.trend_filter:
            out["trend"] = sma(df["close"], self.trend_filter)
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ind = self.indicators(df)
        value = ind["rsi"]

        longs = hold_between(value < self.oversold, value >= self.exit_level, 1.0)

        signal = longs
        if self.allow_short:
            shorts = hold_between(value > self.overbought, value <= self.exit_level, -1.0)
            # Both sides are mutually exclusive by construction; sum combines them.
            signal = longs + shorts

        if self.trend_filter:
            above = df["close"] > ind["trend"]
            signal = signal.where(~((signal > 0) & ~above), 0.0)
            signal = signal.where(~((signal < 0) & above), 0.0)
            signal = signal.where(ind["trend"].notna(), 0.0)

        return signal.where(value.notna(), 0.0)
