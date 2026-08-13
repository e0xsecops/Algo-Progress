"""Donchian channel breakout — the classic Turtle system.

Buy when price makes a new N-bar high, exit when it makes a new M-bar low.
Nothing about it is clever; it survives because it never misses a large trend
and it cuts the small ones quickly. Expect a low win rate and a fat right tail.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from algobot.indicators import donchian, hold_between
from algobot.strategies.base import Strategy


class DonchianBreakoutStrategy(Strategy):
    """Enter on an N-bar breakout, exit on an M-bar breakout the other way."""

    name: ClassVar[str] = "donchian"
    params_schema: ClassVar[dict[str, Any]] = {
        "entry_period": 20,
        "exit_period": 10,
        "allow_short": False,
    }

    def validate(self) -> None:
        if self.entry_period < 2 or self.exit_period < 2:
            raise ValueError("entry_period and exit_period must be >= 2")
        if self.exit_period > self.entry_period:
            raise ValueError(
                f"exit_period ({self.exit_period}) should not exceed "
                f"entry_period ({self.entry_period}); a slower exit than entry "
                "never lets a position close on its own terms"
            )

    @property
    def warmup(self) -> int:
        return self.entry_period + 1

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        entry = donchian(df, self.entry_period)
        exit_ = donchian(df, self.exit_period)
        return pd.DataFrame(
            {
                "entry_upper": entry["upper"],
                "entry_lower": entry["lower"],
                "exit_upper": exit_["upper"],
                "exit_lower": exit_["lower"],
            }
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ind = self.indicators(df)
        close = df["close"]

        breakout_up = close > ind["entry_upper"]
        breakout_down = close < ind["entry_lower"]

        longs = hold_between(breakout_up, close < ind["exit_lower"], 1.0)
        signal = longs
        if self.allow_short:
            shorts = hold_between(breakout_down, close > ind["exit_upper"], -1.0)
            signal = longs + shorts

        ready = ind["entry_upper"].notna() & ind["exit_lower"].notna()
        return signal.where(ready, 0.0)
