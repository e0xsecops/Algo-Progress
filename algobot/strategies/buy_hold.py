"""Buy and hold — the benchmark every strategy has to beat.

Reported alongside each backtest so an impressive-looking equity curve can be
compared against simply owning the asset over the same window.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from algobot.strategies.base import Strategy


class BuyHoldStrategy(Strategy):
    """Always long: the benchmark an active strategy has to beat."""

    name: ClassVar[str] = "buy_hold"
    params_schema: ClassVar[dict[str, Any]] = {}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index, dtype=float)
