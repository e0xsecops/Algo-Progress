"""Position sizing, protective stops and the drawdown kill-switch.

The strategy decides *direction*; this module decides *how much* — and, just as
importantly, when to stop trading altogether. Sizing is volatility-aware: risk
a fixed fraction of equity between the entry and the ATR-derived stop, so a
quiet instrument gets a bigger position than a violent one for the same risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from algobot.config import RiskConfig
from algobot.indicators import atr


@dataclass(frozen=True)
class StopLevels:
    """Protective exits for an open position, in price terms."""

    stop_loss: float | None
    take_profit: float | None


class RiskManager:
    """Turns a desired direction into a quantity, and guards the account."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    # -- inputs --------------------------------------------------------------

    def volatility(self, df: pd.DataFrame) -> pd.Series:
        """The ATR series this manager sizes and places stops from."""
        return atr(df, self.config.atr_period)

    @property
    def warmup(self) -> int:
        return self.config.atr_period if self.uses_atr else 0

    @property
    def uses_atr(self) -> bool:
        cfg = self.config
        return bool(cfg.stop_loss_atr or cfg.take_profit_atr or cfg.risk_per_trade)

    # -- sizing --------------------------------------------------------------

    def stop_distance(self, atr_value: float | None) -> float | None:
        """Distance from entry to the protective stop, in price terms."""
        if not self.config.stop_loss_atr:
            return None
        if atr_value is None or not np.isfinite(atr_value) or atr_value <= 0:
            return None
        return float(atr_value) * float(self.config.stop_loss_atr)

    def position_size(
        self,
        equity: float,
        price: float,
        atr_value: float | None = None,
    ) -> float:
        """Quantity to hold, in units, always capped by the exposure limit.

        With a stop in place the size comes from the risk budget
        (``equity * risk_per_trade / stop_distance``); without one it falls back
        to the notional cap alone.
        """
        cfg = self.config
        if equity <= 0 or price <= 0:
            return 0.0

        cap = equity * cfg.max_position_pct / price

        distance = self.stop_distance(atr_value)
        if distance and cfg.risk_per_trade > 0:
            risk_budget = equity * cfg.risk_per_trade
            size = risk_budget / distance
            return float(max(0.0, min(size, cap)))

        return float(max(0.0, cap))

    def stop_levels(self, entry_price: float, direction: int, atr_value: float | None) -> StopLevels:
        """Stop-loss and take-profit prices for a freshly opened position."""
        cfg = self.config
        if direction == 0 or atr_value is None or not np.isfinite(atr_value) or atr_value <= 0:
            return StopLevels(None, None)

        stop = None
        if cfg.stop_loss_atr:
            stop = entry_price - direction * atr_value * cfg.stop_loss_atr
            stop = max(stop, 0.0) if direction > 0 else stop

        target = None
        if cfg.take_profit_atr:
            target = entry_price + direction * atr_value * cfg.take_profit_atr

        return StopLevels(stop, target)

    # -- guards --------------------------------------------------------------

    def blocks_entry(self, atr_value: float | None) -> bool:
        """True when a stop is required but volatility is not yet measurable."""
        if not self.config.stop_loss_atr:
            return False
        return self.stop_distance(atr_value) is None

    def drawdown_breached(self, equity: float, peak_equity: float) -> bool:
        """True once the account has fallen further than the configured limit."""
        limit = self.config.max_drawdown_stop
        if not limit or peak_equity <= 0:
            return False
        return (equity / peak_equity - 1.0) <= -abs(limit)

    def describe(self) -> str:
        cfg = self.config
        parts = [
            f"risk/trade={cfg.risk_per_trade:.2%}",
            f"max_exposure={cfg.max_position_pct:.0%}",
            f"stop={cfg.stop_loss_atr}xATR({cfg.atr_period})" if cfg.stop_loss_atr else "stop=off",
        ]
        if cfg.take_profit_atr:
            parts.append(f"target={cfg.take_profit_atr}xATR")
        if cfg.max_drawdown_stop:
            parts.append(f"halt_at_dd={cfg.max_drawdown_stop:.0%}")
        return ", ".join(parts)
