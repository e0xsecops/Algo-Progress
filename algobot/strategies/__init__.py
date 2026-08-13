"""Strategy registry.

To add a strategy: subclass :class:`~algobot.strategies.base.Strategy`, give it
a unique ``name``, and register it below. The CLI, the optimizer, the
walk-forward validator and the config file all pick it up from there.

The shipped set deliberately spans both families, because they fail in
opposite conditions: trend followers (``ema_cross``, ``macd``, ``donchian``)
bleed in choppy ranges and pay for it with rare large winners, while mean
reverters (``rsi_reversion``, ``bollinger``) win often and lose badly when a
trend refuses to revert.
"""

from __future__ import annotations

from typing import Any

from algobot.strategies.base import Strategy
from algobot.strategies.bollinger_reversion import BollingerReversionStrategy
from algobot.strategies.buy_hold import BuyHoldStrategy
from algobot.strategies.donchian import DonchianBreakoutStrategy
from algobot.strategies.ema_cross import EmaCrossStrategy
from algobot.strategies.macd_trend import MacdTrendStrategy
from algobot.strategies.rsi_reversion import RsiReversionStrategy

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (
        EmaCrossStrategy,
        MacdTrendStrategy,
        DonchianBreakoutStrategy,
        RsiReversionStrategy,
        BollingerReversionStrategy,
        BuyHoldStrategy,
    )
}

#: Which family a strategy belongs to, for reporting and for choosing a peer group.
FAMILIES: dict[str, str] = {
    "ema_cross": "trend",
    "macd": "trend",
    "donchian": "trend",
    "rsi_reversion": "mean-reversion",
    "bollinger": "mean-reversion",
    "buy_hold": "benchmark",
}


def get_strategy(name: str, **params: Any) -> Strategy:
    """Instantiate a registered strategy by name."""
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}") from None
    return cls(**params)


def available() -> list[str]:
    return sorted(REGISTRY)


def register(cls: type[Strategy]) -> type[Strategy]:
    """Register a strategy class, usable as a decorator on third-party strategies."""
    if not issubclass(cls, Strategy):
        raise TypeError(f"{cls!r} is not a Strategy subclass")
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"strategy name {cls.name!r} is already registered")
    REGISTRY[cls.name] = cls
    return cls


__all__ = [
    "FAMILIES",
    "REGISTRY",
    "BollingerReversionStrategy",
    "BuyHoldStrategy",
    "DonchianBreakoutStrategy",
    "EmaCrossStrategy",
    "MacdTrendStrategy",
    "RsiReversionStrategy",
    "Strategy",
    "available",
    "get_strategy",
    "register",
]
