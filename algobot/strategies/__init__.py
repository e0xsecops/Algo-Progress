"""Strategy registry.

To add a strategy: subclass :class:`~algobot.strategies.base.Strategy`, give it
a unique ``name``, and register it below. The CLI and config pick it up from
there — nothing else needs to change.
"""

from __future__ import annotations

from typing import Any

from algobot.strategies.base import Strategy
from algobot.strategies.buy_hold import BuyHoldStrategy
from algobot.strategies.ema_cross import EmaCrossStrategy

REGISTRY: dict[str, type[Strategy]] = {
    EmaCrossStrategy.name: EmaCrossStrategy,
    BuyHoldStrategy.name: BuyHoldStrategy,
}


def get_strategy(name: str, **params: Any) -> Strategy:
    """Instantiate a registered strategy by name."""
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}; available: {sorted(REGISTRY)}"
        ) from None
    return cls(**params)


def available() -> list[str]:
    return sorted(REGISTRY)


__all__ = [
    "REGISTRY",
    "BuyHoldStrategy",
    "EmaCrossStrategy",
    "Strategy",
    "available",
    "get_strategy",
]
