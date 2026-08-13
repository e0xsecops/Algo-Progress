"""Validation: the tools that try to disprove a backtest.

A backtest tells you what happened once, on data you already had. These do the
harder job of estimating what would happen on data you did not.
"""

from algobot.validation.montecarlo import MonteCarloResult, from_returns, from_trades, monte_carlo
from algobot.validation.walkforward import Fold, WalkForwardResult, walk_forward

__all__ = [
    "Fold",
    "MonteCarloResult",
    "WalkForwardResult",
    "from_returns",
    "from_trades",
    "monte_carlo",
    "walk_forward",
]
