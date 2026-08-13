"""algobot - a backtest-first algorithmic trading framework.

The package is split so that every stage of the pipeline can be swapped out
independently:

    data        loading and normalising OHLCV bars
    strategies  turning bars into a desired position (-1, 0, +1)
    risk        turning a desired position into a quantity, plus stops
    execution   simulating fills, commission, slippage and the trade ledger
    backtest    driving the whole loop bar by bar and scoring the result
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
