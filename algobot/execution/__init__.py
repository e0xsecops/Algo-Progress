"""Order execution. Today a simulated paper broker; the same interface is what
a live adapter (CCXT, Alpaca, Kite) would implement."""

from algobot.execution.broker import Fill, PaperBroker, Trade

__all__ = ["Fill", "PaperBroker", "Trade"]
