"""A paper broker: fill simulation, cash accounting and the trade ledger.

Costs are charged the way a real venue charges them — slippage moves the fill
price against you, commission comes off cash on top — because a backtest that
ignores either one will happily "profit" from noise it can never capture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

Timestamp = Any  # pandas.Timestamp, kept loose so plain datetimes work too


@dataclass(frozen=True)
class Fill:
    """A single executed order."""

    timestamp: Timestamp
    quantity: float  # signed: positive buys, negative sells
    price: float  # price actually paid, slippage included
    reference_price: float  # price before slippage
    commission: float
    slippage_cost: float
    reason: str

    @property
    def side(self) -> str:
        return "buy" if self.quantity > 0 else "sell"

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price


@dataclass(frozen=True)
class Trade:
    """A round trip: one entry and the exit that flattened it."""

    entry_time: Timestamp
    exit_time: Timestamp
    direction: int  # +1 long, -1 short
    quantity: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    pnl: float  # net of commission
    return_pct: float  # net return on the entry notional
    bars_held: int
    exit_reason: str

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class _OpenLot:
    direction: int
    quantity: float
    price: float
    time: Timestamp
    bar_index: int
    commission: float  # entry commission, carried until the round trip closes


@dataclass
class PaperBroker:
    """Simulates fills against reference prices and tracks the resulting book."""

    initial_cash: float = 100_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 2.0

    cash: float = field(init=False)
    position: float = field(init=False, default=0.0)
    fills: list[Fill] = field(init=False, default_factory=list)
    trades: list[Trade] = field(init=False, default_factory=list)
    total_commission: float = field(init=False, default=0.0)
    total_slippage: float = field(init=False, default=0.0)

    _lot: _OpenLot | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.cash = float(self.initial_cash)

    # -- state ---------------------------------------------------------------

    @property
    def direction(self) -> int:
        """+1 long, -1 short, 0 flat."""
        if self.position > 0:
            return 1
        if self.position < 0:
            return -1
        return 0

    @property
    def entry_price(self) -> float | None:
        return self._lot.price if self._lot else None

    def equity(self, price: float) -> float:
        """Mark-to-market account value at *price*."""
        return self.cash + self.position * price

    # -- execution -----------------------------------------------------------

    def fill_price(self, reference_price: float, quantity: float) -> float:
        """Reference price moved against the trader by the slippage assumption."""
        edge = self.slippage_bps / 10_000.0
        return reference_price * (1.0 + edge if quantity > 0 else 1.0 - edge)

    def market_order(
        self,
        timestamp: Timestamp,
        quantity: float,
        reference_price: float,
        reason: str = "signal",
        bar_index: int = 0,
    ) -> Fill | None:
        """Execute a signed market order and update cash, position and ledger.

        A reversal (long to short, or vice versa) is a single order here: the
        open lot is closed and booked as a trade, and the remainder opens a new
        lot in the other direction.
        """
        if reference_price <= 0:
            raise ValueError("reference_price must be positive")
        if not math.isfinite(quantity) or abs(quantity) < 1e-12:
            return None

        price = self.fill_price(reference_price, quantity)
        notional = abs(quantity) * price
        commission = notional * self.commission_bps / 10_000.0
        slippage_cost = abs(quantity) * abs(price - reference_price)

        self.cash -= quantity * price + commission
        self.total_commission += commission
        self.total_slippage += slippage_cost

        self._book(timestamp, quantity, price, commission, reason, bar_index)

        # Snap to exactly flat so float dust never leaves a phantom position.
        if abs(self.position) < 1e-9:
            self.position = 0.0

        fill = Fill(
            timestamp=timestamp,
            quantity=quantity,
            price=price,
            reference_price=reference_price,
            commission=commission,
            slippage_cost=slippage_cost,
            reason=reason,
        )
        self.fills.append(fill)
        return fill

    def close(
        self,
        timestamp: Timestamp,
        reference_price: float,
        reason: str = "close",
        bar_index: int = 0,
    ) -> Fill | None:
        """Flatten any open position."""
        if self.position == 0.0:
            return None
        return self.market_order(timestamp, -self.position, reference_price, reason, bar_index)

    # -- ledger --------------------------------------------------------------

    def _book(
        self,
        timestamp: Timestamp,
        quantity: float,
        price: float,
        commission: float,
        reason: str,
        bar_index: int,
    ) -> None:
        lot = self._lot
        remaining = quantity

        if lot is not None and quantity * lot.direction < 0:
            closing = min(abs(quantity), lot.quantity)
            share = closing / lot.quantity
            entry_commission = lot.commission * share
            exit_commission = commission * (closing / abs(quantity))
            gross = (price - lot.price) * closing * lot.direction
            net = gross - entry_commission - exit_commission
            basis = lot.price * closing

            self.trades.append(
                Trade(
                    entry_time=lot.time,
                    exit_time=timestamp,
                    direction=lot.direction,
                    quantity=closing,
                    entry_price=lot.price,
                    exit_price=price,
                    gross_pnl=gross,
                    commission=entry_commission + exit_commission,
                    pnl=net,
                    return_pct=net / basis if basis else 0.0,
                    bars_held=max(0, bar_index - lot.bar_index),
                    exit_reason=reason,
                )
            )

            lot.quantity -= closing
            lot.commission -= entry_commission
            remaining = quantity + closing * lot.direction  # leftover flips direction
            if lot.quantity <= 1e-9:
                self._lot = None
                lot = None

        self.position += quantity

        if abs(remaining) > 1e-12:
            direction = 1 if remaining > 0 else -1
            if lot is None:
                self._lot = _OpenLot(
                    direction=direction,
                    quantity=abs(remaining),
                    price=price,
                    time=timestamp,
                    bar_index=bar_index,
                    commission=commission * (abs(remaining) / abs(quantity)),
                )
            else:
                # Adding to an existing lot: blend the entry price.
                total = lot.quantity + abs(remaining)
                lot.price = (lot.price * lot.quantity + price * abs(remaining)) / total
                lot.quantity = total
                lot.commission += commission * (abs(remaining) / abs(quantity))

    # -- reporting -----------------------------------------------------------

    def trades_frame(self) -> pd.DataFrame:
        """The closed-trade ledger as a DataFrame (empty frame if no trades)."""
        columns = [
            "entry_time",
            "exit_time",
            "direction",
            "quantity",
            "entry_price",
            "exit_price",
            "gross_pnl",
            "commission",
            "pnl",
            "return_pct",
            "bars_held",
            "exit_reason",
        ]
        if not self.trades:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([t.__dict__ for t in self.trades])[columns]

    def fills_frame(self) -> pd.DataFrame:
        columns = [
            "timestamp",
            "side",
            "quantity",
            "price",
            "reference_price",
            "commission",
            "slippage_cost",
            "reason",
        ]
        if not self.fills:
            return pd.DataFrame(columns=columns)
        rows = [{**f.__dict__, "side": f.side} for f in self.fills]
        return pd.DataFrame(rows)[columns]
