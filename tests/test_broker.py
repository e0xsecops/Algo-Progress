"""Cash accounting, cost handling and the trade ledger."""

from __future__ import annotations

import pytest

from algobot.execution import PaperBroker


def make_broker(**kwargs) -> PaperBroker:
    defaults = {"initial_cash": 100_000.0, "commission_bps": 10.0, "slippage_bps": 5.0}
    return PaperBroker(**{**defaults, **kwargs})


def test_buy_debits_cash_including_costs():
    b = make_broker()
    fill = b.market_order("t0", 100, 100.0, "entry", 0)

    assert fill.price == pytest.approx(100.0 * 1.0005)  # slippage against the buyer
    assert fill.commission == pytest.approx(100 * fill.price * 0.001)
    assert b.cash == pytest.approx(100_000.0 - 100 * fill.price - fill.commission)
    assert b.position == 100
    assert b.direction == 1


def test_sell_credits_cash_and_slippage_works_against_the_seller():
    b = make_broker()
    fill = b.market_order("t0", -100, 100.0, "short", 0)

    assert fill.price == pytest.approx(100.0 * 0.9995)
    assert b.cash == pytest.approx(100_000.0 + 100 * fill.price - fill.commission)
    assert b.position == -100
    assert b.direction == -1


def test_zero_cost_round_trip_pnl_is_exact():
    b = PaperBroker(initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    b.market_order("t0", 10, 100.0, "entry", 0)
    b.close("t1", 110.0, "exit", 3)

    trade = b.trades[0]
    assert trade.gross_pnl == pytest.approx(100.0)
    assert trade.pnl == pytest.approx(100.0)
    assert trade.return_pct == pytest.approx(0.1)
    assert trade.bars_held == 3
    assert trade.is_win
    assert b.cash == pytest.approx(10_100.0)
    assert b.position == 0


def test_short_round_trip_profits_when_price_falls():
    b = PaperBroker(initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    b.market_order("t0", -10, 100.0, "entry", 0)
    b.close("t1", 90.0, "exit", 2)

    trade = b.trades[0]
    assert trade.direction == -1
    assert trade.pnl == pytest.approx(100.0)
    assert b.cash == pytest.approx(10_100.0)


def test_commission_is_charged_on_both_legs():
    b = PaperBroker(initial_cash=10_000.0, commission_bps=10.0, slippage_bps=0.0)
    b.market_order("t0", 10, 100.0, "entry", 0)
    b.close("t1", 100.0, "exit", 1)

    trade = b.trades[0]
    assert trade.gross_pnl == pytest.approx(0.0)
    assert trade.commission == pytest.approx(2.0)  # 1.0 per 1,000 notional leg
    assert trade.pnl == pytest.approx(-2.0)
    assert b.cash == pytest.approx(9_998.0)


def test_reversal_books_the_old_trade_and_opens_the_new_one():
    b = PaperBroker(initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0)
    b.market_order("t0", 100, 100.0, "entry", 0)
    b.market_order("t1", -150, 110.0, "reverse", 5)

    assert len(b.trades) == 1
    trade = b.trades[0]
    assert trade.direction == 1
    assert trade.quantity == pytest.approx(100)
    assert trade.pnl == pytest.approx(1_000.0)

    assert b.position == pytest.approx(-50)
    assert b.direction == -1
    assert b.entry_price == pytest.approx(110.0)


def test_scaling_in_blends_the_entry_price():
    b = PaperBroker(initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0)
    b.market_order("t0", 100, 100.0, "entry", 0)
    b.market_order("t1", 100, 120.0, "add", 1)

    assert b.position == pytest.approx(200)
    assert b.entry_price == pytest.approx(110.0)
    assert not b.trades  # nothing closed yet


def test_partial_close_books_only_the_closed_portion():
    b = PaperBroker(initial_cash=100_000.0, commission_bps=0.0, slippage_bps=0.0)
    b.market_order("t0", 100, 100.0, "entry", 0)
    b.market_order("t1", -40, 110.0, "trim", 2)

    assert len(b.trades) == 1
    assert b.trades[0].quantity == pytest.approx(40)
    assert b.trades[0].pnl == pytest.approx(400.0)
    assert b.position == pytest.approx(60)


def test_cash_reconciles_with_the_trade_ledger():
    """Final cash, once flat, must equal starting cash plus every booked P&L."""
    b = make_broker()
    for qty, price, i in [(100, 100.0, 0), (-150, 105.0, 3), (50, 102.0, 6)]:
        b.market_order(f"t{i}", qty, price, "trade", i)

    assert b.position == 0
    total_pnl = sum(t.pnl for t in b.trades)
    assert b.cash == pytest.approx(100_000.0 + total_pnl)


def test_equity_marks_open_positions_to_market():
    b = PaperBroker(initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    b.market_order("t0", 10, 100.0, "entry", 0)

    assert b.equity(100.0) == pytest.approx(10_000.0)
    assert b.equity(110.0) == pytest.approx(10_100.0)


def test_dust_orders_and_flat_closes_are_no_ops():
    b = make_broker()
    assert b.market_order("t0", 0.0, 100.0) is None
    assert b.close("t0", 100.0) is None
    assert b.cash == pytest.approx(100_000.0)
    assert not b.fills


def test_ledger_frames_have_stable_columns_when_empty():
    b = make_broker()
    assert list(b.trades_frame().columns)[:3] == ["entry_time", "exit_time", "direction"]
    assert "side" in b.fills_frame().columns


def test_negative_reference_price_is_rejected():
    b = make_broker()
    with pytest.raises(ValueError):
        b.market_order("t0", 10, 0.0)
