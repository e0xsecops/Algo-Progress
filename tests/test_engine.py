"""Engine mechanics: execution timing, stops, guards and the lookahead property."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from algobot.backtest import Backtester, run_backtest
from algobot.config import BacktestConfig, RiskConfig
from algobot.strategies import get_strategy
from algobot.strategies.base import Strategy

NO_RISK_LIMITS = RiskConfig(
    risk_per_trade=0.0,
    max_position_pct=1.0,
    stop_loss_atr=None,
    take_profit_atr=None,
    max_drawdown_stop=None,
)
FREE = BacktestConfig(initial_cash=10_000.0, commission_bps=0.0, slippage_bps=0.0)


class ScriptedStrategy(Strategy):
    """Replays a fixed signal sequence so bar-level behaviour is exactly testable."""

    name = "scripted"
    params_schema = {"sequence": ()}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        seq = list(self.sequence)
        seq += [seq[-1] if seq else 0.0] * (len(df) - len(seq))
        return pd.Series(seq[: len(df)], index=df.index, dtype=float)


def flat_bars(n: int = 26, price: float = 100.0, spread: float = 1.0) -> pd.DataFrame:
    """Constant price with a fixed 2-point range, so ATR settles at exactly 2.0."""
    return pd.DataFrame(
        {
            "open": np.full(n, price),
            "high": np.full(n, price + spread),
            "low": np.full(n, price - spread),
            "close": np.full(n, price),
            "volume": np.full(n, 1_000.0),
        },
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )


def make_bars(prices, highs=None, lows=None, opens=None) -> pd.DataFrame:
    close = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": np.asarray(opens, dtype=float) if opens is not None else close,
            "high": np.asarray(highs, dtype=float) if highs is not None else close,
            "low": np.asarray(lows, dtype=float) if lows is not None else close,
            "close": close,
            "volume": 1_000.0,
        },
        index=pd.date_range("2022-01-01", periods=len(close), freq="B"),
    )


# -- execution timing --------------------------------------------------------


def test_signal_is_executed_at_the_next_bar_open():
    bars = make_bars(prices=[100, 101, 102, 103], opens=[100, 100.5, 101.5, 102.5])
    strategy = ScriptedStrategy(sequence=[1, 1, 1, 1])

    result = run_backtest(bars, strategy, FREE, NO_RISK_LIMITS)

    # Signal on bar 0 fills at bar 1's open, not bar 0's close.
    assert result.positions.iloc[0] == 0
    assert result.positions.iloc[1] > 0
    assert result.fills.iloc[0]["price"] == pytest.approx(100.5)
    assert result.fills.iloc[0]["timestamp"] == bars.index[1]


def test_close_fill_mode_trades_on_the_signal_bar():
    bars = make_bars(prices=[100, 101, 102, 103], opens=[100, 100.5, 101.5, 102.5])
    config = BacktestConfig(initial_cash=10_000.0, commission_bps=0, slippage_bps=0, fill="close")

    result = run_backtest(bars, ScriptedStrategy(sequence=[1, 1, 1, 1]), config, NO_RISK_LIMITS)

    assert result.positions.iloc[0] > 0
    assert result.fills.iloc[0]["price"] == pytest.approx(100.0)


def test_position_is_flat_on_the_final_bar():
    bars = make_bars(prices=np.linspace(100, 130, 40))
    result = run_backtest(bars, ScriptedStrategy(sequence=[1]), FREE, NO_RISK_LIMITS)

    assert result.positions.iloc[-1] == 0
    assert not result.trades.empty


def test_open_position_is_liquidated_at_the_final_close():
    bars = make_bars(prices=[100, 101, 102, 103, 104], opens=[100, 100, 101, 102, 103])
    result = run_backtest(bars, ScriptedStrategy(sequence=[1]), FREE, NO_RISK_LIMITS)

    final = result.trades.iloc[-1]
    assert final["exit_reason"] == "end_of_data"
    assert final["exit_price"] == pytest.approx(104.0)  # the close, not the open


def test_no_position_is_opened_on_the_final_bar():
    """Entering on the last bar could only book a cost, never a result."""
    bars = make_bars(prices=[100.0] * 6)
    result = run_backtest(bars, ScriptedStrategy(sequence=[0, 0, 0, 0, 1, 1]), FREE, NO_RISK_LIMITS)

    assert result.trades.empty
    assert (result.positions == 0).all()


def test_long_only_strategy_never_goes_short():
    bars = make_bars(prices=np.linspace(100, 60, 60))
    result = run_backtest(bars, ScriptedStrategy(sequence=[1, 0, 1, 0]), FREE, NO_RISK_LIMITS)
    assert (result.positions >= 0).all()


def test_short_signal_opens_a_short_position():
    bars = make_bars(prices=np.linspace(100, 80, 30))
    result = run_backtest(bars, ScriptedStrategy(sequence=[-1]), FREE, NO_RISK_LIMITS)

    assert (result.positions < 0).any()
    assert result.metrics.total_return > 0  # shorting a falling market pays


# -- costs and accounting ----------------------------------------------------


def test_equity_reflects_a_clean_round_trip():
    bars = make_bars(
        prices=[100, 100, 110, 110, 110],
        opens=[100, 100, 100, 110, 110],
    )
    result = run_backtest(bars, ScriptedStrategy(sequence=[0, 1, 0, 0, 0]), FREE, NO_RISK_LIMITS)

    # Buy at bar 2's open (100), sell at bar 3's open (110): 100 units, +1,000.
    assert result.trades.iloc[0]["entry_price"] == pytest.approx(100.0)
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(110.0)
    assert result.trades.iloc[0]["pnl"] == pytest.approx(1_000.0)
    assert result.metrics.final_equity == pytest.approx(11_000.0)


def test_costs_reduce_returns():
    bars = make_bars(prices=np.linspace(100, 140, 80))
    strategy = get_strategy("ema_cross", fast=3, slow=8, trend_filter=0)

    free = run_backtest(bars, strategy, FREE, NO_RISK_LIMITS)
    costly = run_backtest(
        bars,
        strategy,
        BacktestConfig(initial_cash=10_000.0, commission_bps=50.0, slippage_bps=25.0),
        NO_RISK_LIMITS,
    )

    assert costly.metrics.final_equity < free.metrics.final_equity
    assert costly.metrics.total_commission > 0
    assert costly.metrics.total_slippage > 0


def test_flat_strategy_leaves_equity_untouched():
    bars = make_bars(prices=np.linspace(100, 200, 50))
    result = run_backtest(bars, ScriptedStrategy(sequence=[0]), FREE, NO_RISK_LIMITS)

    assert (result.equity == 10_000.0).all()
    assert result.trades.empty
    assert result.metrics.exposure == 0.0


# -- stops, targets and guards ----------------------------------------------


# These three share a setup: 26 flat bars so ATR settles at exactly 2.0, entry
# signalled on bar 19 and filled at bar 20's open of 100, and one eventful bar
# at index 22 - late enough for the ATR to be warm, early enough that the
# engine's end-of-data liquidation does not pre-empt the exit under test.
ENTRY_SIGNAL = [0.0] * 19 + [1.0]
STOP_RISK = RiskConfig(
    risk_per_trade=0.01,
    stop_loss_atr=2.0,  # ATR 2.0 x 2 => stop 4 points away, at 96
    take_profit_atr=None,
    atr_period=14,
    max_drawdown_stop=None,
)


def test_stop_loss_exits_at_the_stop_price():
    bars = flat_bars()
    bars.iloc[22, bars.columns.get_loc("low")] = 80.0  # trades down through 96

    result = run_backtest(bars, ScriptedStrategy(sequence=ENTRY_SIGNAL), FREE, STOP_RISK)

    stop_trade = result.trades[result.trades["exit_reason"] == "stop_loss"].iloc[0]
    assert stop_trade["entry_price"] == pytest.approx(100.0)
    assert stop_trade["exit_price"] == pytest.approx(96.0)
    # 1% of 10,000 risked over a 4-point stop is 25 units, so exactly -100.
    assert stop_trade["quantity"] == pytest.approx(25.0)
    assert stop_trade["pnl"] == pytest.approx(-100.0)


def test_take_profit_exits_on_a_favourable_bar():
    bars = flat_bars()
    bars.iloc[22, bars.columns.get_loc("high")] = 130.0  # trades up through 106

    risk = dataclasses.replace(STOP_RISK, take_profit_atr=3.0)  # target 6 points away, at 106
    result = run_backtest(bars, ScriptedStrategy(sequence=ENTRY_SIGNAL), FREE, risk)

    target_trade = result.trades[result.trades["exit_reason"] == "take_profit"].iloc[0]
    assert target_trade["exit_price"] == pytest.approx(106.0)
    assert target_trade["pnl"] == pytest.approx(150.0)  # 25 units x 6 points


def test_gap_through_the_stop_fills_at_the_open():
    bars = flat_bars()
    for column, value in (("open", 70.0), ("high", 71.0), ("low", 69.0), ("close", 70.0)):
        bars.iloc[22, bars.columns.get_loc(column)] = value

    result = run_backtest(bars, ScriptedStrategy(sequence=ENTRY_SIGNAL), FREE, STOP_RISK)

    stop_trade = result.trades[result.trades["exit_reason"] == "stop_loss"].iloc[0]
    # Filled at the gapped-down open, not at the unreachable 96 stop.
    assert stop_trade["exit_price"] == pytest.approx(70.0)
    assert stop_trade["pnl"] == pytest.approx(-750.0)  # 25 units x 30 points


def test_drawdown_kill_switch_halts_trading():
    bars = make_bars(prices=np.concatenate([[100.0] * 5, np.linspace(100, 20, 45)]))
    risk = RiskConfig(
        risk_per_trade=0.0,
        max_position_pct=1.0,
        stop_loss_atr=None,
        take_profit_atr=None,
        max_drawdown_stop=0.2,
    )

    result = run_backtest(bars, ScriptedStrategy(sequence=[1]), FREE, risk)

    assert result.halted_at is not None
    assert result.metrics.max_drawdown > -0.45  # halted long before the full -80%
    after_halt = result.positions[result.positions.index > result.halted_at]
    assert (after_halt == 0).all()
    assert "drawdown_halt" in set(result.trades["exit_reason"])


def test_entry_is_blocked_while_atr_is_unmeasurable():
    bars = make_bars(prices=np.linspace(100, 120, 30))
    risk = RiskConfig(risk_per_trade=0.01, stop_loss_atr=2.0, atr_period=14)

    result = run_backtest(bars, ScriptedStrategy(sequence=[1]), FREE, risk)

    # Nothing is traded until the ATR has warmed up.
    assert (result.positions.iloc[:14] == 0).all()
    assert (result.positions.iloc[15:-1] != 0).any()


def test_atr_sizing_risks_the_configured_fraction():
    bars = make_bars(
        prices=[100.0] * 40,
        highs=[102.0] * 40,
        lows=[98.0] * 40,
        opens=[100.0] * 40,
    )
    risk = RiskConfig(
        risk_per_trade=0.02,
        max_position_pct=10.0,  # deliberately not the binding constraint
        stop_loss_atr=2.0,
        atr_period=14,
        max_drawdown_stop=None,
    )

    result = run_backtest(bars, ScriptedStrategy(sequence=[0] * 19 + [1]), FREE, risk)
    position = result.positions[result.positions != 0].iloc[0]

    # ATR settles at 4.0, so the stop sits 8.0 away and 2% of 10,000 buys 25 units.
    assert position == pytest.approx(25.0, rel=0.05)


# -- the property that matters ----------------------------------------------


def test_future_bars_cannot_change_past_results(bars):
    """Rewrite the tail of the data; every earlier trade must be identical.

    If this fails the engine is peeking at data it could not have had, and
    every backtest it has ever produced is worthless.
    """
    strategy = get_strategy("ema_cross", fast=10, slow=30, trend_filter=0)
    config = BacktestConfig(initial_cash=100_000.0)
    risk = RiskConfig(max_drawdown_stop=None)

    cut = len(bars) - 100
    original = run_backtest(bars, strategy, config, risk)

    tampered_bars = bars.copy()
    tampered_bars.iloc[cut:, :] *= 5.0
    tampered = run_backtest(tampered_bars, strategy, config, risk)

    pd.testing.assert_series_equal(original.equity.iloc[:cut], tampered.equity.iloc[:cut])
    pd.testing.assert_series_equal(original.positions.iloc[:cut], tampered.positions.iloc[:cut])

    early = original.trades[original.trades["exit_time"] < bars.index[cut]]
    tampered_early = tampered.trades[tampered.trades["exit_time"] < bars.index[cut]]
    pd.testing.assert_frame_equal(early, tampered_early)


def test_truncating_the_data_reproduces_the_same_history(bars):
    """A backtest over the first N bars must match the first N bars of a longer run.

    Same guarantee from the other direction: results depend only on the past.
    """
    strategy = get_strategy("ema_cross", fast=10, slow=30, trend_filter=0)
    config = BacktestConfig(initial_cash=100_000.0)
    risk = RiskConfig(max_drawdown_stop=None)
    cut = 400

    full = run_backtest(bars, strategy, config, risk)
    short = run_backtest(bars.iloc[:cut], strategy, config, risk)

    # The final bar is excluded: the short run liquidates there, the long one does not.
    pd.testing.assert_series_equal(full.equity.iloc[: cut - 1], short.equity.iloc[: cut - 1])


# -- interface ---------------------------------------------------------------


def test_result_summary_and_export(tmp_path, bars):
    result = run_backtest(bars, get_strategy("ema_cross", fast=10, slow=30), symbol="TEST")

    summary = result.summary()
    assert "TEST" in summary and "Sharpe" in summary and "Benchmark" in summary

    written = result.to_csv(str(tmp_path / "out"))
    assert len(written) == 3
    curve = pd.read_csv(tmp_path / "out" / "equity_curve.csv")
    assert {"equity", "position", "benchmark"} <= set(curve.columns)
    assert len(curve) == len(bars)


def test_benchmark_tracks_buy_and_hold(bars):
    result = run_backtest(bars, get_strategy("buy_hold"), BacktestConfig(), RiskConfig())
    expected = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1.0
    assert result.benchmark.total_return == pytest.approx(expected)


def test_engine_rejects_unusable_input():
    with pytest.raises(ValueError, match="at least 2 bars"):
        run_backtest(make_bars([100.0]), ScriptedStrategy(sequence=[1]))

    bad = make_bars([100.0, 101.0]).drop(columns=["high"])
    with pytest.raises(ValueError, match="missing column"):
        run_backtest(bad, ScriptedStrategy(sequence=[1]))


def test_backtester_accepts_a_risk_config_or_manager(bars):
    from algobot.risk import RiskManager

    strategy = get_strategy("ema_cross", fast=10, slow=30)
    a = Backtester(BacktestConfig(), RiskConfig(risk_per_trade=0.02)).run(bars, strategy)
    b = Backtester(BacktestConfig(), RiskManager(RiskConfig(risk_per_trade=0.02))).run(
        bars, strategy
    )
    assert a.metrics.final_equity == pytest.approx(b.metrics.final_equity)
