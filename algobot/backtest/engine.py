"""The backtest loop.

The engine walks bars in order and never lets a decision touch data that would
not have existed when it was made. Concretely, in the default ``next_open``
mode the signal computed from bar ``t``'s close is executed at bar ``t+1``'s
open. Within a bar the sequence is:

    1. act on the pending signal (entry, exit or reversal)
    2. test the open position's stop-loss and take-profit against the bar range
    3. mark equity at the close and check the drawdown kill-switch

Gaps are filled pessimistically: a stop that gaps through fills at the open,
not at the stop price. The final bar opens no new position and liquidates any
open one at the close, so no result depends on a position the data cannot
close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from algobot.backtest.metrics import Metrics, compute_metrics, format_metrics
from algobot.config import BacktestConfig, RiskConfig
from algobot.execution.broker import PaperBroker
from algobot.risk.manager import RiskManager
from algobot.strategies.base import Strategy


@dataclass
class BacktestResult:
    """Everything a run produced: curves, ledgers and scores."""

    equity: pd.Series
    positions: pd.Series
    signals: pd.Series
    trades: pd.DataFrame
    fills: pd.DataFrame
    metrics: Metrics
    benchmark: Metrics | None = None
    benchmark_equity: pd.Series | None = None
    strategy: str = ""
    symbol: str = ""
    halted_at: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        title = f"{self.strategy} on {self.symbol}" if self.symbol else self.strategy
        out = format_metrics(self.metrics, title or "Backtest results")
        if self.benchmark is not None:
            b = self.benchmark
            edge = self.metrics.total_return - b.total_return
            out += (
                f"\nBenchmark (buy & hold, no costs): "
                f"return {b.total_return * 100:,.2f}%, "
                f"CAGR {b.cagr * 100:,.2f}%, "
                f"Sharpe {b.sharpe:.2f}, "
                f"max DD {b.max_drawdown * 100:,.2f}%"
                f"\nEdge vs benchmark: {edge * 100:+,.2f}% total return"
            )
        if self.halted_at is not None:
            out += f"\nWARNING: trading halted at {self.halted_at} (max drawdown breached)"
        return out

    def to_csv(self, directory: str) -> dict[str, str]:
        """Write equity curve, trades and fills to *directory*."""
        from pathlib import Path

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        curve = pd.DataFrame({"equity": self.equity, "position": self.positions})
        if self.benchmark_equity is not None:
            curve["benchmark"] = self.benchmark_equity
        written = {}
        for name, frame in (
            ("equity_curve.csv", curve),
            ("trades.csv", self.trades),
            ("fills.csv", self.fills),
        ):
            target = path / name
            frame.to_csv(target, index=name == "equity_curve.csv")
            written[name] = str(target)
        return written


class Backtester:
    """Drives a :class:`Strategy` over historical bars against a paper broker."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        risk: RiskManager | RiskConfig | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        if isinstance(risk, RiskManager):
            self.risk = risk
        else:
            self.risk = RiskManager(risk)

    def run(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        *,
        symbol: str = "",
        benchmark: bool = True,
    ) -> BacktestResult:
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"bars are missing column(s) {sorted(missing)}")
        if len(df) < 2:
            raise ValueError("need at least 2 bars to run a backtest")

        cfg = self.config
        risk = self.risk
        n = len(df)
        index = df.index

        signals = (
            strategy.generate_signals(df)
            .reindex(index)
            .astype(float)
            .fillna(0.0)
            .clip(-1.0, 1.0)
        )
        atr_values = (
            risk.volatility(df).to_numpy(dtype=float)
            if risk.uses_atr
            else np.full(n, np.nan)
        )

        open_ = df["open"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        signal_values = signals.to_numpy(dtype=float)

        broker = PaperBroker(
            initial_cash=cfg.initial_cash,
            commission_bps=cfg.commission_bps,
            slippage_bps=cfg.slippage_bps,
        )

        equity = np.empty(n, dtype=float)
        positions = np.zeros(n, dtype=float)
        stop_price: float | None = None
        take_profit: float | None = None
        peak_equity = float(cfg.initial_cash)
        halted_at: Any = None
        on_close = cfg.fill == "close"

        for i in range(n):
            ts = index[i]

            # No new position is opened on the final bar: there is no future
            # left to hold it through, so entering would only book a cost.
            last_bar = i == n - 1

            if halted_at is None:
                if on_close:
                    # Exits are tested first here, because they belong to the
                    # position carried into the bar rather than to this decision.
                    stop_price, take_profit = self._check_exits(
                        broker, ts, i, open_[i], high[i], low[i], stop_price, take_profit
                    )

                if not last_bar:
                    if on_close:
                        target = signal_values[i]
                        ref_price, atr_ref = close[i], atr_values[i]
                    else:
                        target = signal_values[i - 1] if i > 0 else 0.0
                        ref_price, atr_ref = open_[i], atr_values[i - 1] if i > 0 else np.nan

                    stop_price, take_profit = self._apply_target(
                        broker, risk, ts, i, target, ref_price, atr_ref, stop_price, take_profit
                    )

                if not on_close:
                    stop_price, take_profit = self._check_exits(
                        broker, ts, i, open_[i], high[i], low[i], stop_price, take_profit
                    )

            if last_bar and broker.position != 0.0:
                broker.close(ts, close[i], "end_of_data", i)
                stop_price = take_profit = None

            positions[i] = broker.position
            marked = broker.equity(close[i])

            if halted_at is None and risk.drawdown_breached(marked, peak_equity):
                broker.close(ts, close[i], "drawdown_halt", i)
                stop_price = take_profit = None
                positions[i] = broker.position
                marked = broker.equity(close[i])
                halted_at = ts

            equity[i] = marked
            peak_equity = max(peak_equity, marked)

        equity_series = pd.Series(equity, index=index, name="equity")
        position_series = pd.Series(positions, index=index, name="position")
        trades = broker.trades_frame()

        metrics = compute_metrics(
            equity_series,
            trades,
            exposure=position_series,
            total_commission=broker.total_commission,
            total_slippage=broker.total_slippage,
        )

        bench_metrics = bench_equity = None
        if benchmark:
            bench_equity = pd.Series(
                cfg.initial_cash * close / close[0], index=index, name="benchmark"
            )
            bench_metrics = compute_metrics(bench_equity)

        return BacktestResult(
            equity=equity_series,
            positions=position_series,
            signals=signals,
            trades=trades,
            fills=broker.fills_frame(),
            metrics=metrics,
            benchmark=bench_metrics,
            benchmark_equity=bench_equity,
            strategy=strategy.describe(),
            symbol=symbol,
            halted_at=halted_at,
            meta={
                "bars": n,
                "warmup": max(strategy.warmup, risk.warmup),
                "fill": cfg.fill,
                "risk": risk.describe(),
            },
        )

    # -- bar mechanics -------------------------------------------------------

    def _apply_target(
        self,
        broker: PaperBroker,
        risk: RiskManager,
        ts: Any,
        i: int,
        target: float,
        ref_price: float,
        atr_ref: float,
        stop_price: float | None,
        take_profit: float | None,
    ) -> tuple[float | None, float | None]:
        """Move the book towards the target direction at *ref_price*."""
        direction = int(np.sign(target))
        if direction == broker.direction:
            return stop_price, take_profit

        if direction == 0:
            broker.close(ts, ref_price, "signal", i)
            return None, None

        if risk.blocks_entry(None if not np.isfinite(atr_ref) else float(atr_ref)):
            # A stop is required but volatility is not measurable yet: stand aside.
            if broker.direction != 0:
                broker.close(ts, ref_price, "risk_block", i)
            return None, None

        atr_value = float(atr_ref) if np.isfinite(atr_ref) else None
        target_qty = direction * risk.position_size(broker.equity(ref_price), ref_price, atr_value)
        fill = broker.market_order(ts, target_qty - broker.position, ref_price, "signal", i)
        if fill is None:
            return stop_price, take_profit

        levels = risk.stop_levels(fill.price, direction, atr_value)
        return levels.stop_loss, levels.take_profit

    def _check_exits(
        self,
        broker: PaperBroker,
        ts: Any,
        i: int,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        stop_price: float | None,
        take_profit: float | None,
    ) -> tuple[float | None, float | None]:
        """Test stop-loss then take-profit against the bar's range.

        The stop is checked first: when a bar touches both levels there is no
        way to know which came first intrabar, so assume the unfavourable one.
        """
        direction = broker.direction
        if direction == 0:
            return None, None

        if stop_price is not None:
            hit = bar_low <= stop_price if direction > 0 else bar_high >= stop_price
            if hit:
                # A gap through the stop fills at the open, not at the stop.
                price = min(stop_price, bar_open) if direction > 0 else max(stop_price, bar_open)
                broker.close(ts, price, "stop_loss", i)
                return None, None

        if take_profit is not None:
            hit = bar_high >= take_profit if direction > 0 else bar_low <= take_profit
            if hit:
                # A limit order gapped in our favour fills at the better price.
                price = max(take_profit, bar_open) if direction > 0 else min(take_profit, bar_open)
                broker.close(ts, price, "take_profit", i)
                return None, None

        return stop_price, take_profit


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    risk: RiskManager | RiskConfig | None = None,
    *,
    symbol: str = "",
) -> BacktestResult:
    """Convenience wrapper around :class:`Backtester`."""
    return Backtester(config, risk).run(df, strategy, symbol=symbol)
