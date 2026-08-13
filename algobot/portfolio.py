"""Multi-instrument backtests.

Capital is split into per-symbol sleeves that each run the strategy
independently, and the sleeve equity curves are summed into a portfolio curve.
That models the common real setup — one system, many markets, a fixed share of
the account each — and it is what makes diversification visible: the portfolio's
drawdown is normally shallower than the average sleeve's, because the sleeves
do not all lose at the same time.

What this deliberately does *not* model: a shared cash pool that lets one
sleeve borrow another's idle capital, cross-sectional selection (hold the
strongest N of M symbols), or rebalancing between sleeves. Each sleeve is
funded once and left alone, so a sleeve that doubles keeps trading at its own
larger size rather than being trimmed back to target weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from algobot.backtest import Backtester, BacktestResult
from algobot.backtest.metrics import Metrics, compute_metrics, format_metrics
from algobot.config import BacktestConfig, RiskConfig
from algobot.data import load
from algobot.risk import RiskManager
from algobot.strategies import Strategy, get_strategy

log = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    """The combined record, plus every sleeve that produced it."""

    equity: pd.Series
    sleeves: dict[str, BacktestResult] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    metrics: Metrics = field(default_factory=Metrics)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy: str = ""
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def symbols(self) -> list[str]:
        return list(self.sleeves)

    def sleeve_returns(self) -> pd.DataFrame:
        """Per-sleeve bar returns, aligned on a common index."""
        if not self.sleeves:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {symbol: run.equity for symbol, run in self.sleeves.items()}
        ).sort_index()
        return frame.ffill().pct_change().dropna(how="all")

    def correlation(self) -> pd.DataFrame:
        """Correlation of sleeve returns - the thing diversification depends on."""
        returns = self.sleeve_returns()
        if returns.empty or returns.shape[1] < 2:
            return pd.DataFrame()
        return returns.corr()

    def diversification_ratio(self) -> float:
        """Weighted average sleeve volatility divided by portfolio volatility.

        Above 1.0 means the sleeves are offsetting each other; at exactly 1.0
        they move as one instrument and the extra symbols bought nothing.
        """
        returns = self.sleeve_returns()
        if returns.empty or returns.shape[1] < 2:
            return 1.0
        weights = np.array([self.weights.get(c, 0.0) for c in returns.columns])
        weights = weights / weights.sum() if weights.sum() else weights
        weighted_vol = float((returns.std(ddof=1) * weights).sum())
        portfolio_vol = float((returns * weights).sum(axis=1).std(ddof=1))
        if portfolio_vol <= 0:
            return 1.0
        return weighted_vol / portfolio_vol

    def contributions(self) -> pd.DataFrame:
        """What each sleeve contributed, in currency and as a share of the total."""
        if not self.sleeves:
            return pd.DataFrame()
        rows = []
        for symbol, run in self.sleeves.items():
            m = run.metrics
            rows.append(
                {
                    "symbol": symbol,
                    "weight_pct": round(self.weights.get(symbol, 0.0) * 100, 1),
                    "allocated": round(m.initial_equity, 2),
                    "final": round(m.final_equity, 2),
                    "pnl": round(m.final_equity - m.initial_equity, 2),
                    "return_pct": round(m.total_return * 100, 2),
                    "sharpe": round(m.sharpe, 2),
                    "max_dd_pct": round(m.max_drawdown * 100, 2),
                    "trades": m.num_trades,
                }
            )
        frame = pd.DataFrame(rows).sort_values("pnl", ascending=False)
        # Contribution to the portfolio's total return, so the column sums to
        # that return. A share-of-net-profit column would explode whenever the
        # winners and losers nearly cancel.
        allocated = frame["allocated"].sum()
        frame["contribution_pct"] = (
            (frame["pnl"] / allocated * 100).round(2) if allocated else 0.0
        )
        return frame.reset_index(drop=True)

    def summary(self) -> str:
        title = f"Portfolio: {self.strategy} across {len(self.sleeves)} symbol(s)"
        out = format_metrics(self.metrics, title)

        worst = min(
            (r.metrics.max_drawdown for r in self.sleeves.values()), default=0.0
        )
        out += (
            f"\nDiversification ratio: {self.diversification_ratio():.2f} "
            "(>1 means the sleeves offset each other)"
            f"\nWorst single-sleeve drawdown: {worst * 100:,.2f}% "
            f"vs portfolio {self.metrics.max_drawdown * 100:,.2f}%"
        )
        if self.failures:
            out += "\nSkipped: " + ", ".join(f"{s} ({e})" for s, e in self.failures.items())
        return out


def run_portfolio(
    data: dict[str, pd.DataFrame],
    strategy: Strategy | str,
    *,
    strategy_params: dict | None = None,
    config: BacktestConfig | None = None,
    risk: RiskConfig | RiskManager | None = None,
    weights: dict[str, float] | None = None,
) -> PortfolioResult:
    """Run *strategy* over every symbol in *data* and combine the results.

    Each sleeve is funded with ``initial_cash * weight``, so the portfolio
    starts with exactly the configured capital however many symbols there are.
    """
    if not data:
        raise ValueError("no symbols to run")

    config = config or BacktestConfig()
    symbols = list(data)

    if weights is None:
        weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
    else:
        missing = set(symbols) - set(weights)
        if missing:
            raise ValueError(f"no weight given for {sorted(missing)}")
        total = sum(weights[s] for s in symbols)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        weights = {s: weights[s] / total for s in symbols}

    sleeves: dict[str, BacktestResult] = {}
    failures: dict[str, str] = {}
    trade_frames = []

    for symbol in symbols:
        allocation = config.initial_cash * weights[symbol]
        sleeve_config = BacktestConfig(
            initial_cash=allocation,
            commission_bps=config.commission_bps,
            slippage_bps=config.slippage_bps,
            fill=config.fill,
        )
        instance = (
            get_strategy(strategy, **(strategy_params or {}))
            if isinstance(strategy, str)
            else strategy
        )
        try:
            run = Backtester(sleeve_config, risk).run(
                data[symbol], instance, symbol=symbol, benchmark=False
            )
        except ValueError as exc:  # too few bars, missing columns, ...
            log.warning("skipping %s: %s", symbol, exc)
            failures[symbol] = str(exc)
            continue

        sleeves[symbol] = run
        if len(run.trades):
            frame = run.trades.copy()
            frame.insert(0, "symbol", symbol)
            trade_frames.append(frame)

    if not sleeves:
        raise ValueError(f"every symbol failed to backtest: {failures}")

    equity = _combine(sleeves, weights, config.initial_cash)
    positions = _combine_exposure(sleeves)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    name = next(iter(sleeves.values())).strategy
    return PortfolioResult(
        equity=equity,
        sleeves=sleeves,
        weights={s: weights[s] for s in sleeves},
        metrics=compute_metrics(
            equity,
            all_trades,
            exposure=positions,
            total_commission=sum(r.metrics.total_commission for r in sleeves.values()),
            total_slippage=sum(r.metrics.total_slippage for r in sleeves.values()),
        ),
        trades=all_trades,
        strategy=name,
        failures=failures,
    )


def _combine(
    sleeves: dict[str, BacktestResult],
    weights: dict[str, float],
    initial_cash: float,
) -> pd.Series:
    """Sum sleeve equity curves onto a shared calendar.

    Sleeves whose history starts late hold their allocation in cash until they
    begin, rather than being treated as missing - otherwise the portfolio would
    appear to start smaller than it was funded.
    """
    index = pd.DatetimeIndex([])
    for run in sleeves.values():
        index = index.union(run.equity.index)

    total = pd.Series(0.0, index=index)
    for symbol, run in sleeves.items():
        allocation = initial_cash * weights[symbol]
        aligned = run.equity.reindex(index).ffill().fillna(allocation)
        total += aligned
    total.name = "equity"
    return total


def _combine_exposure(sleeves: dict[str, BacktestResult]) -> pd.Series:
    """A bar counts as exposed if any sleeve held a position on it."""
    index = pd.DatetimeIndex([])
    for run in sleeves.values():
        index = index.union(run.positions.index)
    held = pd.Series(0.0, index=index)
    for run in sleeves.values():
        held += run.positions.reindex(index).fillna(0.0).abs()
    return held


def load_many(
    symbols: list[str],
    *,
    source: str = "yahoo",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    directory: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Load bars for several symbols, reporting failures instead of raising.

    With ``source='csv'`` the files are expected at ``<directory>/<SYMBOL>.csv``.
    Returns ``(data, failures)`` so one dead ticker does not sink the run.
    """
    from pathlib import Path

    from algobot.data import DataError

    data: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    for symbol in symbols:
        try:
            if source == "csv":
                if not directory:
                    raise DataError("source 'csv' needs --path pointing at a directory")
                path = Path(directory) / f"{symbol}.csv"
                data[symbol] = load(source="csv", path=path)
            else:
                data[symbol] = load(
                    source=source, symbol=symbol, start=start, end=end, interval=interval
                )
        except DataError as exc:
            log.warning("could not load %s: %s", symbol, exc)
            failures[symbol] = str(exc)

    if not data:
        raise DataError(f"no symbols could be loaded: {failures}")
    return data, failures
