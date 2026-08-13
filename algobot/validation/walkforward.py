"""Walk-forward validation.

A single backtest over a period you also tuned on measures how well you fitted
the past, not how well the strategy works. Walk-forward fixes the ordering: for
each fold the parameters are chosen using *only* data before the test window,
then applied forward, unchanged, to data the search never saw. Stitching those
out-of-sample segments together gives an equity curve nobody optimised.

Two schemes, both standard:

    anchored   train windows all start at the beginning and grow
    rolling    train windows are a fixed length and slide forward

Anchored uses more history as it goes; rolling assumes the distant past stops
being relevant. Neither is universally right, so both are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from algobot.backtest import Backtester
from algobot.backtest.metrics import Metrics, compute_metrics, format_metrics
from algobot.config import BacktestConfig, RiskConfig
from algobot.risk import RiskManager
from algobot.search import grid_search
from algobot.strategies import get_strategy

log = logging.getLogger(__name__)


@dataclass
class Fold:
    """One train/test split and what it produced."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict[str, Any]
    train_metrics: Metrics
    test_metrics: Metrics
    test_equity: pd.Series
    trades: pd.DataFrame

    @property
    def train_bars(self) -> int:
        return int(self.train_metrics.num_trades)

    def as_row(self) -> dict[str, Any]:
        return {
            "fold": self.index,
            "train": f"{self.train_start.date()} to {self.train_end.date()}",
            "test": f"{self.test_start.date()} to {self.test_end.date()}",
            "params": ", ".join(f"{k}={v}" for k, v in self.params.items()),
            "is_return_pct": round(self.train_metrics.total_return * 100, 2),
            "oos_return_pct": round(self.test_metrics.total_return * 100, 2),
            "is_sharpe": round(self.train_metrics.sharpe, 2),
            "oos_sharpe": round(self.test_metrics.sharpe, 2),
            "oos_max_dd_pct": round(self.test_metrics.max_drawdown * 100, 2),
            "oos_trades": self.test_metrics.num_trades,
        }


@dataclass
class WalkForwardResult:
    """The stitched out-of-sample record across every fold."""

    folds: list[Fold] = field(default_factory=list)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    positions: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: Metrics = field(default_factory=Metrics)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    scheme: str = "anchored"
    objective: str = "sharpe"
    strategy: str = ""
    symbol: str = ""

    @property
    def efficiency(self) -> float:
        """Out-of-sample return as a fraction of in-sample return.

        Around 1.0 means the strategy travelled; near 0 or negative means the
        search was fitting noise. Anything far above 1.0 is usually luck, not a
        strategy that works better on unseen data.
        """
        in_sample = (
            np.mean([f.train_metrics.total_return for f in self.folds]) if self.folds else 0.0
        )
        out_sample = (
            np.mean([f.test_metrics.total_return for f in self.folds]) if self.folds else 0.0
        )
        if in_sample <= 0:
            return float("nan")
        return float(out_sample / in_sample)

    @property
    def consistency(self) -> float:
        """Fraction of folds that were profitable out of sample."""
        if not self.folds:
            return 0.0
        return float(np.mean([f.test_metrics.total_return > 0 for f in self.folds]))

    def folds_table(self) -> pd.DataFrame:
        return pd.DataFrame([f.as_row() for f in self.folds])

    def parameter_stability(self) -> pd.DataFrame:
        """How often the search settled on each parameter value.

        A strategy whose optimal parameters jump around every fold is being
        re-fitted, not validated - the numbers may still be positive, but they
        describe a different strategy in each window.
        """
        if not self.folds:
            return pd.DataFrame()
        rows = []
        keys = sorted({k for f in self.folds for k in f.params})
        for key in keys:
            values = [f.params.get(key) for f in self.folds]
            counts = pd.Series(values).value_counts()
            rows.append(
                {
                    "parameter": key,
                    "distinct_values": int(counts.size),
                    "most_common": counts.index[0],
                    "chosen_in_folds": f"{int(counts.iloc[0])}/{len(values)}",
                    "values": ", ".join(str(v) for v in values),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        title = f"Walk-forward ({self.scheme}) - {self.strategy}"
        if self.symbol:
            title += f" on {self.symbol}"
        out = format_metrics(self.metrics, title + " [out-of-sample only]")

        efficiency = self.efficiency
        efficiency_text = (
            "n/a (no in-sample profit)" if efficiency != efficiency else f"{efficiency:.2f}"
        )
        out += (
            f"\nFolds: {len(self.folds)} ({self.scheme}, selected by {self.objective})"
            f"\nWalk-forward efficiency: {efficiency_text}"
            f"  (out-of-sample return / in-sample return, ~1.0 is healthy)"
            f"\nProfitable folds: {self.consistency:.0%}"
        )
        return out


def _slice_positions(
    n: int, n_splits: int, train_size: float, scheme: str
) -> list[tuple[int, int, int, int]]:
    """Compute (train_start, train_end, test_start, test_end) index positions."""
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if not 0.1 <= train_size < 1.0:
        raise ValueError("train_size must be between 0.1 and 1.0 (exclusive)")

    # Carve the series into n_splits test windows, each preceded by its training data.
    initial_train = int(n * train_size)
    remaining = n - initial_train
    if remaining < n_splits:
        raise ValueError(
            f"not enough bars: {n} bars with train_size={train_size} leaves {remaining} "
            f"for {n_splits} test windows"
        )
    test_len = remaining // n_splits

    folds = []
    for i in range(n_splits):
        test_start = initial_train + i * test_len
        test_end = n if i == n_splits - 1 else test_start + test_len
        train_start = 0 if scheme == "anchored" else max(0, test_start - initial_train)
        folds.append((train_start, test_start, test_start, test_end))
    return folds


def walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    grid: dict[str, list[Any]],
    *,
    base_params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    risk: RiskConfig | RiskManager | None = None,
    n_splits: int = 5,
    train_size: float = 0.5,
    scheme: str = "anchored",
    objective: str = "sharpe",
    min_trades: int = 5,
    symbol: str = "",
) -> WalkForwardResult:
    """Run a walk-forward validation and return the stitched out-of-sample record."""
    if scheme not in ("anchored", "rolling"):
        raise ValueError(f"scheme must be 'anchored' or 'rolling', got {scheme!r}")
    if len(df) < 50:
        raise ValueError("walk-forward needs at least 50 bars to be meaningful")

    config = config or BacktestConfig()
    backtester = Backtester(config, risk)
    positions = _slice_positions(len(df), n_splits, train_size, scheme)

    folds: list[Fold] = []
    oos_returns: list[pd.Series] = []
    oos_trades: list[pd.DataFrame] = []
    oos_positions: list[pd.Series] = []
    commission = slippage = 0.0

    for i, (train_start, train_end, test_start, test_end) in enumerate(positions, start=1):
        train = df.iloc[train_start:train_end]
        log.info(
            "fold %d/%d: train %s..%s (%d bars), test %s..%s (%d bars)",
            i,
            len(positions),
            train.index[0].date(),
            train.index[-1].date(),
            len(train),
            df.index[test_start].date(),
            df.index[test_end - 1].date(),
            test_end - test_start,
        )

        search = grid_search(
            train,
            strategy_name,
            grid,
            base_params=base_params,
            config=config,
            risk=risk,
            objective=objective,
            min_trades=min_trades,
        )
        best = search.best
        if best is None:
            raise ValueError(f"fold {i}: no valid parameter combination in the grid")

        params = {**(base_params or {}), **best.params}
        strategy = get_strategy(strategy_name, **params)

        # Run from the start of the training window so indicators are warm by
        # the time the test window begins, then measure only the test window.
        run = backtester.run(df.iloc[train_start:test_end], strategy, benchmark=False)
        test_index = df.index[test_start:test_end]
        segment = run.equity.reindex(test_index)

        # Rebase each segment to the configured starting cash so folds are
        # comparable; compounding happens when they are stitched together.
        rebased = segment / segment.iloc[0] * config.initial_cash
        fold_trades = run.trades
        if len(fold_trades):
            fold_trades = fold_trades[fold_trades["exit_time"] >= test_index[0]].copy()
            fold_trades.insert(0, "fold", i)

        # Only the costs incurred inside the test window belong to the
        # out-of-sample record; the warmup run's fills are not part of it.
        if len(run.fills):
            test_fills = run.fills[run.fills["timestamp"] >= test_index[0]]
            commission += float(test_fills["commission"].sum())
            slippage += float(test_fills["slippage_cost"].sum())

        folds.append(
            Fold(
                index=i,
                train_start=train.index[0],
                train_end=train.index[-1],
                test_start=test_index[0],
                test_end=test_index[-1],
                params=best.params,
                train_metrics=best.metrics,
                test_metrics=compute_metrics(rebased, fold_trades),
                test_equity=rebased,
                trades=fold_trades,
            )
        )
        oos_returns.append(segment.pct_change().dropna())
        oos_positions.append(run.positions.reindex(test_index))
        if len(fold_trades):
            oos_trades.append(fold_trades)

    stitched = _stitch(oos_returns, config.initial_cash)
    all_trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    positions = pd.concat(oos_positions) if oos_positions else pd.Series(dtype=float)
    positions = positions[~positions.index.duplicated(keep="first")].sort_index()

    return WalkForwardResult(
        folds=folds,
        equity=stitched,
        positions=positions,
        metrics=compute_metrics(
            stitched,
            all_trades,
            exposure=positions,
            total_commission=commission,
            total_slippage=slippage,
        ),
        trades=all_trades,
        scheme=scheme,
        objective=objective,
        strategy=strategy_name,
        symbol=symbol,
    )


def _stitch(segments: list[pd.Series], initial_cash: float) -> pd.Series:
    """Chain per-fold return series into one compounding equity curve."""
    if not segments:
        return pd.Series(dtype=float)
    returns = pd.concat(segments)
    returns = returns[~returns.index.duplicated(keep="first")].sort_index()
    equity = initial_cash * (1.0 + returns).cumprod()
    # Prepend the starting point so the curve begins at the initial cash level.
    first = equity.index[0] - (
        equity.index[1] - equity.index[0] if len(equity) > 1 else pd.Timedelta(days=1)
    )
    return pd.concat([pd.Series([initial_cash], index=[first]), equity])
