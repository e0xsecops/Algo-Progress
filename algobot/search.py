"""Parameter search shared by the optimizer and the walk-forward validator.

Kept in one place so an in-sample search during walk-forward selects parameters
exactly the way a manual ``optimize`` run would. If the two ever diverged, the
walk-forward result would stop describing the workflow it is meant to validate.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from algobot.backtest import Backtester
from algobot.backtest.metrics import Metrics
from algobot.config import BacktestConfig, RiskConfig
from algobot.risk import RiskManager
from algobot.strategies import get_strategy

log = logging.getLogger(__name__)

#: Metrics worth maximising, and whether a bigger number is better.
OBJECTIVES = {
    "sharpe": True,
    "sortino": True,
    "calmar": True,
    "total_return": True,
    "cagr": True,
    "profit_factor": True,
    "expectancy": True,
    "max_drawdown": True,  # least negative wins
}


@dataclass
class SearchRow:
    """One tested parameter combination and how it scored."""

    params: dict[str, Any]
    metrics: Metrics
    score: float
    eligible: bool = True
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        m = self.metrics
        return {
            **self.params,
            "score": round(self.score, 4),
            "return_pct": round(m.total_return * 100, 2),
            "cagr_pct": round(m.cagr * 100, 2),
            "sharpe": round(m.sharpe, 3),
            "sortino": round(m.sortino, 3),
            "max_dd_pct": round(m.max_drawdown * 100, 2),
            "calmar": round(m.calmar, 3),
            "trades": m.num_trades,
            "win_rate_pct": round(m.win_rate * 100, 1),
            "profit_factor": round(m.profit_factor, 3),
        }


@dataclass
class SearchResult:
    rows: list[SearchRow] = field(default_factory=list)
    objective: str = "sharpe"
    skipped: int = 0

    @property
    def eligible(self) -> list[SearchRow]:
        return [r for r in self.rows if r.eligible]

    @property
    def best(self) -> SearchRow | None:
        candidates = self.eligible or self.rows
        return max(candidates, key=lambda r: r.score) if candidates else None

    def table(self, include_ineligible: bool = False) -> pd.DataFrame:
        rows = self.rows if include_ineligible else (self.eligible or self.rows)
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame([r.as_dict() for r in rows])
        return frame.sort_values("score", ascending=False).reset_index(drop=True)


def expand_grid(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Every combination of the supplied parameter values."""
    if not grid:
        return
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo, strict=True))


def grid_size(grid: dict[str, list[Any]]) -> int:
    total = 1
    for values in grid.values():
        total *= max(1, len(values))
    return total if grid else 0


def grid_search(
    df: pd.DataFrame,
    strategy_name: str,
    grid: dict[str, list[Any]],
    *,
    base_params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    risk: RiskConfig | RiskManager | None = None,
    objective: str = "sharpe",
    min_trades: int = 5,
) -> SearchResult:
    """Test every combination in *grid* and score it by *objective*.

    Combinations producing fewer than *min_trades* round trips are marked
    ineligible rather than dropped: a parameter set that traded twice and got
    lucky will top a Sharpe ranking, and hiding it entirely makes the search
    look more decisive than it was.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {sorted(OBJECTIVES)}")

    backtester = Backtester(config, risk)
    result = SearchResult(objective=objective)

    for params in expand_grid(grid):
        merged = {**(base_params or {}), **params}
        try:
            strategy = get_strategy(strategy_name, **merged)
        except ValueError as exc:
            result.skipped += 1
            log.debug("skipping %s: %s", merged, exc)
            continue

        run = backtester.run(df, strategy, benchmark=False)
        metrics = run.metrics
        score = float(getattr(metrics, objective))
        if score != score or score in (float("inf"), float("-inf")):
            # inf/NaN scores (a perfect profit factor, say) beat everything by
            # accident rather than by merit.
            score = float("-inf")

        enough_trades = metrics.num_trades >= min_trades
        result.rows.append(
            SearchRow(
                params=params,
                metrics=metrics,
                score=score,
                eligible=enough_trades,
                reason="" if enough_trades else f"only {metrics.num_trades} trades",
            )
        )

    return result
