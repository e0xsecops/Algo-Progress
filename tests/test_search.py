"""Parameter search: grid expansion, scoring and the thin-sample guard."""

from __future__ import annotations

import pytest

from algobot.config import BacktestConfig, RiskConfig
from algobot.search import expand_grid, grid_search, grid_size


def test_expand_grid_produces_every_combination():
    combos = list(expand_grid({"a": [1, 2], "b": ["x", "y", "z"]}))

    assert len(combos) == 6
    assert {"a": 1, "b": "x"} in combos
    assert {"a": 2, "b": "z"} in combos
    assert len({tuple(sorted(c.items())) for c in combos}) == 6


def test_expand_grid_is_empty_for_an_empty_grid():
    assert list(expand_grid({})) == []
    assert grid_size({}) == 0
    assert grid_size({"a": [1, 2], "b": [1, 2, 3]}) == 6


def test_grid_search_scores_and_ranks(bars):
    result = grid_search(
        bars,
        "ema_cross",
        {"fast": [5, 10], "slow": [30, 60]},
        base_params={"trend_filter": 0},
        objective="sharpe",
        min_trades=1,
    )

    assert len(result.rows) == 4
    assert result.best is not None
    # The winner really does have the highest score of everything tested.
    assert result.best.score == max(r.score for r in result.rows)

    table = result.table()
    assert table["score"].is_monotonic_decreasing
    assert {"fast", "slow", "sharpe", "max_dd_pct", "trades"} <= set(table.columns)


def test_invalid_combinations_are_skipped_not_scored(bars):
    result = grid_search(
        bars,
        "ema_cross",
        {"fast": [10, 60], "slow": [30]},
        base_params={"trend_filter": 0},
        min_trades=1,
    )

    assert result.skipped == 1  # fast=60 with slow=30 is rejected by the strategy
    assert len(result.rows) == 1


def test_thin_results_are_flagged_rather_than_hidden(bars):
    result = grid_search(
        bars,
        "ema_cross",
        {"fast": [5], "slow": [30]},
        base_params={"trend_filter": 0},
        min_trades=10_000,  # nothing can clear this
    )

    assert len(result.rows) == 1
    assert result.eligible == []
    assert result.rows[0].eligible is False
    assert "trades" in result.rows[0].reason
    # With nothing eligible the search still reports its best guess.
    assert result.best is result.rows[0]


def test_eligible_results_outrank_thin_ones(bars):
    result = grid_search(
        bars,
        "ema_cross",
        {"fast": [5, 10], "slow": [30, 60]},
        base_params={"trend_filter": 0},
        min_trades=1,
    )
    result.rows[0].eligible = False
    result.rows[0].score = 1e9  # a spectacular but untradeable result

    assert result.best is not result.rows[0]


def test_infinite_scores_do_not_win(bars):
    result = grid_search(
        bars,
        "ema_cross",
        {"fast": [5, 10], "slow": [30, 60]},
        base_params={"trend_filter": 0},
        objective="profit_factor",
        min_trades=1,
    )
    # A loss-free run scores an infinite profit factor; it must not top the table.
    assert all(r.score != float("inf") for r in result.rows)


def test_unknown_objective_is_rejected(bars):
    with pytest.raises(ValueError, match="unknown objective"):
        grid_search(bars, "ema_cross", {"fast": [5]}, objective="vibes")


def test_search_respects_backtest_and_risk_settings(bars):
    cheap = grid_search(
        bars,
        "ema_cross",
        {"fast": [5], "slow": [30]},
        base_params={"trend_filter": 0},
        config=BacktestConfig(commission_bps=0.0, slippage_bps=0.0),
        risk=RiskConfig(),
        min_trades=1,
    )
    costly = grid_search(
        bars,
        "ema_cross",
        {"fast": [5], "slow": [30]},
        base_params={"trend_filter": 0},
        config=BacktestConfig(commission_bps=100.0, slippage_bps=50.0),
        risk=RiskConfig(),
        min_trades=1,
    )

    assert costly.rows[0].metrics.total_return < cheap.rows[0].metrics.total_return
