"""Metric formulas, checked against values computed by hand."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from algobot.backtest.metrics import (
    compute_metrics,
    drawdown_series,
    format_metrics,
    infer_periods_per_year,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
)


def curve(values, freq="B"):
    return pd.Series(
        [float(v) for v in values],
        index=pd.date_range("2020-01-01", periods=len(values), freq=freq),
    )


def test_max_drawdown_measures_peak_to_trough():
    equity = curve([100, 120, 90, 110])
    assert max_drawdown(equity) == pytest.approx(-0.25)  # 120 -> 90
    assert drawdown_series(equity).iloc[1] == pytest.approx(0.0)


def test_max_drawdown_duration_counts_bars_underwater():
    equity = curve([100, 90, 95, 105, 100, 101, 110])
    # Two separate underwater runs: bars 1-2 and bars 4-5.
    assert max_drawdown_duration(equity) == 2


def test_no_drawdown_on_a_monotonic_curve():
    equity = curve([100, 101, 102, 103])
    assert max_drawdown(equity) == pytest.approx(0.0)
    assert max_drawdown_duration(equity) == 0


def test_sharpe_matches_the_definition():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(returns, 252) == pytest.approx(expected)


def test_sharpe_is_zero_without_variance_or_data():
    assert sharpe_ratio(pd.Series([0.01, 0.01, 0.01]), 252) == 0.0
    assert sharpe_ratio(pd.Series([0.01]), 252) == 0.0


def test_sortino_only_penalises_downside():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    downside = returns[returns < 0]
    expected = returns.mean() / math.sqrt((downside**2).mean()) * math.sqrt(252)
    assert sortino_ratio(returns, 252) == pytest.approx(expected)
    assert sortino_ratio(pd.Series([0.01, 0.02, 0.03]), 252) == math.inf


def test_periods_per_year_inferred_from_spacing():
    assert infer_periods_per_year(curve(range(10)).index) == pytest.approx(252, rel=0.35)
    weekly = curve(range(10), freq="W-FRI").index
    assert infer_periods_per_year(weekly) == pytest.approx(36, rel=0.1)


def test_cagr_compounds_over_the_measured_span():
    index = pd.DatetimeIndex(["2020-01-01", "2022-01-01"])
    equity = pd.Series([100_000.0, 121_000.0], index=index)
    m = compute_metrics(equity)
    assert m.total_return == pytest.approx(0.21)
    assert m.cagr == pytest.approx(0.10, abs=0.001)  # 21% over ~2 years


def test_trade_statistics():
    trades = pd.DataFrame(
        {
            "pnl": [100.0, -50.0, 200.0, -25.0],
            "bars_held": [5, 3, 8, 2],
        }
    )
    m = compute_metrics(curve([100_000, 100_225]), trades)

    assert m.num_trades == 4
    assert m.win_rate == pytest.approx(0.5)
    assert m.profit_factor == pytest.approx(300 / 75)
    assert m.expectancy == pytest.approx(56.25)
    assert m.avg_win == pytest.approx(150.0)
    assert m.avg_loss == pytest.approx(-37.5)
    assert m.best_trade == pytest.approx(200.0)
    assert m.worst_trade == pytest.approx(-50.0)
    assert m.avg_bars_held == pytest.approx(4.5)


def test_profit_factor_is_infinite_without_losses():
    trades = pd.DataFrame({"pnl": [10.0, 20.0], "bars_held": [1, 1]})
    assert math.isinf(compute_metrics(curve([100, 130]), trades).profit_factor)


def test_exposure_is_the_fraction_of_bars_holding_something():
    equity = curve([100, 101, 102, 103])
    positions = pd.Series([0.0, 10.0, 10.0, 0.0], index=equity.index)
    assert compute_metrics(equity, exposure=positions).exposure == pytest.approx(0.5)


def test_empty_equity_yields_neutral_metrics():
    m = compute_metrics(pd.Series(dtype=float))
    assert m.num_trades == 0 and m.total_return == 0.0 and m.start is None


def test_report_renders_all_rows():
    m = compute_metrics(curve([100_000, 105_000, 103_000]))
    report = format_metrics(m, "Test run")
    for label in ("Total return", "Sharpe", "Max drawdown", "Profit factor", "Costs paid"):
        assert label in report
    assert "Test run" in report


def test_report_handles_infinities():
    trades = pd.DataFrame({"pnl": [10.0, 20.0], "bars_held": [1, 1]})
    report = format_metrics(compute_metrics(curve([100, 130]), trades))
    assert "inf" in report
    assert "nan" not in report.lower()
