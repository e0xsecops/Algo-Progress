"""Walk-forward and Monte Carlo validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.backtest import run_backtest
from algobot.config import BacktestConfig, RiskConfig
from algobot.data import load_synthetic
from algobot.strategies import get_strategy
from algobot.validation import monte_carlo, walk_forward
from algobot.validation.montecarlo import from_returns, from_trades
from algobot.validation.walkforward import _slice_positions

GRID = {"fast": [5, 10], "slow": [20, 40]}
BASE = {"trend_filter": 0}


@pytest.fixture(scope="module")
def long_bars():
    return load_synthetic(periods=1_200, seed=31)


@pytest.fixture(scope="module")
def wf(long_bars):
    return walk_forward(
        long_bars,
        "ema_cross",
        GRID,
        base_params=BASE,
        n_splits=4,
        train_size=0.5,
        min_trades=1,
        symbol="SYN",
    )


# -- fold construction -------------------------------------------------------


def test_test_windows_are_contiguous_disjoint_and_cover_the_tail():
    positions = _slice_positions(1_000, 4, 0.5, "anchored")

    assert len(positions) == 4
    assert positions[0][2] == 500  # testing starts after the first training window
    assert positions[-1][3] == 1_000  # and runs to the last bar

    for (_, _, start, end), (_, _, next_start, _) in zip(positions, positions[1:]):
        assert end == next_start  # no gap, no overlap
        assert start < end


def test_training_never_reaches_into_its_own_test_window():
    """The whole point: parameters must be chosen without seeing the test data."""
    for scheme in ("anchored", "rolling"):
        for train_start, train_end, test_start, test_end in _slice_positions(800, 5, 0.4, scheme):
            assert train_start < train_end <= test_start < test_end


def test_anchored_windows_grow_and_rolling_windows_slide():
    anchored = _slice_positions(1_000, 4, 0.5, "anchored")
    rolling = _slice_positions(1_000, 4, 0.5, "rolling")

    assert all(start == 0 for start, _, _, _ in anchored)
    assert [end - start for start, end, _, _ in anchored] == sorted(
        end - start for start, end, _, _ in anchored
    )

    assert rolling[-1][0] > rolling[0][0]  # the window has moved forward
    lengths = {end - start for start, end, _, _ in rolling}
    assert len(lengths) == 1  # and kept a constant size


def test_impossible_splits_are_rejected():
    with pytest.raises(ValueError, match="not enough bars"):
        _slice_positions(100, 60, 0.9, "anchored")
    with pytest.raises(ValueError, match="n_splits"):
        _slice_positions(1_000, 0, 0.5, "anchored")
    with pytest.raises(ValueError, match="train_size"):
        _slice_positions(1_000, 4, 1.5, "anchored")


# -- walk-forward results ----------------------------------------------------


def test_walk_forward_produces_one_fold_per_split(wf):
    assert len(wf.folds) == 4
    assert [f.index for f in wf.folds] == [1, 2, 3, 4]
    assert all(f.params for f in wf.folds)


def test_folds_are_chronological_and_do_not_overlap(wf):
    for previous, current in zip(wf.folds, wf.folds[1:]):
        assert previous.test_end <= current.test_start
        assert previous.train_end <= previous.test_start


def test_out_of_sample_equity_starts_at_the_configured_cash(wf):
    assert wf.equity.iloc[0] == pytest.approx(100_000.0)
    assert wf.equity.index.is_monotonic_increasing
    assert wf.equity.index.is_unique
    assert not wf.equity.isna().any()


def test_stitched_equity_compounds_the_folds(wf):
    """The stitched curve must equal the folds' returns chained together."""
    expected = 100_000.0
    for fold in wf.folds:
        expected *= 1.0 + fold.test_metrics.total_return
    assert wf.metrics.final_equity == pytest.approx(expected, rel=1e-6)


def test_reported_costs_and_exposure_are_from_the_test_windows_only(wf):
    assert wf.metrics.total_commission > 0
    assert wf.metrics.total_slippage > 0
    assert 0.0 < wf.metrics.exposure <= 1.0
    assert len(wf.positions) == sum(len(f.test_equity) for f in wf.folds)


def test_consistency_and_efficiency_are_well_defined(wf):
    assert 0.0 <= wf.consistency <= 1.0
    efficiency = wf.efficiency
    assert isinstance(efficiency, float)  # NaN is allowed when in-sample lost money


def test_efficiency_is_not_a_number_without_in_sample_profit(wf):
    import copy

    broken = copy.copy(wf)
    broken.folds = []
    assert broken.efficiency != broken.efficiency  # NaN


def test_reports_render(wf):
    summary = wf.summary()
    assert "Walk-forward" in summary and "out-of-sample" in summary
    assert "Profitable folds" in summary

    folds = wf.folds_table()
    assert len(folds) == 4
    assert {"is_return_pct", "oos_return_pct", "oos_sharpe"} <= set(folds.columns)

    stability = wf.parameter_stability()
    assert set(stability["parameter"]) == {"fast", "slow"}
    assert (stability["distinct_values"] >= 1).all()


def test_rolling_scheme_runs_and_differs_from_anchored(long_bars, wf):
    rolling = walk_forward(
        long_bars,
        "ema_cross",
        GRID,
        base_params=BASE,
        n_splits=4,
        train_size=0.5,
        scheme="rolling",
        min_trades=1,
    )
    assert rolling.scheme == "rolling"
    assert len(rolling.folds) == 4
    # Same test windows, different training data, so the record is not identical.
    assert [f.test_start for f in rolling.folds] == [f.test_start for f in wf.folds]


def test_walk_forward_rejects_bad_input(long_bars):
    with pytest.raises(ValueError, match="scheme"):
        walk_forward(long_bars, "ema_cross", GRID, scheme="sideways")
    with pytest.raises(ValueError, match="at least 50 bars"):
        walk_forward(long_bars.iloc[:20], "ema_cross", GRID)


def test_walk_forward_is_deterministic(long_bars, wf):
    again = walk_forward(
        long_bars, "ema_cross", GRID, base_params=BASE, n_splits=4, train_size=0.5,
        min_trades=1, symbol="SYN",
    )
    pd.testing.assert_series_equal(wf.equity, again.equity)


# -- Monte Carlo -------------------------------------------------------------


@pytest.fixture(scope="module")
def backtest(long_bars):
    return run_backtest(
        long_bars,
        get_strategy("macd"),
        BacktestConfig(),
        RiskConfig(),
        symbol="SYN",
    )


def test_trade_resampling_produces_a_distribution(backtest):
    result = monte_carlo(backtest.equity, backtest.trades, method="trades", trials=500)

    assert result.trials == 500
    assert result.final_equity.shape == (500,)
    assert result.observations == len(backtest.trades)
    assert 0.0 <= result.probability_of_profit <= 1.0
    assert 0.0 <= result.probability_of_ruin <= 1.0
    assert (result.max_drawdowns <= 0).all()
    assert (result.final_equity >= 0).all()


def test_percentiles_are_ordered(backtest):
    table = monte_carlo(backtest.equity, backtest.trades, trials=500).percentiles()

    assert list(table["percentile"]) == ["p5", "p25", "p50", "p75", "p95"]
    assert table["total_return_pct"].is_monotonic_increasing
    assert table["final_equity"].is_monotonic_increasing
    # Deeper drawdowns sit at the low percentiles.
    assert table["max_drawdown_pct"].is_monotonic_increasing


def test_the_observed_result_lands_inside_its_own_distribution(backtest):
    result = monte_carlo(backtest.equity, backtest.trades, trials=2_000)
    assert 0.0 <= result.observed_return_percentile <= 100.0
    assert result.observed_return == pytest.approx(backtest.metrics.total_return)
    assert result.observed_max_drawdown == pytest.approx(backtest.metrics.max_drawdown)


def test_simulation_is_reproducible_for_a_given_seed(backtest):
    a = monte_carlo(backtest.equity, backtest.trades, trials=300, seed=42)
    b = monte_carlo(backtest.equity, backtest.trades, trials=300, seed=42)
    c = monte_carlo(backtest.equity, backtest.trades, trials=300, seed=43)

    np.testing.assert_array_equal(a.final_equity, b.final_equity)
    assert not np.array_equal(a.final_equity, c.final_equity)


def test_block_bootstrap_of_returns(backtest):
    result = monte_carlo(backtest.equity, method="returns", trials=500, block=10)

    assert result.observations == len(backtest.equity) - 1
    assert result.final_equity.shape == (500,)
    assert "block=10" in result.method


def test_both_methods_broadly_agree(backtest):
    by_trade = monte_carlo(backtest.equity, backtest.trades, method="trades", trials=3_000)
    by_return = monte_carlo(backtest.equity, method="returns", trials=3_000, block=10)

    # They resample different things, so they should land in the same
    # neighbourhood without being identical.
    assert abs(np.median(by_trade.total_returns) - np.median(by_return.total_returns)) < 0.05


def test_a_losing_system_rarely_simulates_a_profit():
    """Sanity anchor: consistently negative trades must not bootstrap into gains."""
    equity = pd.Series(
        np.linspace(100_000, 80_000, 200),
        index=pd.date_range("2022-01-01", periods=200, freq="B"),
    )
    trades = pd.DataFrame(
        {
            "entry_time": equity.index[:50],
            "pnl": np.full(50, -400.0),
            "bars_held": np.full(50, 3),
        }
    )
    result = from_trades(trades, equity, trials=500)

    assert result.probability_of_profit == 0.0
    assert result.total_returns.max() < 0


def test_empty_inputs_return_an_empty_simulation():
    equity = pd.Series(
        [100_000.0, 101_000.0],
        index=pd.date_range("2022-01-01", periods=2, freq="B"),
    )
    empty = from_trades(pd.DataFrame(), equity, trials=100)

    assert empty.total_returns.size == 0
    assert empty.probability_of_profit == 0.0
    assert "not enough data" in empty.summary()
    assert from_returns(pd.Series(dtype=float), trials=100).total_returns.size == 0


def test_unknown_method_is_rejected(backtest):
    with pytest.raises(ValueError, match="unknown method"):
        monte_carlo(backtest.equity, backtest.trades, method="crystal_ball")


def test_summary_renders(backtest):
    text = monte_carlo(backtest.equity, backtest.trades, trials=200).summary()
    for label in ("Monte Carlo robustness", "Probability of profit", "percentile"):
        assert label in text
