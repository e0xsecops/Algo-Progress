"""Multi-instrument backtests: allocation, combination and diversification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.config import BacktestConfig, RiskConfig
from algobot.data import DataError, load_synthetic
from algobot.portfolio import load_many, run_portfolio

SYMBOLS = ["ALFA", "BETA", "GAMA"]


@pytest.fixture(scope="module")
def market() -> dict[str, pd.DataFrame]:
    return {
        symbol: load_synthetic(symbol, periods=600, seed=300 + i)
        for i, symbol in enumerate(SYMBOLS)
    }


@pytest.fixture(scope="module")
def portfolio(market):
    return run_portfolio(
        market,
        "macd",
        config=BacktestConfig(initial_cash=100_000.0),
        risk=RiskConfig(),
    )


# -- allocation --------------------------------------------------------------


def test_capital_is_split_equally_by_default(portfolio):
    assert portfolio.symbols == SYMBOLS
    assert all(w == pytest.approx(1 / 3) for w in portfolio.weights.values())
    for run in portfolio.sleeves.values():
        assert run.metrics.initial_equity == pytest.approx(100_000 / 3)


def test_portfolio_starts_with_exactly_the_configured_capital(portfolio):
    assert portfolio.equity.iloc[0] == pytest.approx(100_000.0)
    assert portfolio.metrics.initial_equity == pytest.approx(100_000.0)


def test_custom_weights_are_normalised(market):
    result = run_portfolio(market, "macd", weights={"ALFA": 2, "BETA": 1, "GAMA": 1})

    assert result.weights["ALFA"] == pytest.approx(0.5)
    assert result.weights["BETA"] == pytest.approx(0.25)
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert result.equity.iloc[0] == pytest.approx(100_000.0)


def test_missing_or_degenerate_weights_are_rejected(market):
    with pytest.raises(ValueError, match="no weight given"):
        run_portfolio(market, "macd", weights={"ALFA": 1.0})
    with pytest.raises(ValueError, match="positive"):
        run_portfolio(market, "macd", weights={s: 0.0 for s in SYMBOLS})


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="no symbols"):
        run_portfolio({}, "macd")


# -- combination -------------------------------------------------------------


def test_portfolio_equity_is_the_sum_of_its_sleeves(portfolio):
    total = sum(run.metrics.final_equity for run in portfolio.sleeves.values())
    assert portfolio.metrics.final_equity == pytest.approx(total)


def test_sleeves_with_different_histories_still_start_fully_funded():
    """A sleeve whose data starts late holds cash rather than going missing."""
    market = {
        "EARLY": load_synthetic("EARLY", start="2020-01-01", periods=400, seed=1),
        "LATE": load_synthetic("LATE", start="2021-01-01", periods=200, seed=2),
    }
    result = run_portfolio(market, "macd", config=BacktestConfig(initial_cash=100_000.0))

    assert result.equity.iloc[0] == pytest.approx(100_000.0)
    assert not result.equity.isna().any()
    assert result.equity.index.is_monotonic_increasing


def test_trades_are_tagged_with_their_symbol(portfolio):
    assert "symbol" in portfolio.trades.columns
    assert set(portfolio.trades["symbol"]) <= set(SYMBOLS)
    assert len(portfolio.trades) == sum(
        len(run.trades) for run in portfolio.sleeves.values()
    )


def test_costs_are_summed_across_sleeves(portfolio):
    assert portfolio.metrics.total_commission == pytest.approx(
        sum(r.metrics.total_commission for r in portfolio.sleeves.values())
    )


def test_a_failing_symbol_is_reported_not_fatal(market):
    broken = dict(market)
    broken["STUB"] = market["ALFA"].iloc[:1]  # too short to backtest

    result = run_portfolio(broken, "macd")

    assert "STUB" in result.failures
    assert set(result.symbols) == set(SYMBOLS)


def test_every_symbol_failing_raises(market):
    with pytest.raises(ValueError, match="every symbol failed"):
        run_portfolio({s: df.iloc[:1] for s, df in market.items()}, "macd")


# -- diversification ---------------------------------------------------------


def test_diversification_shows_up_as_a_shallower_drawdown(portfolio):
    """Independent sleeves should not all be losing on the same days."""
    worst_sleeve = min(r.metrics.max_drawdown for r in portfolio.sleeves.values())
    assert portfolio.metrics.max_drawdown > worst_sleeve
    assert portfolio.diversification_ratio() > 1.0


def test_correlation_matrix_is_well_formed(portfolio):
    correlation = portfolio.correlation()

    assert list(correlation.columns) == SYMBOLS
    assert np.allclose(np.diag(correlation), 1.0)
    assert ((correlation >= -1.0) & (correlation <= 1.0)).all().all()
    pd.testing.assert_frame_equal(correlation, correlation.T)


def test_a_single_sleeve_has_no_diversification():
    single = {"ALFA": load_synthetic("ALFA", periods=300, seed=4)}
    result = run_portfolio(single, "macd")

    assert result.diversification_ratio() == 1.0
    assert result.correlation().empty


def test_identical_sleeves_diversify_nothing():
    df = load_synthetic("TWIN", periods=400, seed=5)
    result = run_portfolio({"TWIN_A": df, "TWIN_B": df.copy()}, "macd")

    assert result.correlation().iloc[0, 1] == pytest.approx(1.0)
    assert result.diversification_ratio() == pytest.approx(1.0, abs=0.01)


def test_contributions_sum_to_the_portfolio_return(portfolio):
    contributions = portfolio.contributions()

    assert set(contributions["symbol"]) == set(SYMBOLS)
    assert contributions["contribution_pct"].sum() == pytest.approx(
        portfolio.metrics.total_return * 100, abs=0.01
    )
    assert contributions["pnl"].is_monotonic_decreasing


def test_summary_renders(portfolio):
    summary = portfolio.summary()
    for label in ("Portfolio:", "Diversification ratio", "Worst single-sleeve drawdown"):
        assert label in summary


# -- loading -----------------------------------------------------------------


def test_load_many_reads_a_directory_of_csvs():
    data, failures = load_many(SYMBOLS, source="csv", directory="examples/portfolio")

    assert set(data) == set(SYMBOLS)
    assert not failures
    assert all(len(df) > 500 for df in data.values())


def test_load_many_reports_missing_symbols_without_failing():
    data, failures = load_many(
        ["ALFA", "NOPE"], source="csv", directory="examples/portfolio"
    )

    assert set(data) == {"ALFA"}
    assert "NOPE" in failures


def test_load_many_raises_only_when_everything_fails():
    with pytest.raises(DataError, match="no symbols could be loaded"):
        load_many(["NOPE", "ALSO_NOPE"], source="csv", directory="examples/portfolio")


def test_csv_source_requires_a_directory():
    with pytest.raises(DataError, match="no symbols could be loaded"):
        load_many(["ALFA"], source="csv", directory=None)
