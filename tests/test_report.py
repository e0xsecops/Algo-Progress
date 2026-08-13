"""HTML reports: self-containment, escaping and chart generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.backtest import run_backtest
from algobot.config import BacktestConfig, RiskConfig
from algobot.data import load_synthetic
from algobot.portfolio import run_portfolio
from algobot.report import (
    build_html,
    drawdown_chart,
    equity_chart,
    monthly_table,
    trade_histogram,
    write_backtest_report,
    write_portfolio_report,
)
from algobot.strategies import get_strategy


@pytest.fixture(scope="module")
def result():
    return run_backtest(
        load_synthetic(periods=900, seed=77),
        get_strategy("macd"),
        BacktestConfig(),
        RiskConfig(),
        symbol="DEMO",
    )


@pytest.fixture(scope="module")
def document(result, tmp_path_factory):
    path = tmp_path_factory.mktemp("report") / "report.html"
    write_backtest_report(result, path)
    return path.read_text(encoding="utf-8")


# -- the property that makes a report portable -------------------------------


def test_report_makes_no_external_requests(document):
    """A report must open correctly on a machine with no network at all."""
    for token in ("http://", "https://", "//cdn", "<script", "src=", "@import"):
        assert token not in document, f"report references {token!r}"


def test_report_is_a_complete_document(document):
    assert document.startswith("<!doctype html>")
    assert document.rstrip().endswith("</html>")
    assert "<title>" in document and "<style>" in document
    assert document.count("<svg") >= 2


def test_report_adapts_to_dark_mode(document):
    assert "prefers-color-scheme:dark" in document


# -- content -----------------------------------------------------------------


def test_report_contains_every_section(document):
    for heading in (
        "PERFORMANCE",
        "Equity curve",
        "Drawdown",
        "Monthly returns",
        "Details",
        "Trades",
    ):
        assert heading.lower() in document.lower()


def test_report_shows_the_headline_numbers(result, document):
    assert f"{result.metrics.total_return * 100:,.2f}%" in document
    assert f"{result.metrics.sharpe:.2f}" in document
    assert f"{result.metrics.num_trades:,}" in document


def test_report_is_written_to_disk(result, tmp_path):
    path = write_backtest_report(result, tmp_path / "nested" / "out.html")
    assert path.exists() and path.stat().st_size > 5_000


def test_custom_title_is_used(result, tmp_path):
    path = write_backtest_report(result, tmp_path / "t.html", title="Q3 review")
    assert "<title>Q3 review</title>" in path.read_text()


# -- escaping ----------------------------------------------------------------


def test_untrusted_text_is_escaped():
    equity = pd.Series(
        [100.0, 110.0, 105.0],
        index=pd.date_range("2022-01-01", periods=3, freq="B"),
    )
    from algobot.backtest.metrics import compute_metrics

    document = build_html(
        title="<script>alert(1)</script>",
        subtitle="a & b <img onerror=x>",
        metrics=compute_metrics(equity),
        equity=equity,
    )

    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;" in document
    assert "a &amp; b" in document


# -- charts ------------------------------------------------------------------


def test_equity_chart_draws_both_series(result):
    svg = equity_chart(result.equity, result.benchmark_equity)
    assert 'class="equity"' in svg and 'class="benchmark"' in svg
    assert "buy &amp; hold" in svg


def test_equity_chart_without_a_benchmark(result):
    svg = equity_chart(result.equity)
    assert 'class="benchmark"' not in svg


def test_charts_handle_empty_input():
    empty = pd.Series(dtype=float)
    assert "No equity curve" in equity_chart(empty)
    assert drawdown_chart(empty) == ""
    assert monthly_table(empty) == ""
    assert trade_histogram(pd.DataFrame()) == ""
    assert trade_histogram(None) == ""


def test_long_series_are_downsampled():
    """A decade of intraday bars must not produce a megabyte of SVG path."""
    index = pd.date_range("2010-01-01", periods=60_000, freq="h")
    equity = pd.Series(np.linspace(100_000, 150_000, len(index)), index=index)

    svg = equity_chart(equity)
    assert svg.count(" L") < 2_000
    assert len(svg) < 120_000


def test_downsampling_keeps_the_final_point():
    index = pd.date_range("2020-01-01", periods=5_000, freq="D")
    equity = pd.Series(np.arange(5_000, dtype=float) + 100_000, index=index)
    svg = equity_chart(equity)
    # The last date label comes from the last retained point.
    assert str(index[-1].date()) in svg


def test_monthly_table_covers_every_year(result):
    table = monthly_table(result.equity)
    years = {ts.year for ts in result.equity.index}
    for year in years:
        assert f"<th>{year}</th>" in table
    assert table.count("<th>Jan</th>") == 1


def test_trade_histogram_separates_wins_from_losses(result):
    svg = trade_histogram(result.trades)
    assert 'class="loss"' in svg and 'class="win"' in svg


def test_drawdown_chart_is_filled(result):
    svg = drawdown_chart(result.equity)
    assert "dd-area" in svg and "dd-line" in svg
    assert svg.rstrip().count("Z") >= 1  # the area path is closed


# -- portfolio ---------------------------------------------------------------


def test_portfolio_report_renders(tmp_path):
    market = {
        symbol: load_synthetic(symbol, periods=500, seed=400 + i)
        for i, symbol in enumerate(["ALFA", "BETA"])
    }
    portfolio = run_portfolio(market, "macd", config=BacktestConfig(), risk=RiskConfig())

    path = write_portfolio_report(portfolio, tmp_path / "pf.html")
    document = path.read_text()

    assert "Per-symbol contribution" in document
    assert "Sleeve correlation" in document
    assert "Diversification ratio" in document
    assert "http" not in document
    # The trades table keeps its P&L colouring despite the extra symbol column.
    assert 'class="pnl"' in document
