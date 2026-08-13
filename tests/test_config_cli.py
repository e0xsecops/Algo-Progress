"""Config loading, override precedence and the command line entry points."""

from __future__ import annotations

import pandas as pd
import pytest

from algobot.cli import _coerce, _parse_params, main
from algobot.config import Config, apply_overrides, load_config

SAMPLE = "examples/sample_bars.csv"


def write_config(tmp_path, body: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return str(path)


# -- config ------------------------------------------------------------------


def test_defaults_when_no_file_given():
    config = load_config(None)
    assert config.data.symbol == "SPY"
    assert config.backtest.fill == "next_open"
    assert config.risk.atr_period == 14


def test_repository_config_file_parses():
    config = load_config("config.yaml")
    assert config.strategy.name == "ema_cross"
    assert config.strategy.params["fast"] == 20


def test_partial_config_keeps_defaults_for_the_rest(tmp_path):
    config = load_config(write_config(tmp_path, "data:\n  symbol: MSFT\n"))
    assert config.data.symbol == "MSFT"
    assert config.data.source == "yahoo"
    assert config.backtest.initial_cash == 100_000.0


def test_unknown_keys_and_sections_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown config section"):
        load_config(write_config(tmp_path, "trading:\n  x: 1\n"))
    with pytest.raises(ValueError, match="unknown key"):
        load_config(write_config(tmp_path, "data:\n  ticker: MSFT\n"))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_invalid_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="fill must be"):
        load_config(write_config(tmp_path, "backtest:\n  fill: whenever\n"))
    with pytest.raises(ValueError, match="risk_per_trade"):
        load_config(write_config(tmp_path, "risk:\n  risk_per_trade: 5\n"))
    with pytest.raises(ValueError, match="initial_cash"):
        load_config(write_config(tmp_path, "backtest:\n  initial_cash: 0\n"))


def test_overrides_apply_only_where_provided():
    config = apply_overrides(Config(), symbol="AAPL", initial_cash=50_000.0, atr_period=None)

    assert config.data.symbol == "AAPL"
    assert config.backtest.initial_cash == 50_000.0
    assert config.risk.atr_period == 14  # None means "not provided"
    assert config.data.source == "yahoo"


def test_overrides_accept_qualified_names():
    config = apply_overrides(Config(), data_symbol="TSLA", strategy_name="buy_hold")
    assert config.data.symbol == "TSLA"
    assert config.strategy.name == "buy_hold"


def test_unmatched_override_is_an_error():
    with pytest.raises(ValueError, match="does not match any config field"):
        apply_overrides(Config(), leverage=3)


# -- CLI helpers -------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10", 10),
        ("2.5", 2.5),
        ("true", True),
        ("False", False),
        ("none", None),
        ("ema_cross", "ema_cross"),
    ],
)
def test_value_coercion(raw, expected):
    assert _coerce(raw) == expected


def test_param_parsing():
    assert _parse_params(["fast=10", "allow_short=true"]) == {"fast": 10, "allow_short": True}
    with pytest.raises(ValueError, match="key=value"):
        _parse_params(["fast"])


# -- commands ----------------------------------------------------------------


def test_backtest_command_runs(capsys):
    code = main(["backtest", "--source", "csv", "--path", SAMPLE, "--symbol", "DEMO"])
    out = capsys.readouterr().out

    assert code == 0
    assert "Total return" in out and "Benchmark" in out and "Sizing:" in out


def test_backtest_command_exports_and_prints_extras(tmp_path, capsys):
    code = main(
        [
            "backtest",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--strategy",
            "ema_cross",
            "--param",
            "fast=5",
            "--param",
            "slow=20",
            "--param",
            "trend_filter=0",
            "--show-trades",
            "3",
            "--json",
            "--out",
            str(tmp_path / "results"),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "fast=5" in out and "Trades" in out and '"sharpe"' in out
    assert (tmp_path / "results" / "equity_curve.csv").exists()
    assert (tmp_path / "results" / "trades.csv").exists()


def test_cli_flags_override_the_config_file(capsys):
    main(
        [
            "backtest",
            "-c",
            "config.yaml",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--cash",
            "25000",
            "--param",
            "fast=8",
        ]
    )
    out = capsys.readouterr().out

    assert "25,000.00" in out
    assert "fast=8" in out
    assert "slow=50" in out  # untouched values still come from the file


def test_switching_strategy_drops_stale_config_params(capsys):
    # config.yaml's params belong to ema_cross; buy_hold accepts none of them.
    code = main(
        [
            "backtest",
            "-c",
            "config.yaml",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--strategy",
            "buy_hold",
        ]
    )
    assert code == 0
    assert "buy_hold" in capsys.readouterr().out


def test_optimize_command_ranks_results(capsys):
    code = main(
        [
            "optimize",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--fast",
            "5,10",
            "--slow",
            "20,50",
            "--objective",
            "sharpe",
            "--min-trades",
            "1",
            "--top",
            "3",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "Tested 4 combination(s)" in out
    assert "sharpe" in out and "max_dd_pct" in out
    # The ranking must point the user at the validation step, not stop at a winner.
    assert "walkforward" in out


def test_optimize_skips_invalid_combinations(capsys):
    main(
        [
            "optimize",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--fast",
            "10,60",
            "--slow",
            "20,50",
            "--min-trades",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert "skipped" in out  # fast=60 with slow=20 and slow=50 are both invalid


def test_optimize_hides_thin_results_unless_asked(capsys):
    args = [
        "optimize",
        "--source",
        "csv",
        "--path",
        SAMPLE,
        "--grid",
        "fast=5,10",
        "--grid",
        "slow=20,50",
        "--min-trades",
        "10000",
    ]
    main(args)
    assert "hid 4 with fewer than 10000 trades" in capsys.readouterr().out

    main(args + ["--include-thin"])
    assert "hid" not in capsys.readouterr().out


def test_optimize_requires_a_grid(capsys):
    code = main(["optimize", "--source", "csv", "--path", SAMPLE])
    assert code == 2
    assert "at least one --grid" in capsys.readouterr().err


def test_grid_with_no_values_is_rejected(capsys):
    code = main(["optimize", "--source", "csv", "--path", SAMPLE, "--grid", "fast="])
    assert code == 2
    assert "lists no values" in capsys.readouterr().err


def test_walkforward_command_reports_folds(capsys):
    code = main(
        [
            "walkforward",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--symbol",
            "DEMO",
            "--strategy",
            "macd",
            "--grid",
            "fast=8,12",
            "--grid",
            "slow=21,26",
            "--splits",
            "3",
            "--min-trades",
            "1",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "out-of-sample" in out
    assert "Walk-forward efficiency" in out
    assert "Parameter stability across folds" in out
    assert out.count("\n 1 ") or "fold" in out


def test_walkforward_exports(tmp_path, capsys):
    main(
        [
            "walkforward",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--strategy",
            "macd",
            "--grid",
            "fast=8,12",
            "--grid",
            "slow=26",
            "--splits",
            "2",
            "--min-trades",
            "1",
            "--out",
            str(tmp_path / "wf"),
        ]
    )
    capsys.readouterr()
    assert (tmp_path / "wf" / "oos_equity.csv").exists()
    assert (tmp_path / "wf" / "folds.csv").exists()


def test_montecarlo_command_reports_a_distribution(capsys):
    code = main(
        [
            "montecarlo",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--symbol",
            "DEMO",
            "--strategy",
            "macd",
            "--trials",
            "300",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "Monte Carlo robustness" in out
    assert "Probability of profit" in out


def test_montecarlo_returns_method_and_export(tmp_path, capsys):
    code = main(
        [
            "montecarlo",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--strategy",
            "macd",
            "--method",
            "returns",
            "--trials",
            "200",
            "--block",
            "5",
            "--out",
            str(tmp_path / "mc.csv"),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0 and "block=5" in out
    exported = pd.read_csv(tmp_path / "mc.csv")
    assert len(exported) == 200
    assert {"total_return", "final_equity", "max_drawdown"} == set(exported.columns)


def test_montecarlo_warns_on_a_thin_sample(capsys):
    main(
        [
            "montecarlo",
            "--source",
            "csv",
            "--path",
            SAMPLE,
            "--strategy",
            "donchian",
            "--trials",
            "200",
        ]
    )
    assert "Caution: only" in capsys.readouterr().out


def test_portfolio_command_combines_symbols(capsys):
    code = main(
        [
            "portfolio",
            "--source",
            "csv",
            "--path",
            "examples/portfolio",
            "--symbols",
            "ALFA,BETA,GAMA",
            "--strategy",
            "macd",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "Portfolio:" in out
    assert "Per-symbol contribution" in out
    assert "Sleeve return correlation" in out
    assert "Average pairwise correlation" in out


def test_portfolio_command_accepts_weights_and_exports(tmp_path, capsys):
    code = main(
        [
            "portfolio",
            "--source",
            "csv",
            "--path",
            "examples/portfolio",
            "--symbols",
            "ALFA,BETA",
            "--weights",
            "3,1",
            "--strategy",
            "macd",
            "--out",
            str(tmp_path / "pf"),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "75.0" in out  # ALFA normalised to 75% of capital
    assert (tmp_path / "pf" / "portfolio_equity.csv").exists()
    assert (tmp_path / "pf" / "contributions.csv").exists()


def test_portfolio_rejects_a_single_symbol_and_bad_weights(capsys):
    assert (
        main(["portfolio", "--source", "csv", "--path", "examples/portfolio", "--symbols", "ALFA"])
        == 2
    )
    assert "at least 2 symbols" in capsys.readouterr().err

    assert (
        main(
            [
                "portfolio",
                "--source",
                "csv",
                "--path",
                "examples/portfolio",
                "--symbols",
                "ALFA,BETA",
                "--weights",
                "1,2,3",
            ]
        )
        == 2
    )
    assert "one weight per symbol" in capsys.readouterr().err


def test_strategies_command_lists_parameters(capsys):
    assert main(["strategies"]) == 0
    out = capsys.readouterr().out
    assert "ema_cross" in out and "buy_hold" in out and "trend_filter" in out


def test_data_errors_surface_as_exit_code_2(capsys):
    code = main(["backtest", "--source", "csv", "--path", "does_not_exist.csv"])
    assert code == 2
    assert "error:" in capsys.readouterr().err
