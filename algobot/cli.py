"""Command line interface.

python -m algobot backtest --source synthetic --symbol DEMO
python -m algobot backtest --symbol AAPL --start 2018-01-01 --out results/aapl
python -m algobot optimize --fast 5,10,20 --slow 50,100,200
python -m algobot strategies
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from algobot import __version__
from algobot.backtest import Backtester
from algobot.config import apply_overrides, load_config
from algobot.data import DataError, load
from algobot.portfolio import load_many, run_portfolio
from algobot.report import write_backtest_report, write_portfolio_report
from algobot.risk import RiskManager
from algobot.search import OBJECTIVES, grid_search, grid_size
from algobot.strategies import FAMILIES, available, get_strategy
from algobot.validation import monte_carlo, walk_forward

log = logging.getLogger("algobot")


def _int_list(raw: str) -> list[int]:
    return [int(part) for part in str(raw).replace(" ", "").split(",") if part]


def _parse_params(pairs: list[str] | None) -> dict[str, Any]:
    """Parse ``--param key=value`` pairs, coercing to bool/int/float where sensible."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--param expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = _coerce(value.strip())
    return out


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null", ""):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _add_common(parser: argparse.ArgumentParser) -> None:
    # Also accepted before the subcommand. SUPPRESS keeps the subparser from
    # clobbering a top-level `-v` with its own default.
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="log data loading details",
    )
    parser.add_argument("-c", "--config", default=None, help="path to a YAML config file")
    parser.add_argument("--source", choices=["yahoo", "csv", "synthetic"], help="data source")
    parser.add_argument("--symbol", help="ticker or instrument name")
    parser.add_argument("--path", help="CSV path (with --source csv)")
    parser.add_argument("--start", help="first date, YYYY-MM-DD")
    parser.add_argument("--end", help="last date, YYYY-MM-DD")
    parser.add_argument("--interval", help="bar interval, e.g. 1d, 1h, 1wk")
    parser.add_argument("--strategy", dest="name", help=f"one of: {', '.join(available())}")
    parser.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="strategy parameter override; repeatable",
    )
    parser.add_argument("--cash", dest="initial_cash", type=float, help="starting equity")
    parser.add_argument("--commission-bps", type=float, help="commission in basis points")
    parser.add_argument("--slippage-bps", type=float, help="slippage in basis points")
    parser.add_argument("--fill", choices=["next_open", "close"], help="fill timing")
    parser.add_argument("--risk-per-trade", type=float, help="equity fraction risked per trade")
    parser.add_argument("--max-position-pct", type=float, help="max notional as equity fraction")
    parser.add_argument("--stop-loss-atr", type=float, help="stop distance in ATR multiples")
    parser.add_argument("--take-profit-atr", type=float, help="target distance in ATR multiples")
    parser.add_argument("--atr-period", type=int, help="ATR lookback")
    parser.add_argument("--max-drawdown-stop", type=float, help="halt trading below this drawdown")


def _resolve(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Merge config file, CLI flags and --param into a Config plus strategy params."""
    config = load_config(args.config)
    base_strategy = config.strategy.name
    overrides = {
        key: getattr(args, key, None)
        for key in (
            "source",
            "symbol",
            "path",
            "start",
            "end",
            "interval",
            "name",
            "initial_cash",
            "commission_bps",
            "slippage_bps",
            "fill",
            "risk_per_trade",
            "max_position_pct",
            "stop_loss_atr",
            "take_profit_atr",
            "atr_period",
            "max_drawdown_stop",
        )
    }
    config = apply_overrides(config, **overrides)

    cli_params = _parse_params(getattr(args, "param", None))
    if args.name and args.name != base_strategy:
        # Strategy switched on the command line: the config file's params
        # belonged to the old strategy, so start from the new one's defaults.
        params = cli_params
    else:
        params = {**config.strategy.params, **cli_params}
    return config, params


def _load_bars(config: Any) -> pd.DataFrame:
    d = config.data
    log.info("loading %s bars for %s from %s", d.interval, d.symbol, d.source)
    df = load(
        source=d.source,
        symbol=d.symbol,
        start=d.start,
        end=d.end,
        interval=d.interval,
        path=d.path,
    )
    log.info("loaded %d bars: %s to %s", len(df), df.index[0].date(), df.index[-1].date())
    return df


def cmd_backtest(args: argparse.Namespace) -> int:
    config, params = _resolve(args)
    df = _load_bars(config)
    strategy = get_strategy(config.strategy.name, **params)

    result = Backtester(config.backtest, RiskManager(config.risk)).run(
        df, strategy, symbol=config.data.symbol
    )

    print(result.summary())
    print(f"\nSizing: {result.meta['risk']}")
    print(
        f"Bars: {result.meta['bars']} (warmup {result.meta['warmup']}), fill: {result.meta['fill']}"
    )

    if args.show_trades and len(result.trades):
        print("\nTrades")
        shown = result.trades.tail(args.show_trades).copy()
        for col in ("entry_price", "exit_price", "pnl", "quantity", "gross_pnl", "commission"):
            shown[col] = shown[col].round(2)
        shown["return_pct"] = (shown["return_pct"] * 100).round(2)
        print(shown.to_string(index=False))

    if args.out:
        written = result.to_csv(args.out)
        print("\nWrote " + ", ".join(written.values()))

    if args.report:
        path = write_backtest_report(result, args.report)
        print(f"Wrote report {path}")

    if args.json:
        payload = {
            "strategy": result.strategy,
            "symbol": result.symbol,
            "metrics": {
                k: (str(v) if isinstance(v, pd.Timestamp) else v)
                for k, v in result.metrics.to_dict().items()
            },
        }
        print("\n" + json.dumps(payload, indent=2, default=str))

    return 0


def _build_grid(args: argparse.Namespace) -> dict[str, list[Any]]:
    """Assemble the parameter grid from --grid and the --fast/--slow shorthands."""
    grid: dict[str, list[Any]] = {}
    for spec in getattr(args, "grid", None) or []:
        if "=" not in spec:
            raise ValueError(f"--grid expects key=v1,v2,..., got {spec!r}")
        key, _, values = spec.partition("=")
        parsed = [_coerce(v) for v in values.split(",") if v != ""]
        if not parsed:
            raise ValueError(f"--grid {spec!r} lists no values")
        grid[key.strip()] = parsed
    if getattr(args, "fast", None):
        grid["fast"] = _int_list(args.fast)
    if getattr(args, "slow", None):
        grid["slow"] = _int_list(args.slow)
    if not grid:
        raise ValueError(
            "this command needs at least one --grid key=v1,v2 (or the --fast/--slow shorthands)"
        )
    return grid


def cmd_optimize(args: argparse.Namespace) -> int:
    """Grid-search strategy parameters and rank the runs.

    Treat the output as a hypothesis, not a result: the best cell of a grid is
    partly luck, and the more cells there are the luckier it gets. Confirm the
    winner with `walkforward`, which never lets a search see its own test data.
    """
    config, base_params = _resolve(args)
    df = _load_bars(config)
    grid = _build_grid(args)

    log.info("testing %d combination(s)", grid_size(grid))
    result = grid_search(
        df,
        config.strategy.name,
        grid,
        base_params=base_params,
        config=config.backtest,
        risk=RiskManager(config.risk),
        objective=args.objective,
        min_trades=args.min_trades,
    )

    if not result.rows:
        print("No valid parameter combinations to test.")
        return 1

    table = result.table(include_ineligible=args.include_thin)
    tested = f"\nTested {len(result.rows)} combination(s)"
    if result.skipped:
        tested += f", skipped {result.skipped} invalid"
    thin = len(result.rows) - len(result.eligible)
    if thin and not args.include_thin:
        tested += f", hid {thin} with fewer than {args.min_trades} trades"
    print(tested)
    print(f"Ranked by {args.objective} (top {min(args.top, len(table))}):\n")
    print(table.head(args.top).to_string(index=False))

    print(
        f"\nCaution: {len(result.rows)} combinations were scored against the same "
        f"{len(df)} bars, so the top row is the best fit to this sample rather than "
        "the best strategy. Confirm it with:\n"
        f"  python -m algobot walkforward --strategy {config.strategy.name} "
        + " ".join(f"--grid {k}={','.join(str(v) for v in vals)}" for k, vals in grid.items())
    )

    if args.out:
        from pathlib import Path

        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.table(include_ineligible=True).to_csv(path, index=False)
        print(f"Wrote {path}")
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    """Validate a strategy on data its parameter search never saw."""
    config, base_params = _resolve(args)
    df = _load_bars(config)
    grid = _build_grid(args)

    result = walk_forward(
        df,
        config.strategy.name,
        grid,
        base_params=base_params,
        config=config.backtest,
        risk=RiskManager(config.risk),
        n_splits=args.splits,
        train_size=args.train_size,
        scheme=args.scheme,
        objective=args.objective,
        min_trades=args.min_trades,
        symbol=config.data.symbol,
    )

    print(result.summary())
    print("\nPer-fold results\n")
    print(result.folds_table().to_string(index=False))
    print("\nParameter stability across folds\n")
    print(result.parameter_stability().to_string(index=False))
    print(
        "\nRead it this way: efficiency near 1.0 with parameters that barely move "
        "means the edge survived contact with unseen data. Parameters that change "
        "every fold mean the search is refitting, not validating."
    )

    if args.out:
        from pathlib import Path

        path = Path(args.out)
        path.mkdir(parents=True, exist_ok=True)
        result.equity.to_frame("equity").to_csv(path / "oos_equity.csv")
        result.folds_table().to_csv(path / "folds.csv", index=False)
        if len(result.trades):
            result.trades.to_csv(path / "oos_trades.csv", index=False)
        print(f"\nWrote {path}/oos_equity.csv, folds.csv, oos_trades.csv")
    return 0


def cmd_montecarlo(args: argparse.Namespace) -> int:
    """Resample a backtest to see how much of it was the luck of the draw."""
    config, params = _resolve(args)
    df = _load_bars(config)
    strategy = get_strategy(config.strategy.name, **params)

    result = Backtester(config.backtest, RiskManager(config.risk)).run(
        df, strategy, symbol=config.data.symbol
    )
    if not len(result.trades) and args.method == "trades":
        print("The backtest produced no closed trades, so there is nothing to resample.")
        return 1

    simulation = monte_carlo(
        result.equity,
        result.trades,
        method=args.method,
        trials=args.trials,
        block=args.block,
        seed=args.seed,
        ruin_threshold=args.ruin_threshold,
    )

    print(f"Strategy: {result.strategy} on {config.data.symbol}\n")
    print(simulation.summary())
    if simulation.observations < 30 and args.method == "trades":
        print(
            f"\nCaution: only {simulation.observations} trades were resampled. "
            "The spread below roughly 30 observations reflects the small sample "
            "as much as the strategy."
        )

    if args.out:
        from pathlib import Path

        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "total_return": simulation.total_returns,
                "final_equity": simulation.final_equity,
                "max_drawdown": simulation.max_drawdowns,
            }
        ).to_csv(path, index=False)
        print(f"Wrote {path}")
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Run one strategy across several instruments and combine the sleeves."""
    config, params = _resolve(args)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if len(symbols) < 2:
        raise ValueError("portfolio needs at least 2 symbols, e.g. --symbols AAPL,MSFT,SPY")

    weights = None
    if args.weights:
        values = [float(w) for w in args.weights.split(",")]
        if len(values) != len(symbols):
            raise ValueError(
                f"got {len(values)} weight(s) for {len(symbols)} symbol(s); "
                "supply one weight per symbol or omit --weights for equal sizing"
            )
        weights = dict(zip(symbols, values, strict=True))

    log.info("loading %d symbol(s)", len(symbols))
    data, load_failures = load_many(
        symbols,
        source=config.data.source,
        start=config.data.start,
        end=config.data.end,
        interval=config.data.interval,
        directory=config.data.path,
    )

    result = run_portfolio(
        data,
        config.strategy.name,
        strategy_params=params,
        config=config.backtest,
        risk=RiskManager(config.risk),
        weights=weights,
    )
    result.failures.update(load_failures)

    print(result.summary())
    print("\nPer-symbol contribution\n")
    print(result.contributions().to_string(index=False))

    correlation = result.correlation()
    if not correlation.empty:
        print("\nSleeve return correlation\n")
        print(correlation.round(2).to_string())
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        print(f"\nAverage pairwise correlation: {upper.stack().mean():.2f}")
        print(
            "The lower that number, the more the extra symbols are actually "
            "buying you. Near 1.0 and you own one position in four disguises."
        )

    if args.out:
        from pathlib import Path

        path = Path(args.out)
        path.mkdir(parents=True, exist_ok=True)
        result.equity.to_frame("equity").to_csv(path / "portfolio_equity.csv")
        result.contributions().to_csv(path / "contributions.csv", index=False)
        if len(result.trades):
            result.trades.to_csv(path / "trades.csv", index=False)
        if not correlation.empty:
            correlation.to_csv(path / "correlation.csv")
        print(f"\nWrote {path}/portfolio_equity.csv, contributions.csv, trades.csv")

    if args.report:
        path = write_portfolio_report(result, args.report)
        print(f"Wrote report {path}")
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    from algobot.strategies import REGISTRY

    print("Available strategies:\n")
    for family in ("trend", "mean-reversion", "benchmark"):
        names = [n for n in available() if FAMILIES.get(n) == family]
        if not names:
            continue
        print(f"  [{family}]")
        for name in names:
            cls = REGISTRY[name]
            doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
            print(f"    {name:<14} {doc}")
            for key, default in cls.params_schema.items():
                print(f"        {key} = {default!r}")
            print()
    print("Use with:  --strategy NAME --param key=value")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algobot",
        description="Backtest-first algorithmic trading framework",
    )
    parser.add_argument("--version", action="version", version=f"algobot {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log data loading details")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="run a single backtest")
    _add_common(bt)
    bt.add_argument("--out", help="directory for equity_curve.csv, trades.csv, fills.csv")
    bt.add_argument(
        "--report",
        metavar="FILE.html",
        help="write a standalone HTML report with charts (no external assets)",
    )
    bt.add_argument("--json", action="store_true", help="also print metrics as JSON")
    bt.add_argument(
        "--show-trades",
        type=int,
        default=0,
        metavar="N",
        help="print the last N trades",
    )
    bt.set_defaults(func=cmd_backtest)

    def add_grid_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--grid",
            action="append",
            metavar="KEY=V1,V2",
            help="parameter values to sweep; repeatable",
        )
        p.add_argument("--fast", help="shorthand for --grid fast=...")
        p.add_argument("--slow", help="shorthand for --grid slow=...")
        p.add_argument(
            "--objective",
            default="sharpe",
            choices=sorted(OBJECTIVES),
            help="metric to maximise (default: sharpe)",
        )
        p.add_argument(
            "--min-trades",
            type=int,
            default=5,
            help="ignore parameter sets with fewer round trips (default: 5)",
        )

    opt = sub.add_parser("optimize", help="grid-search strategy parameters")
    _add_common(opt)
    add_grid_args(opt)
    opt.add_argument("--top", type=int, default=15, help="rows to display")
    opt.add_argument(
        "--include-thin",
        action="store_true",
        help="also rank parameter sets that traded fewer than --min-trades times",
    )
    opt.add_argument("--out", help="write the full ranking to this CSV path")
    opt.set_defaults(func=cmd_optimize)

    wf = sub.add_parser(
        "walkforward",
        help="validate on data the parameter search never saw",
    )
    _add_common(wf)
    add_grid_args(wf)
    wf.add_argument("--splits", type=int, default=5, help="number of folds (default: 5)")
    wf.add_argument(
        "--train-size",
        type=float,
        default=0.5,
        help="fraction of the series used for the first training window (default: 0.5)",
    )
    wf.add_argument(
        "--scheme",
        choices=["anchored", "rolling"],
        default="anchored",
        help="anchored training windows grow, rolling windows slide",
    )
    wf.add_argument("--out", help="directory for oos_equity.csv, folds.csv, oos_trades.csv")
    wf.set_defaults(func=cmd_walkforward)

    mc = sub.add_parser(
        "montecarlo",
        help="resample a backtest to measure how much of it was luck",
    )
    _add_common(mc)
    mc.add_argument(
        "--method",
        choices=["trades", "returns"],
        default="trades",
        help="resample trade outcomes or block-resample bar returns",
    )
    mc.add_argument("--trials", type=int, default=2_000, help="simulated paths (default: 2000)")
    mc.add_argument("--block", type=int, default=10, help="block length for --method returns")
    mc.add_argument("--seed", type=int, default=7, help="RNG seed, for reproducibility")
    mc.add_argument(
        "--ruin-threshold",
        type=float,
        default=0.5,
        help="drawdown that counts as ruin (default: 0.5)",
    )
    mc.add_argument("--out", help="write every simulated path outcome to this CSV path")
    mc.set_defaults(func=cmd_montecarlo)

    pf = sub.add_parser("portfolio", help="run one strategy across several instruments")
    _add_common(pf)
    pf.add_argument(
        "--symbols",
        required=True,
        metavar="A,B,C",
        help="comma-separated symbols; with --source csv, files are read from "
        "<--path>/<SYMBOL>.csv",
    )
    pf.add_argument(
        "--weights",
        metavar="W1,W2,W3",
        help="capital split, one weight per symbol (default: equal); "
        "values are normalised, so 2,1,1 means 50/25/25",
    )
    pf.add_argument("--out", help="directory for portfolio_equity.csv and friends")
    pf.add_argument(
        "--report",
        metavar="FILE.html",
        help="write a standalone HTML report with charts (no external assets)",
    )
    pf.set_defaults(func=cmd_portfolio)

    ls = sub.add_parser("strategies", help="list registered strategies and parameters")
    ls.set_defaults(func=cmd_strategies)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        return int(args.func(args))
    except (DataError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
