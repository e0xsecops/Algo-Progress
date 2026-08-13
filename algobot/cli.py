"""Command line interface.

    python -m algobot backtest --source synthetic --symbol DEMO
    python -m algobot backtest --symbol AAPL --start 2018-01-01 --out results/aapl
    python -m algobot optimize --fast 5,10,20 --slow 50,100,200
    python -m algobot strategies
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from typing import Any

import pandas as pd

from algobot import __version__
from algobot.backtest import Backtester
from algobot.config import apply_overrides, load_config
from algobot.data import DataError, load
from algobot.risk import RiskManager
from algobot.strategies import available, get_strategy

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
    print(f"Bars: {result.meta['bars']} (warmup {result.meta['warmup']}), fill: {result.meta['fill']}")

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


def cmd_optimize(args: argparse.Namespace) -> int:
    """Grid-search strategy parameters and rank the runs.

    Treat the output as a hypothesis, not a result: the best cell of a grid is
    partly luck, and the more cells there are the luckier it gets. Re-test the
    winner on a period the search never saw.
    """
    config, base_params = _resolve(args)
    df = _load_bars(config)

    grid: dict[str, list[Any]] = {}
    for spec in args.grid or []:
        if "=" not in spec:
            raise ValueError(f"--grid expects key=v1,v2,..., got {spec!r}")
        key, _, values = spec.partition("=")
        grid[key.strip()] = [_coerce(v) for v in values.split(",") if v != ""]
    if args.fast:
        grid["fast"] = _int_list(args.fast)
    if args.slow:
        grid["slow"] = _int_list(args.slow)
    if not grid:
        raise ValueError("optimize needs at least one --grid key=v1,v2 (or --fast/--slow)")

    backtester = Backtester(config.backtest, RiskManager(config.risk))
    keys = list(grid)
    rows = []
    skipped = 0

    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {**base_params, **dict(zip(keys, combo))}
        try:
            strategy = get_strategy(config.strategy.name, **params)
        except ValueError as exc:  # invalid combination, e.g. fast >= slow
            skipped += 1
            log.debug("skipping %s: %s", params, exc)
            continue
        result = backtester.run(df, strategy, symbol=config.data.symbol, benchmark=False)
        m = result.metrics
        rows.append(
            {
                **dict(zip(keys, combo)),
                "return_pct": round(m.total_return * 100, 2),
                "cagr_pct": round(m.cagr * 100, 2),
                "sharpe": round(m.sharpe, 3),
                "max_dd_pct": round(m.max_drawdown * 100, 2),
                "trades": m.num_trades,
                "win_rate_pct": round(m.win_rate * 100, 1),
            }
        )

    if not rows:
        print("No valid parameter combinations to test.")
        return 1

    table = pd.DataFrame(rows).sort_values(args.sort, ascending=False).reset_index(drop=True)
    tested = f"\nTested {len(rows)} combination(s)"
    print(tested + (f", skipped {skipped} invalid" if skipped else ""))
    print(f"Ranked by {args.sort} (top {min(args.top, len(table))}):\n")
    print(table.head(args.top).to_string(index=False))
    print(
        "\nNote: the top row is the best fit to this sample, which is not the same as "
        "the best strategy. Re-test it on data the search never saw."
    )

    if args.out:
        from pathlib import Path

        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        print(f"Wrote {path}")
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    from algobot.strategies import REGISTRY

    print("Available strategies:\n")
    for name in available():
        cls = REGISTRY[name]
        doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        print(f"  {name:<12} {doc}")
        for key, default in cls.params_schema.items():
            print(f"      {key} = {default!r}")
        print()
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
    bt.add_argument("--json", action="store_true", help="also print metrics as JSON")
    bt.add_argument(
        "--show-trades",
        type=int,
        default=0,
        metavar="N",
        help="print the last N trades",
    )
    bt.set_defaults(func=cmd_backtest)

    opt = sub.add_parser("optimize", help="grid-search strategy parameters")
    _add_common(opt)
    opt.add_argument(
        "--grid",
        action="append",
        metavar="KEY=V1,V2",
        help="parameter values to sweep; repeatable",
    )
    opt.add_argument("--fast", help="shorthand for --grid fast=...")
    opt.add_argument("--slow", help="shorthand for --grid slow=...")
    opt.add_argument("--sort", default="sharpe", help="ranking column (default: sharpe)")
    opt.add_argument("--top", type=int, default=15, help="rows to display")
    opt.add_argument("--out", help="write the full ranking to this CSV path")
    opt.set_defaults(func=cmd_optimize)

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
