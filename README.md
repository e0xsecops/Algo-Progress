# Algo-Progress

A backtest-first algorithmic trading framework in Python. It loads OHLCV bars,
turns them into position signals, sizes those positions against a risk budget,
simulates fills with realistic costs, and scores the result against buy-and-hold.

The design keeps each stage independent, so a strategy written today can be run
against a live broker later without being rewritten.

```
data  ->  strategy  ->  risk  ->  execution  ->  metrics
bars      -1/0/+1       size      fills+costs    Sharpe, drawdown, ...
```

## Quick start

```bash
pip install -r requirements.txt

# No network needed - runs on the bundled sample bars
python -m algobot backtest --source csv --path examples/sample_bars.csv --symbol DEMO

# Or on generated data
python -m algobot backtest --source synthetic --symbol DEMO

# Real market data (needs internet access to Yahoo Finance)
python -m algobot backtest --symbol AAPL --start 2018-01-01 --show-trades 10
```

Output:

```
========================================================================
ema_cross(fast=20, slow=50, allow_short=False, trend_filter=200) on DEMO
========================================================================
  Period              2020-01-01 to 2022-11-15  (2.87 years)
  Equity              100,000.00 -> 95,455.64
  Total return        -4.54%
  CAGR                -1.61%
  Volatility (ann.)   1.88%
  Sharpe              -0.82
  Sortino             -0.38
  Max drawdown        -5.23%  (368 bars underwater)
  Calmar              -0.31
  Time in market      23.47%
  Trades              13
  Win rate            7.69%
  Profit factor       0.08
  Expectancy/trade    -349.57
  Avg win / avg loss  411.62 / -413.00
  Best / worst trade  411.62 / -985.00
  Avg bars held       13.5
  Costs paid          commission 224.51, slippage 89.80
========================================================================
Benchmark (buy & hold, no costs): return -22.41%, CAGR -8.46%, Sharpe -0.33, max DD -34.18%
Edge vs benchmark: +17.87% total return
```

Those are losing numbers, and that is the point: the sample file is random
noise, and on noise a trend strategy pays costs for nothing. Losing less than
buy-and-hold is what a stop-loss buys you, not alpha. Point it at real bars
before drawing conclusions.

## Commands

| Command | What it does |
| --- | --- |
| `backtest` | Run one strategy over one instrument and print the report |
| `optimize` | Grid-search parameters and rank the runs |
| `strategies` | List registered strategies and their parameters |

Useful flags: `--out DIR` writes `equity_curve.csv`, `trades.csv` and
`fills.csv`; `--json` prints the metrics as JSON; `--show-trades N` prints the
last N round trips; `-v` logs data loading.

```bash
python -m algobot optimize --source csv --path examples/sample_bars.csv \
    --fast 5,10,20 --slow 50,100,200 --sort sharpe
```

## Configuration

Every setting lives in `config.yaml` and every one can be overridden on the
command line. Load a different file with `-c my_config.yaml`.

```yaml
data:
  source: yahoo          # yahoo | csv | synthetic
  symbol: SPY
  start: "2015-01-01"
strategy:
  name: ema_cross
  params: {fast: 20, slow: 50, allow_short: false, trend_filter: 200}
backtest:
  initial_cash: 100000.0
  commission_bps: 5.0    # 0.05% of notional per trade
  slippage_bps: 2.0      # 0.02% adverse fill
  fill: next_open        # next_open (realistic) | close
risk:
  risk_per_trade: 0.01   # 1% of equity between entry and stop
  max_position_pct: 1.0
  stop_loss_atr: 2.0
  atr_period: 14
  max_drawdown_stop: 0.25
```

## How the backtest avoids fooling you

A backtester is only as useful as the mistakes it refuses to make. This one:

- **Never looks ahead.** The signal computed from bar `t`'s close is executed at
  bar `t+1`'s open. Every indicator is causal by construction.
- **Charges for trading.** Slippage moves the fill price against you and
  commission comes off cash on top; both are reported so you can see what the
  strategy paid to exist.
- **Fills gaps pessimistically.** A stop that gaps through fills at the open,
  not at the stop price. When a bar touches both the stop and the target, the
  stop is assumed to have hit first.
- **Shows the benchmark.** Buy-and-hold over the same window is printed next to
  every result, because beating an index is the only return that counts.
- **Warms up before trading.** No position is taken until every indicator and
  the ATR are fully formed.
- **Closes its books.** The final bar opens nothing new and liquidates whatever
  is open at the close, so no reported return rests on an unclosed position.

What it does *not* model: partial fills, order-book depth, borrow costs and
shorting fees, dividends beyond Yahoo's adjusted prices, intrabar path detail
(a bar is only OHLC), and survivorship bias in your symbol choice. Treat a
backtest as evidence a strategy is *not* broken, never as proof it works.

## Risk model

Position size comes from the risk budget rather than from a fixed share count:

```
size = (equity x risk_per_trade) / (ATR x stop_loss_atr)     capped by
       (equity x max_position_pct) / price
```

A volatile instrument therefore gets a smaller position than a quiet one for
the same 1% of equity at risk. Stops and targets are set at entry from the same
ATR, and if the account draws down past `max_drawdown_stop` the engine
liquidates and stops trading for the rest of the run.

## Adding a strategy

Subclass `Strategy`, declare the parameters, return a position per bar, and
register it. Sizing, stops and execution are handled for you.

```python
# algobot/strategies/rsi_reversion.py
from typing import Any, ClassVar
import numpy as np, pandas as pd
from algobot.indicators import rsi
from algobot.strategies.base import Strategy


class RsiReversionStrategy(Strategy):
    """Buy oversold, exit at the midline."""

    name: ClassVar[str] = "rsi_reversion"
    params_schema: ClassVar[dict[str, Any]] = {"period": 14, "oversold": 30}

    @property
    def warmup(self) -> int:
        return self.period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        value = rsi(df["close"], self.period)
        return pd.Series(np.where(value < self.oversold, 1.0, 0.0), index=df.index)
```

Add it to `REGISTRY` in `algobot/strategies/__init__.py` and it is immediately
available to `--strategy rsi_reversion`, to the optimizer, and to the config file.

The one rule: the value at bar `t` may only depend on bars `<= t`. Anything else
produces a beautiful equity curve that cannot be traded.

## Project layout

```
algobot/
  config.py            typed config, YAML loading, CLI overrides
  indicators.py        causal EMA, SMA, ATR, RSI, rolling extremes
  data/loader.py       CSV / Yahoo / synthetic loaders, OHLCV normalisation
  strategies/          Strategy base class, registry, ema_cross, buy_hold
  risk/manager.py      position sizing, ATR stops, drawdown kill-switch
  execution/broker.py  paper fills, cash accounting, trade ledger
  backtest/engine.py   the bar loop
  backtest/metrics.py  Sharpe, Sortino, drawdown, profit factor, ...
  cli.py               backtest / optimize / strategies commands
tests/                 pytest suite
examples/              sample bars for an offline run
```

## Tests

```bash
pytest -q
```

The suite covers indicator correctness, broker cash and P&L accounting through
reversals, metric formulas against hand-computed values, and — most importantly
— a lookahead guard that asserts changing future bars cannot change past trades.

## Roadmap

The backtester is the foundation; live trading is the next layer, not a rewrite.

- [ ] Live/paper execution adapter (CCXT or Alpaca) behind the broker interface
- [ ] Walk-forward validation to replace single-period grid search
- [ ] Portfolio backtests across multiple instruments
- [ ] Equity curve plotting and an HTML report

## Disclaimer

This is software for research and education. Backtested performance is not
indicative of future results, and nothing here is financial advice. Do not risk
money you cannot afford to lose.
