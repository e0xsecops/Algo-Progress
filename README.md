# Algo-Progress

**A backtest-first algorithmic trading framework in Python — built to disprove strategies, not to flatter them.**

[![CI](https://github.com/e0xsecops/Algo-Progress/actions/workflows/ci.yml/badge.svg)](https://github.com/e0xsecops/Algo-Progress/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-240%20passing-brightgreen)](tests/)

Anyone can write a backtest that produces a rising equity curve. The hard part
is building one that tells you the truth — where the signal can't peek at
tomorrow's price, where costs are charged the way a real venue charges them,
and where a parameter search never grades its own homework.

That is what this framework is for.

```
 data  ─→  strategy  ─→   risk   ─→  execution  ─→  metrics  ─→  validation
 bars      -1 / 0 / +1    sizing     fills + costs   Sharpe,      walk-forward,
                          + stops                    drawdown     Monte Carlo
```

Each stage is independent and swappable. A strategy written today runs
unchanged against a backtest, a portfolio of 30 instruments, or — when you add
a broker adapter — a live feed.

---

## Table of contents

- [Quick start](#quick-start)
- [The workflow](#the-workflow) — the six-step process from idea to verdict
- [Commands](#commands)
- [Configuration](#configuration)
- [Strategies](#strategies)
- [The risk model](#the-risk-model)
- [How the engine avoids fooling you](#how-the-engine-avoids-fooling-you)
- [Writing your own strategy](#writing-your-own-strategy)
- [Python API](#python-api)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Quick start

```bash
git clone https://github.com/e0xsecops/Algo-Progress.git
cd Algo-Progress

pip install -e .                 # core: pandas, numpy, PyYAML
pip install -e ".[data,dev]"     # + yfinance (live data) and pytest
```

Requires Python 3.10 or newer.

Everything below runs **offline** against the bundled sample bars — no API key,
no account, no network:

```bash
# 1. Run a backtest and open an HTML report
python -m algobot backtest \
    --source csv --path examples/sample_bars.csv --symbol DEMO \
    --report report.html

# 2. See every strategy and its parameters
python -m algobot strategies
```

With internet access and the `data` extra installed, point it at a real ticker
instead:

```bash
python -m algobot backtest --symbol AAPL --start 2018-01-01 --show-trades 10
```

<details>
<summary><b>Sample output</b></summary>

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

Those are losing numbers, deliberately. The bundled file is **random synthetic
noise**, and on noise a trend strategy simply pays costs for nothing. Losing
less than buy-and-hold is what a stop-loss buys you, not alpha. Point it at real
bars before drawing any conclusion.

</details>

---

## The workflow

A backtest is the *first* step, not the answer. This is the process the tool is
built around, and each step exists to kill ideas that survived the last one.

### Step 1 — Explore: does the idea do anything at all?

```bash
python -m algobot backtest --source csv --path examples/sample_bars.csv \
    --strategy macd --show-trades 10 --report report.html
```

Read the trade count first, not the return. Three trades that made money tell
you nothing. Then read `Edge vs benchmark` — if the strategy can't beat holding
the asset, the rest is noise.

### Step 2 — Search: which parameters even work?

```bash
python -m algobot optimize --source csv --path examples/sample_bars.csv \
    --strategy macd --grid fast=8,12,16 --grid slow=21,26,34 --objective sharpe
```

```
Tested 9 combination(s)
Ranked by sharpe (top 15):

 fast  slow   score  return_pct  sharpe  max_dd_pct  trades  win_rate_pct   (columns trimmed)
    8    26 -0.4182       -2.86  -0.418       -4.77      34          32.4
    8    34 -0.5817       -3.82  -0.582       -4.82      33          27.3
    8    21 -0.6260       -4.22  -0.626       -5.04      38          28.9
```

The tool deliberately makes this step feel unsatisfying. Parameter sets with
too few trades are flagged and hidden (`--min-trades`, `--include-thin`),
infinite scores can't win by accident, and the footer tells you to go validate:

> Caution: 9 combinations were scored against the same 750 bars, so the top row
> is the best fit to this sample rather than the best strategy. Confirm it with:
> `python -m algobot walkforward --strategy macd --grid fast=8,12,16 --grid slow=21,26,34`

### Step 3 — Validate: does it survive data the search never saw?

**This is the step that matters.** Walk-forward picks parameters using only
data *before* each test window, applies them forward unchanged, and stitches
the out-of-sample segments into an equity curve nobody optimised.

```bash
python -m algobot walkforward --source csv --path examples/sample_bars.csv \
    --strategy macd --grid fast=8,12 --grid slow=21,26 --splits 3
```

```
Folds: 3 (anchored, selected by sharpe)
Walk-forward efficiency: -5.04  (out-of-sample return / in-sample return, ~1.0 is healthy)
Profitable folds: 0%

 fold                    train                     test          params  is_return_pct  oos_return_pct
    1 2020-01-01 to 2021-06-08 2021-06-09 to 2021-11-30 fast=8, slow=26           1.11           -1.25
    2 2020-01-01 to 2021-11-30 2021-12-01 to 2022-05-24 fast=8, slow=26           0.49           -1.19
    3 2020-01-01 to 2022-05-24 2022-05-25 to 2022-11-15 fast=8, slow=26          -0.70           -2.08
```

Three numbers decide the verdict:

| Signal | Healthy | What it means |
| --- | --- | --- |
| **Walk-forward efficiency** | ~0.5 – 1.0 | Out-of-sample return ÷ in-sample return. Near zero or negative means the search was fitting noise. Far above 1.0 is luck, not skill. |
| **Profitable folds** | > 60% | An edge that only appears in one fold out of five is one lucky period. |
| **Parameter stability** | few distinct values | Optimal parameters that jump every fold mean you're refitting a *different* strategy each window, not validating one. |

The example above scores −5.04 efficiency with 0% profitable folds. That is the
tool doing its job: the idea is dead, and it cost you one command to find out.

Two schemes are available — `--scheme anchored` (training windows grow from the
start) and `--scheme rolling` (fixed-length windows slide forward, assuming the
distant past stops being relevant).

### Step 4 — Stress test: how much of this was luck?

A backtest shows *one* path out of many the same strategy could have produced.
Monte Carlo resamples the building blocks thousands of times to measure the
spread.

```bash
python -m algobot montecarlo --source csv --path examples/sample_bars.csv \
    --strategy macd --trials 2000
```

```
  Resampled from 31 observation(s)
  Backtest actually returned -5.83% with a -6.49% drawdown
  That return sits at the 49th percentile of simulated paths

percentile  total_return_pct  final_equity  max_drawdown_pct
        p5            -10.78      89217.31            -11.21
       p50             -5.76      94235.27             -6.85
       p95             -0.14      99859.07             -3.01

  Probability of profit: 4.8%
  Probability of a drawdown past 50%: 0.0%
```

Two methods, because they assume different things:

- `--method trades` (default) resamples trade outcomes with replacement. It
  answers *"how much of the result came from the order the trades happened to
  arrive in?"* and assumes trades are independent.
- `--method returns` block-resamples bar returns, preserving the volatility
  clustering and short-run autocorrelation that make real drawdowns deeper than
  an independent shuffle ever would.

Look at **p5 drawdown**, not the median return. If you couldn't hold the
position through the 5th-percentile path, the strategy is untradeable however
good its average looks.

### Step 5 — Diversify: does it hold up across instruments?

An edge that only exists on one symbol is usually an artifact of that symbol.

```bash
python -m algobot portfolio --source csv --path examples/portfolio \
    --symbols ALFA,BETA,GAMA --strategy macd --report portfolio.html
```

```
Diversification ratio: 1.71 (>1 means the sleeves offset each other)
Worst single-sleeve drawdown: -6.77% vs portfolio -3.93%

symbol  weight_pct  allocated    final      pnl  return_pct  sharpe  max_dd_pct  trades  contribution_pct
  BETA        33.3   33333.33 33207.48  -125.85       -0.38   -0.04       -5.61      28             -0.13
  GAMA        33.3   33333.33 31875.47 -1457.87       -4.37   -0.65       -6.77      30             -1.46
  ALFA        33.3   33333.33 31857.59 -1475.75       -4.43   -0.61       -6.23      39             -1.48
```

Capital is split into per-symbol sleeves that run independently and are then
summed. That makes diversification *measurable*: the worst sleeve draws down
6.77% while the portfolio only reaches 3.93%, and the sleeve correlation matrix
shows whether the extra symbols are genuinely buying you anything or whether
you own one position in three disguises.

Use `--weights 2,1,1` for an unequal split (values are normalised, so that's
50/25/25).

### Step 6 — Report: keep the evidence

```bash
python -m algobot backtest --source csv --path examples/sample_bars.csv \
    --report report.html --out results/
```

`--report` produces a **single self-contained HTML file** — equity curve
against benchmark, underwater drawdown plot, monthly returns calendar, trade
histogram, and full metric and trade tables. All charts are inline SVG: no CDN,
no JavaScript, no external requests. It opens on an air-gapped machine and can
be archived as a record of exactly what a run produced.

`--out` writes `equity_curve.csv`, `trades.csv` and `fills.csv` for your own
analysis.

---

## Commands

| Command | Purpose |
| --- | --- |
| `backtest` | Run one strategy over one instrument; print the report, export CSVs and HTML |
| `optimize` | Grid-search parameters and rank them by any metric |
| `walkforward` | Validate on data the parameter search never saw |
| `montecarlo` | Resample a backtest to measure how much was luck |
| `portfolio` | Run one strategy across many instruments and combine the sleeves |
| `strategies` | List registered strategies, grouped by family, with their parameters |

**Shared flags** (every command): `--source {yahoo,csv,synthetic}` · `--symbol`
· `--path` · `--start` · `--end` · `--interval` · `--strategy` · `--param
key=value` (repeatable) · `--cash` · `--commission-bps` · `--slippage-bps` ·
`--fill {next_open,close}` · `--risk-per-trade` · `--stop-loss-atr` ·
`--take-profit-atr` · `--atr-period` · `--max-drawdown-stop` · `-c CONFIG` ·
`-v`

Run `python -m algobot <command> --help` for the full list.

---

## Configuration

Every setting lives in `config.yaml`, and every one can be overridden on the
command line. Load a different file with `-c my_config.yaml`.

```yaml
data:
  source: yahoo          # yahoo | csv | synthetic
  symbol: SPY
  start: "2015-01-01"
  end: null              # null = today
  interval: 1d           # 1d | 1h | 1wk

strategy:
  name: ema_cross
  params:
    fast: 20
    slow: 50
    allow_short: false
    trend_filter: 200    # 0 disables

backtest:
  initial_cash: 100000.0
  commission_bps: 5.0    # 0.05% of notional per trade
  slippage_bps: 2.0      # 0.02% adverse fill
  fill: next_open        # next_open (realistic) | close

risk:
  risk_per_trade: 0.01   # 1% of equity between entry and stop
  max_position_pct: 1.0  # cap on notional exposure
  stop_loss_atr: 2.0     # stop distance in ATR multiples (null disables)
  take_profit_atr: null
  atr_period: 14
  max_drawdown_stop: 0.25  # liquidate and halt below this drawdown
```

Precedence is `CLI flag` > `config file` > `dataclass default`. Unknown keys
and impossible values (a negative commission, `fast >= slow`) are rejected at
load time with a message naming the offending field, rather than silently
producing a nonsense backtest.

**Data sources.** `csv` reads any file with a date column or index and
OHLC columns under almost any spelling (`Adj Close`, `o/h/l/c`, MultiIndex
columns from `yfinance` — all normalised). `yahoo` downloads
split/dividend-adjusted bars. `synthetic` generates reproducible
geometric-Brownian-motion bars from a seed, for offline demos and tests.

---

## Strategies

Six ship in the box, spanning both families — deliberately, because they fail in
opposite conditions.

| Strategy | Family | Idea | Key parameters |
| --- | --- | --- | --- |
| `ema_cross` | trend | Long while the fast EMA leads the slow one | `fast`, `slow`, `trend_filter` |
| `macd` | trend | Long while MACD leads its signal line | `fast`, `slow`, `signal`, `min_histogram` |
| `donchian` | trend | Buy an N-bar breakout, exit on an M-bar breakout back | `entry_period`, `exit_period` |
| `rsi_reversion` | mean-reversion | Buy oversold, exit at the midline | `period`, `oversold`, `exit_level` |
| `bollinger` | mean-reversion | Fade the lower band, exit at the middle | `period`, `num_std` |
| `buy_hold` | benchmark | Always long — the bar every strategy must clear | — |

Every one accepts `allow_short=true`.

**Trend followers** bleed in choppy ranges and pay for it with rare large
winners — expect a win rate near 30% and a fat right tail. **Mean reverters**
win often and lose badly when a trend refuses to revert — expect a win rate
above 60% and an occasional loss that erases twenty wins. A high win rate is
not evidence of quality; check `profit_factor` and `worst_trade` instead.

---

## The risk model

Position size comes from a **risk budget**, not a fixed share count:

```
size = (equity × risk_per_trade) ÷ (ATR × stop_loss_atr)
       capped by (equity × max_position_pct) ÷ price
```

A volatile instrument therefore gets a *smaller* position than a quiet one for
the same 1% of equity at risk. Stops and targets are set at entry from the same
ATR reading, so they scale with the instrument too.

Three guards sit on top:

1. **No stop, no trade.** If the ATR isn't warm yet, the entry is skipped
   rather than sized blind.
2. **Drawdown kill-switch.** Past `max_drawdown_stop`, the engine liquidates
   and stops trading for the rest of the run — the result shows you where it
   halted instead of a fantasy recovery.
3. **Exposure cap.** `max_position_pct` bounds notional regardless of what the
   risk budget suggests.

---

## How the engine avoids fooling you

A backtester is only as useful as the mistakes it refuses to make.

- **It never looks ahead.** The signal computed from bar `t`'s close is executed
  at bar `t+1`'s open. Every indicator is causal by construction, and the test
  suite proves it: rewriting the last 200 bars of the price series leaves every
  earlier equity value, position and closed trade byte-identical.
- **It charges for trading.** Slippage moves the fill price against you;
  commission comes off cash on top. Both are reported so you can see what the
  strategy paid merely to exist.
- **It fills gaps pessimistically.** A stop that gaps through fills at the open,
  not at the stop price. When a bar touches both the stop and the target, the
  stop is assumed to have hit first — there's no way to know the intrabar order,
  so it assumes the unfavourable one.
- **It warms up before trading.** No position is taken until every indicator and
  the ATR are fully formed.
- **It closes its books.** The final bar opens nothing new and liquidates
  whatever is open at the close, so no reported return rests on an unclosed
  position.
- **It shows the benchmark.** Buy-and-hold over the same window sits next to
  every result.

Per-bar sequence, in order:

```
1. act on the pending signal          (entry, exit or reversal)
2. test stop-loss, then take-profit   (against this bar's high/low)
3. mark equity at the close           (and check the drawdown kill-switch)
```

---

## Writing your own strategy

Subclass `Strategy`, declare your parameters, return a position per bar. Sizing,
stops, fills and accounting are handled for you.

```python
# algobot/strategies/my_strategy.py
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from algobot.indicators import rsi, sma
from algobot.strategies.base import Strategy


class MyStrategy(Strategy):
    """One-line description — this shows up in `algobot strategies`."""

    name: ClassVar[str] = "my_strategy"
    params_schema: ClassVar[dict[str, Any]] = {"period": 14, "threshold": 40}

    def validate(self) -> None:
        if not 0 < self.threshold < 100:
            raise ValueError("threshold must be between 0 and 100")

    @property
    def warmup(self) -> int:
        return self.period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        value = rsi(df["close"], self.period)
        return pd.Series(
            np.where(value < self.threshold, 1.0, 0.0), index=df.index, dtype=float
        )
```

Register it:

```python
# algobot/strategies/__init__.py
from algobot.strategies.my_strategy import MyStrategy

REGISTRY[MyStrategy.name] = MyStrategy
FAMILIES[MyStrategy.name] = "mean-reversion"
```

Or, from outside the package, use the decorator:

```python
from algobot.strategies import register

@register
class MyStrategy(Strategy):
    ...
```

It's now available to `--strategy my_strategy`, to the optimizer, to
walk-forward, to portfolios and to the config file. The parametrised test suite
picks it up automatically and holds it to the signal-shape, warmup and
causality contracts.

**The one rule:** the value at bar `t` may depend only on bars `≤ t`. Anything
else produces a beautiful equity curve that cannot be traded. For stateful
logic (enter on X, hold until Y), use `hold_between` rather than a Python loop:

```python
from algobot.indicators import hold_between

signal = hold_between(entry=value < 30, exit_=value >= 50, value=1.0)
```

Available indicators: `ema`, `sma`, `atr`, `true_range`, `rsi`, `macd`,
`bollinger`, `donchian`, `rolling_max`, `rolling_min`, `rolling_std`,
`hold_between`.

---

## Python API

The CLI is a thin wrapper; everything is callable directly.

```python
from algobot.backtest import run_backtest
from algobot.config import BacktestConfig, RiskConfig
from algobot.data import load
from algobot.strategies import get_strategy

bars = load(source="csv", path="examples/sample_bars.csv")
strategy = get_strategy("macd", fast=8, slow=26)

result = run_backtest(
    bars,
    strategy,
    BacktestConfig(initial_cash=50_000, commission_bps=5, slippage_bps=2),
    RiskConfig(risk_per_trade=0.01, stop_loss_atr=2.0),
    symbol="DEMO",
)

print(result.summary())
print(result.metrics.sharpe, result.metrics.max_drawdown)
result.trades.head()          # DataFrame of closed round trips
result.equity                 # Series, marked at every bar close
result.to_csv("results/")
```

```python
# Validation
from algobot.validation import walk_forward, monte_carlo

wf = walk_forward(bars, "macd", {"fast": [8, 12], "slow": [21, 26]}, n_splits=5)
print(wf.efficiency, wf.consistency)
print(wf.parameter_stability())

mc = monte_carlo(result.equity, result.trades, method="trades", trials=5_000)
print(mc.probability_of_profit, mc.percentiles())
```

```python
# Portfolio and reporting
from algobot.portfolio import load_many, run_portfolio
from algobot.report import write_backtest_report

data, failures = load_many(["ALFA", "BETA"], source="csv", directory="examples/portfolio")
portfolio = run_portfolio(data, "macd", weights={"ALFA": 2, "BETA": 1})
print(portfolio.correlation(), portfolio.diversification_ratio())

write_backtest_report(result, "report.html")
```

---

## Project layout

```
algobot/
  config.py              typed config, YAML loading, CLI override precedence
  indicators.py          causal EMA, SMA, ATR, RSI, MACD, Bollinger, Donchian
  search.py              grid expansion and scoring, shared by optimize + walkforward
  portfolio.py           multi-symbol sleeves, correlation, diversification
  report.py              self-contained HTML reports with inline SVG charts
  cli.py                 six commands, argparse
  data/
    loader.py            CSV / Yahoo / synthetic loaders, OHLCV normalisation
  strategies/
    base.py              the Strategy contract
    __init__.py          registry, families, register()
    ema_cross.py  macd_trend.py  donchian.py
    rsi_reversion.py  bollinger_reversion.py  buy_hold.py
  risk/
    manager.py           ATR position sizing, stops, drawdown kill-switch
  execution/
    broker.py            paper fills, cash accounting, trade ledger
  backtest/
    engine.py            the bar loop
    metrics.py           Sharpe, Sortino, drawdown, profit factor, exposure
  validation/
    walkforward.py       anchored/rolling folds, OOS stitching, stability
    montecarlo.py        trade and block-return bootstraps
tests/                   240 tests across 11 modules
examples/                sample bars for offline runs
  sample_bars.csv          single instrument
  portfolio/               ALFA.csv, BETA.csv, GAMA.csv
```

---

## Testing

```bash
pytest -q                    # 240 tests, ~4 seconds
ruff check . && ruff format --check .
```

The suite is built around properties rather than smoke tests:

- **No lookahead** — rewriting future bars cannot change past trades; a run over
  the first N bars reproduces the first N bars of a longer run. Parametrised
  over the registry, so new strategies inherit the check.
- **Exact accounting** — stops fill at 96.00, gaps fill at the open, ATR sizing
  buys exactly 25 units, cash reconciles with the trade ledger through
  reversals and partial closes.
- **Hand-computed metrics** — Sharpe, Sortino, drawdown duration and profit
  factor checked against values worked out by hand, not against themselves.
- **Offline by construction** — CI re-runs the whole suite with the network
  blackholed to prove no test quietly depends on a download.
- **Portable reports** — the HTML output is scanned for `http`, `script` and
  `src` tokens, so a report can never grow an external dependency.

CI runs on Python 3.10–3.13, plus an end-to-end smoke job that executes all six
commands and uploads the generated report as an artifact.

---

## Limitations

Stated plainly, because a backtester that hides these is lying to you:

- **No partial fills or order-book depth.** Every order fills completely at one
  price. Size a strategy near an instrument's real liquidity and the backtest
  will flatter it.
- **Intrabar path is unknown.** A bar is only OHLC. When both a stop and a
  target sit inside one bar, the engine assumes the stop hit first — pessimistic,
  but a guess either way.
- **No borrow costs, financing or dividends** beyond what's already in Yahoo's
  adjusted prices. Short strategies will look better than they are.
- **No margin model.** `max_position_pct > 1.0` gives you leverage with no
  financing cost and no margin call.
- **Survivorship bias is yours to manage.** The framework backtests the symbols
  you give it; if you picked them because they did well, the result is
  predetermined.
- **Portfolio sleeves are independent.** No shared cash pool, no cross-sectional
  selection, no rebalancing between symbols.
- **Live trading is not implemented.** The broker interface is designed for it,
  but no adapter ships yet.

Treat a backtest as evidence a strategy is *not* broken. It is never proof that
it works.

---

## Roadmap

- [ ] Live/paper execution adapter (Alpaca or CCXT) behind the existing broker interface
- [ ] Cross-sectional portfolio selection — hold the strongest N of M symbols
- [ ] Position sizing modes: volatility targeting, fractional Kelly
- [ ] Data caching layer so repeated backtests stop re-downloading
- [ ] Parameter sensitivity heatmaps in the HTML report

---

## Contributing

Pull requests are welcome. Please make sure `pytest -q` and `ruff check .` pass,
and that any new strategy satisfies the causality contract — the parametrised
tests will tell you if it doesn't.

## License

[MIT](LICENSE).

## Disclaimer

This is software for research and education. **Backtested performance is not
indicative of future results, and nothing here is financial advice.** Markets
can and do behave in ways no historical simulation anticipates. Do not risk
money you cannot afford to lose.
