"""Monte Carlo robustness analysis.

A backtest reports one path out of the many the same strategy could plausibly
have produced. Reshuffling its building blocks and re-running thousands of times
answers the questions the single path cannot:

    How much of that return came from the *order* the trades happened to arrive in?
    How bad could the drawdown have been with the same trades in a different sequence?
    What fraction of plausible paths lose money at all?

Two resampling methods, because they make different assumptions:

    trades    resample trade outcomes with replacement - tests sequence risk,
              and assumes trades are independent of one another
    returns   block-resample bar returns - preserves short-run autocorrelation
              and volatility clustering that an i.i.d. shuffle destroys

Neither invents information. If the backtest has 12 trades, no number of trials
turns that into a reliable estimate: it just measures the uncertainty honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_PERCENTILES = (5, 25, 50, 75, 95)


def _ordinal(value: float) -> str:
    n = int(round(value))
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass
class MonteCarloResult:
    """The distribution of outcomes across every simulated path."""

    method: str
    trials: int
    observations: int
    initial_equity: float
    final_equity: np.ndarray = field(default_factory=lambda: np.array([]))
    total_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    max_drawdowns: np.ndarray = field(default_factory=lambda: np.array([]))
    observed_return: float = 0.0
    observed_max_drawdown: float = 0.0
    ruin_threshold: float = 0.5

    @property
    def probability_of_profit(self) -> float:
        return float((self.total_returns > 0).mean()) if self.total_returns.size else 0.0

    @property
    def probability_of_ruin(self) -> float:
        """Share of paths that drew down past the ruin threshold."""
        if not self.max_drawdowns.size:
            return 0.0
        return float((self.max_drawdowns <= -abs(self.ruin_threshold)).mean())

    @property
    def observed_return_percentile(self) -> float:
        """Where the actual backtest sits in the simulated distribution."""
        if not self.total_returns.size:
            return float("nan")
        return float((self.total_returns < self.observed_return).mean() * 100)

    def percentiles(self, points=DEFAULT_PERCENTILES) -> pd.DataFrame:
        if not self.total_returns.size:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "percentile": [f"p{p}" for p in points],
                "total_return_pct": np.percentile(self.total_returns, points) * 100,
                "final_equity": np.percentile(self.final_equity, points),
                "max_drawdown_pct": np.percentile(self.max_drawdowns, points) * 100,
            }
        ).round(2)

    def summary(self) -> str:
        if not self.total_returns.size:
            return "Monte Carlo: not enough data to simulate."

        table = self.percentiles().to_string(index=False)
        line = "=" * 60
        return (
            f"{line}\nMonte Carlo robustness ({self.method}, {self.trials:,} trials)\n{line}\n"
            f"  Resampled from {self.observations} observation(s)\n"
            f"  Backtest actually returned {self.observed_return * 100:,.2f}% "
            f"with a {self.observed_max_drawdown * 100:,.2f}% drawdown\n"
            f"  That return sits at the {_ordinal(self.observed_return_percentile)} percentile "
            f"of simulated paths\n\n{table}\n\n"
            f"  Probability of profit: {self.probability_of_profit:.1%}\n"
            f"  Probability of a drawdown past {self.ruin_threshold:.0%}: "
            f"{self.probability_of_ruin:.1%}\n{line}"
        )


def _path_stats(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Final value and worst drawdown for each row of an equity-path matrix."""
    running_peak = np.maximum.accumulate(paths, axis=1)
    drawdowns = paths / running_peak - 1.0
    return paths[:, -1], drawdowns.min(axis=1)


def from_trades(
    trades: pd.DataFrame,
    equity: pd.Series,
    *,
    trials: int = 2_000,
    seed: int = 7,
    ruin_threshold: float = 0.5,
) -> MonteCarloResult:
    """Resample trade outcomes with replacement and rebuild the equity path.

    Each trade's profit is expressed as a return on the equity that was at risk
    when it opened, so the resampled path compounds the way a percentage-risk
    system actually does rather than re-adding fixed dollar amounts.
    """
    initial = float(equity.iloc[0]) if len(equity) else 0.0
    result = MonteCarloResult(
        method="trades",
        trials=trials,
        observations=0,
        initial_equity=initial,
        ruin_threshold=ruin_threshold,
    )
    if trades is None or not len(trades) or not len(equity):
        return result

    # Equity as it stood when each trade opened.
    equity_at_entry = (
        equity.reindex(equity.index.union(pd.DatetimeIndex(trades["entry_time"])))
        .ffill()
        .reindex(pd.DatetimeIndex(trades["entry_time"]))
    )
    basis = equity_at_entry.to_numpy(dtype=float)
    basis = np.where(np.isfinite(basis) & (basis > 0), basis, initial)

    trade_returns = trades["pnl"].to_numpy(dtype=float) / basis
    n = trade_returns.size

    rng = np.random.default_rng(seed)
    draws = rng.choice(trade_returns, size=(trials, n), replace=True)
    paths = initial * np.cumprod(1.0 + draws, axis=1)
    # Equity cannot go below zero; a blown-up path stays blown up.
    paths = np.maximum(paths, 0.0)

    final, drawdowns = _path_stats(np.column_stack([np.full(trials, initial), paths]))

    result.observations = n
    result.final_equity = final
    result.total_returns = final / initial - 1.0
    result.max_drawdowns = drawdowns
    result.observed_return = float(equity.iloc[-1] / initial - 1.0)
    result.observed_max_drawdown = float((equity / equity.cummax() - 1.0).min())
    return result


def from_returns(
    equity: pd.Series,
    *,
    trials: int = 2_000,
    block: int = 10,
    seed: int = 7,
    ruin_threshold: float = 0.5,
) -> MonteCarloResult:
    """Block-bootstrap the bar returns of an equity curve.

    Sampling in contiguous blocks rather than one bar at a time keeps the
    volatility clustering and short-run autocorrelation that make real drawdowns
    deeper than an independent shuffle would ever produce.
    """
    initial = float(equity.iloc[0]) if len(equity) else 0.0
    result = MonteCarloResult(
        method=f"returns (block={block})",
        trials=trials,
        observations=0,
        initial_equity=initial,
        ruin_threshold=ruin_threshold,
    )
    returns = equity.pct_change().dropna().to_numpy(dtype=float)
    if returns.size < 2:
        return result
    if block < 1:
        raise ValueError("block must be >= 1")

    block = min(block, returns.size)
    n = returns.size
    n_blocks = int(np.ceil(n / block))

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(trials, n_blocks))
    offsets = np.arange(block)
    # (trials, n_blocks, block) -> flatten to a return series per trial
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(trials, -1)[:, :n]
    draws = returns[indices]

    paths = initial * np.cumprod(1.0 + draws, axis=1)
    paths = np.maximum(paths, 0.0)
    final, drawdowns = _path_stats(np.column_stack([np.full(trials, initial), paths]))

    result.observations = n
    result.final_equity = final
    result.total_returns = final / initial - 1.0
    result.max_drawdowns = drawdowns
    result.observed_return = float(equity.iloc[-1] / initial - 1.0)
    result.observed_max_drawdown = float((equity / equity.cummax() - 1.0).min())
    return result


def monte_carlo(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    *,
    method: str = "trades",
    trials: int = 2_000,
    block: int = 10,
    seed: int = 7,
    ruin_threshold: float = 0.5,
) -> MonteCarloResult:
    """Dispatch to the requested resampling method."""
    if method == "trades":
        return from_trades(
            trades if trades is not None else pd.DataFrame(),
            equity,
            trials=trials,
            seed=seed,
            ruin_threshold=ruin_threshold,
        )
    if method == "returns":
        return from_returns(
            equity, trials=trials, block=block, seed=seed, ruin_threshold=ruin_threshold
        )
    raise ValueError(f"unknown method {method!r}; expected 'trades' or 'returns'")
