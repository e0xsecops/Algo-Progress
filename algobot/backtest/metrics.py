"""Performance statistics for an equity curve and its trade ledger."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Estimate how many bars make up a year, from the index spacing."""
    if len(index) < 3:
        return float(TRADING_DAYS)
    deltas = pd.Series(index[1:]) - pd.Series(index[:-1])
    median = deltas.median()
    seconds = median.total_seconds() if hasattr(median, "total_seconds") else 0.0
    if seconds <= 0:
        return float(TRADING_DAYS)
    if seconds >= 86_400:
        # Daily or slower: count calendar-day spacing against a 252-day year.
        return max(1.0, TRADING_DAYS / (seconds / 86_400))
    # Intraday: assume a 6.5-hour session, 252 days a year.
    return (6.5 * 3600 / seconds) * TRADING_DAYS


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak, as a negative series."""
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(drawdown_series(equity).min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest run of bars spent below a previous peak."""
    if equity.empty:
        return 0
    underwater = equity < equity.cummax()
    longest = current = 0
    for flag in underwater.to_numpy():
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def sharpe_ratio(returns: pd.Series, periods_per_year: float, risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio; *risk_free* is an annual rate."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = excess.std(ddof=1)
    if not std or not np.isfinite(std):
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float, risk_free: float = 0.0) -> float:
    """Sharpe's downside-only cousin: penalises losses, ignores upside vol."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf") if excess.mean() > 0 else 0.0
    dd = math.sqrt((downside**2).mean())
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * math.sqrt(periods_per_year))


@dataclass
class Metrics:
    """Everything the report prints, in one flat record."""

    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    years: float = 0.0
    initial_equity: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    annual_volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_bars: int = 0
    calmar: float = 0.0
    exposure: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_bars_held: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    *,
    exposure: pd.Series | None = None,
    risk_free: float = 0.0,
    total_commission: float = 0.0,
    total_slippage: float = 0.0,
) -> Metrics:
    """Score an equity curve, optionally enriched with trade-level stats."""
    equity = equity.dropna()
    m = Metrics(total_commission=total_commission, total_slippage=total_slippage)
    if equity.empty:
        return m

    m.start = equity.index[0]
    m.end = equity.index[-1]
    m.initial_equity = float(equity.iloc[0])
    m.final_equity = float(equity.iloc[-1])
    m.total_return = m.final_equity / m.initial_equity - 1.0 if m.initial_equity else 0.0

    span_days = (m.end - m.start).total_seconds() / 86_400 if len(equity) > 1 else 0.0
    m.years = span_days / 365.25
    if m.years > 0 and m.initial_equity > 0 and m.final_equity > 0:
        m.cagr = (m.final_equity / m.initial_equity) ** (1.0 / m.years) - 1.0

    returns = equity.pct_change().dropna()
    ppy = infer_periods_per_year(equity.index)
    if len(returns) > 1:
        m.annual_volatility = float(returns.std(ddof=1) * math.sqrt(ppy))
    m.sharpe = sharpe_ratio(returns, ppy, risk_free)
    m.sortino = sortino_ratio(returns, ppy, risk_free)
    m.max_drawdown = max_drawdown(equity)
    m.max_drawdown_bars = max_drawdown_duration(equity)
    if m.max_drawdown < 0:
        m.calmar = m.cagr / abs(m.max_drawdown)

    if exposure is not None and len(exposure):
        m.exposure = float((exposure.abs() > 1e-9).mean())

    if trades is not None and len(trades):
        pnl = trades["pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        m.num_trades = int(len(pnl))
        m.win_rate = float(len(wins) / len(pnl))
        gross_loss = float(-losses.sum())
        m.profit_factor = float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf")
        m.expectancy = float(pnl.mean())
        m.avg_win = float(wins.mean()) if len(wins) else 0.0
        m.avg_loss = float(losses.mean()) if len(losses) else 0.0
        m.best_trade = float(pnl.max())
        m.worst_trade = float(pnl.min())
        if "bars_held" in trades:
            m.avg_bars_held = float(trades["bars_held"].astype(float).mean())

    return m


def _fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def format_metrics(m: Metrics, title: str = "Backtest results") -> str:
    """Render metrics as a fixed-width report block."""
    start = m.start.date() if m.start is not None else "-"
    end = m.end.date() if m.end is not None else "-"
    pf = "inf" if math.isinf(m.profit_factor) else f"{m.profit_factor:.2f}"
    sortino = "inf" if math.isinf(m.sortino) else f"{m.sortino:.2f}"

    rows = [
        ("Period", f"{start} to {end}  ({m.years:.2f} years)"),
        ("Equity", f"{m.initial_equity:,.2f} -> {m.final_equity:,.2f}"),
        ("Total return", _fmt_pct(m.total_return)),
        ("CAGR", _fmt_pct(m.cagr)),
        ("Volatility (ann.)", _fmt_pct(m.annual_volatility)),
        ("Sharpe", f"{m.sharpe:.2f}"),
        ("Sortino", sortino),
        ("Max drawdown", f"{_fmt_pct(m.max_drawdown)}  ({m.max_drawdown_bars} bars underwater)"),
        ("Calmar", f"{m.calmar:.2f}"),
        ("Time in market", _fmt_pct(m.exposure)),
        ("Trades", f"{m.num_trades}"),
        ("Win rate", _fmt_pct(m.win_rate)),
        ("Profit factor", pf),
        ("Expectancy/trade", f"{m.expectancy:,.2f}"),
        ("Avg win / avg loss", f"{m.avg_win:,.2f} / {m.avg_loss:,.2f}"),
        ("Best / worst trade", f"{m.best_trade:,.2f} / {m.worst_trade:,.2f}"),
        ("Avg bars held", f"{m.avg_bars_held:.1f}"),
        ("Costs paid", f"commission {m.total_commission:,.2f}, slippage {m.total_slippage:,.2f}"),
    ]

    width = max(len(k) for k, _ in rows)
    body = "\n".join(f"  {k.ljust(width)}  {v}" for k, v in rows)
    line = "=" * max(len(title), 60)
    return f"{line}\n{title}\n{line}\n{body}\n{line}"
