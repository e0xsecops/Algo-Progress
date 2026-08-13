"""Self-contained HTML reports.

Charts are generated as inline SVG rather than through a plotting library, so a
report is a single file with no external requests, no CDN and no runtime
dependency beyond what already produced the numbers. Open it anywhere, mail it
to anyone, keep it as an archive of exactly what a run produced.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from algobot.backtest.metrics import Metrics, drawdown_series

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MAX_POINTS = 1_500  # plenty for a chart; keeps the file small on decade-long runs


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------


@dataclass
class Box:
    """Plot area inside an SVG canvas."""

    width: int = 720
    height: int = 260
    left: int = 56
    right: int = 12
    top: int = 12
    bottom: int = 28

    @property
    def plot_width(self) -> float:
        return self.width - self.left - self.right

    @property
    def plot_height(self) -> float:
        return self.height - self.top - self.bottom


def _downsample(series: pd.Series, limit: int = MAX_POINTS) -> pd.Series:
    if len(series) <= limit:
        return series
    step = math.ceil(len(series) / limit)
    thinned = series.iloc[::step]
    # Always keep the final point so the chart ends where the run ended.
    if thinned.index[-1] != series.index[-1]:
        thinned = pd.concat([thinned, series.iloc[[-1]]])
    return thinned


def _path(series: pd.Series, box: Box, lo: float, hi: float) -> str:
    """Turn a series into an SVG path, scaled into the plot area."""
    if series.empty:
        return ""
    span = hi - lo or 1.0
    n = len(series)
    points = []
    for i, value in enumerate(series.to_numpy(dtype=float)):
        x = box.left + (i / max(1, n - 1)) * box.plot_width
        y = box.top + (1.0 - (value - lo) / span) * box.plot_height
        points.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(points)


def _y_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / count
    return [lo + i * step for i in range(count + 1)]


def _grid(box: Box, lo: float, hi: float, fmt) -> str:
    parts = []
    span = hi - lo or 1.0
    for tick in _y_ticks(lo, hi):
        y = box.top + (1.0 - (tick - lo) / span) * box.plot_height
        parts.append(
            f'<line class="grid" x1="{box.left}" y1="{y:.1f}" '
            f'x2="{box.width - box.right}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{box.left - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end">{html.escape(fmt(tick))}</text>'
        )
    return "".join(parts)


def _x_labels(series: pd.Series, box: Box) -> str:
    if series.empty:
        return ""
    y = box.height - 8
    first, last = series.index[0], series.index[-1]
    middle = series.index[len(series) // 2]
    label = lambda ts: html.escape(str(pd.Timestamp(ts).date()))  # noqa: E731
    return (
        f'<text class="tick" x="{box.left}" y="{y}">{label(first)}</text>'
        f'<text class="tick" x="{box.left + box.plot_width / 2:.0f}" y="{y}" '
        f'text-anchor="middle">{label(middle)}</text>'
        f'<text class="tick" x="{box.width - box.right}" y="{y}" '
        f'text-anchor="end">{label(last)}</text>'
    )


def equity_chart(equity: pd.Series, benchmark: pd.Series | None = None) -> str:
    """Equity curve, optionally against a benchmark, as an SVG figure."""
    if equity.empty:
        return "<p class='empty'>No equity curve to plot.</p>"

    box = Box()
    series = _downsample(equity)
    lines = [("equity", series)]
    if benchmark is not None and not benchmark.empty:
        lines.append(("benchmark", _downsample(benchmark.reindex(series.index).ffill())))

    values = pd.concat([s for _, s in lines])
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.05 or abs(hi) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad

    paths = "".join(
        f'<path class="{name}" d="{_path(s, box, lo, hi)}"/>' for name, s in lines
    )
    legend = (
        '<span class="key equity">strategy</span>'
        + ('<span class="key benchmark">buy &amp; hold</span>' if len(lines) > 1 else "")
    )

    return (
        f'<div class="chart"><svg viewBox="0 0 {box.width} {box.height}" '
        f'preserveAspectRatio="none" role="img" aria-label="Equity curve">'
        f"{_grid(box, lo, hi, lambda v: f'{v:,.0f}')}{paths}"
        f"{_x_labels(series, box)}</svg><div class='legend'>{legend}</div></div>"
    )


def drawdown_chart(equity: pd.Series) -> str:
    """Underwater plot: how far below the previous peak the account sat."""
    if equity.empty:
        return ""

    box = Box(height=180)
    series = _downsample(drawdown_series(equity) * 100)
    lo = float(series.min())
    lo = min(lo * 1.1, -0.5)
    hi = 0.0

    line = _path(series, box, lo, hi)
    # Close the path along the zero line to fill the underwater area.
    baseline_y = box.top + box.plot_height
    area = f"{line} L{box.left + box.plot_width:.1f},{baseline_y:.1f} L{box.left:.1f},{baseline_y:.1f} Z"

    return (
        f'<div class="chart"><svg viewBox="0 0 {box.width} {box.height}" '
        f'preserveAspectRatio="none" role="img" aria-label="Drawdown">'
        f"{_grid(box, lo, hi, lambda v: f'{v:.0f}%')}"
        f'<path class="dd-area" d="{area}"/><path class="dd-line" d="{line}"/>'
        f"{_x_labels(series, box)}</svg></div>"
    )


def trade_histogram(trades: pd.DataFrame, bins: int = 25) -> str:
    """Distribution of trade P&L, losses left of zero and wins right."""
    if trades is None or not len(trades):
        return ""

    pnl = trades["pnl"].to_numpy(dtype=float)
    counts, edges = np.histogram(pnl, bins=min(bins, max(5, len(pnl) // 2)))
    if not counts.size:
        return ""

    box = Box(height=200, left=44)
    top = int(counts.max()) or 1
    width = box.plot_width / len(counts)

    bars = []
    for i, count in enumerate(counts):
        height = (count / top) * box.plot_height
        x = box.left + i * width
        y = box.top + box.plot_height - height
        losing = edges[i + 1] <= 0
        bars.append(
            f'<rect class="{"loss" if losing else "win"}" x="{x:.1f}" y="{y:.1f}" '
            f'width="{max(1.0, width - 1):.1f}" height="{height:.1f}"/>'
        )

    zero_x = box.left + (0 - edges[0]) / (edges[-1] - edges[0] or 1) * box.plot_width
    zero = (
        f'<line class="zero" x1="{zero_x:.1f}" y1="{box.top}" '
        f'x2="{zero_x:.1f}" y2="{box.top + box.plot_height:.1f}"/>'
        if edges[0] < 0 < edges[-1]
        else ""
    )

    return (
        f'<div class="chart"><svg viewBox="0 0 {box.width} {box.height}" '
        f'role="img" aria-label="Trade profit and loss distribution">'
        f"{_grid(box, 0, top, lambda v: f'{v:.0f}')}{''.join(bars)}{zero}"
        f'<text class="tick" x="{box.left}" y="{box.height - 8}">{edges[0]:,.0f}</text>'
        f'<text class="tick" x="{box.width - box.right}" y="{box.height - 8}" '
        f'text-anchor="end">{edges[-1]:,.0f}</text></svg></div>'
    )


def monthly_table(equity: pd.Series) -> str:
    """Calendar of monthly returns, shaded by size."""
    if equity.empty or len(equity) < 2:
        return ""

    monthly = equity.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return ""

    yearly = equity.resample("YE").last()
    yearly = pd.concat([equity.iloc[[0]], yearly]).pct_change().dropna()

    scale = float(monthly.abs().quantile(0.9)) or 1.0
    rows = []
    for year, group in monthly.groupby(monthly.index.year):
        cells = []
        by_month = {ts.month: value for ts, value in group.items()}
        for month in range(1, 13):
            value = by_month.get(month)
            if value is None:
                cells.append('<td class="void"></td>')
                continue
            intensity = min(1.0, abs(value) / scale)
            tone = "pos" if value >= 0 else "neg"
            cells.append(
                f'<td class="{tone}" style="--i:{intensity:.2f}">{value * 100:,.1f}</td>'
            )
        total = next(
            (v for ts, v in yearly.items() if ts.year == year), None
        )
        total_cell = (
            f'<td class="total {"pos" if total >= 0 else "neg"}">{total * 100:,.1f}</td>'
            if total is not None
            else '<td class="void"></td>'
        )
        rows.append(f"<tr><th>{year}</th>{''.join(cells)}{total_cell}</tr>")

    header = "".join(f"<th>{m}</th>" for m in MONTHS)
    return (
        '<table class="calendar"><thead><tr><th></th>'
        f"{header}<th>Year</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _metric_cards(m: Metrics, benchmark: Metrics | None) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:,.2f}%"

    cards = [
        ("Total return", pct(m.total_return), m.total_return >= 0),
        ("CAGR", pct(m.cagr), m.cagr >= 0),
        ("Sharpe", f"{m.sharpe:.2f}", m.sharpe >= 1),
        ("Max drawdown", pct(m.max_drawdown), m.max_drawdown > -0.2),
        ("Win rate", pct(m.win_rate), m.win_rate >= 0.5),
        (
            "Profit factor",
            "inf" if math.isinf(m.profit_factor) else f"{m.profit_factor:.2f}",
            m.profit_factor >= 1,
        ),
        ("Trades", f"{m.num_trades:,}", True),
        ("Time in market", pct(m.exposure), True),
    ]
    if benchmark is not None:
        edge = m.total_return - benchmark.total_return
        cards.append(("Edge vs buy &amp; hold", f"{edge * 100:+,.2f}%", edge >= 0))

    return "".join(
        f'<div class="card"><span class="label">{label}</span>'
        f'<span class="value {"good" if good else "bad"}">{value}</span></div>'
        for label, value, good in cards
    )


def _detail_table(m: Metrics, benchmark: Metrics | None) -> str:
    def pct(v: float) -> str:
        return f"{v * 100:,.2f}%"

    rows = [
        ("Period", f"{m.start.date() if m.start is not None else '-'} to "
                   f"{m.end.date() if m.end is not None else '-'} ({m.years:.2f} years)"),
        ("Starting equity", f"{m.initial_equity:,.2f}"),
        ("Final equity", f"{m.final_equity:,.2f}"),
        ("Annual volatility", pct(m.annual_volatility)),
        ("Sortino", "inf" if math.isinf(m.sortino) else f"{m.sortino:.2f}"),
        ("Calmar", f"{m.calmar:.2f}"),
        ("Longest drawdown", f"{m.max_drawdown_bars:,} bars underwater"),
        ("Expectancy per trade", f"{m.expectancy:,.2f}"),
        ("Average win", f"{m.avg_win:,.2f}"),
        ("Average loss", f"{m.avg_loss:,.2f}"),
        ("Best trade", f"{m.best_trade:,.2f}"),
        ("Worst trade", f"{m.worst_trade:,.2f}"),
        ("Average holding period", f"{m.avg_bars_held:.1f} bars"),
        ("Commission paid", f"{m.total_commission:,.2f}"),
        ("Slippage paid", f"{m.total_slippage:,.2f}"),
    ]
    if benchmark is not None:
        rows += [
            ("Benchmark return", pct(benchmark.total_return)),
            ("Benchmark Sharpe", f"{benchmark.sharpe:.2f}"),
            ("Benchmark max drawdown", pct(benchmark.max_drawdown)),
        ]
    body = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
    )
    return f'<table class="detail"><tbody>{body}</tbody></table>'


def _trades_table(trades: pd.DataFrame, limit: int = 50) -> str:
    if trades is None or not len(trades):
        return "<p class='empty'>This run closed no trades.</p>"

    shown = trades.tail(limit)
    columns = [c for c in
               ["symbol", "entry_time", "exit_time", "direction", "quantity",
                "entry_price", "exit_price", "pnl", "return_pct", "bars_held", "exit_reason"]
               if c in shown.columns]

    header = "".join(f"<th>{html.escape(c.replace('_', ' '))}</th>" for c in columns)
    rows = []
    for _, trade in shown.iterrows():
        cells = []
        for column in columns:
            value = trade[column]
            if column == "return_pct":
                text = f"{float(value) * 100:,.2f}%"
            elif column == "direction":
                text = "long" if value > 0 else "short"
            elif column == "bars_held":
                text = f"{int(value):,}"
            elif isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                text = f"{float(value):,.2f}"
            else:
                text = str(pd.Timestamp(value).date()) if "time" in column else str(value)
            # Tag the P&L columns by name rather than by position: a portfolio
            # report inserts a symbol column and shifts every index along.
            css = ' class="pnl"' if column in ("pnl", "return_pct") else ""
            cells.append(f"<td{css}>{html.escape(text)}</td>")
        tone = "win" if trade["pnl"] > 0 else "loss"
        rows.append(f'<tr class="{tone}">{"".join(cells)}</tr>')

    note = (
        f"<p class='note'>Showing the last {len(shown)} of {len(trades)} trades.</p>"
        if len(trades) > limit
        else ""
    )
    return (
        f'<table class="trades"><thead><tr>{header}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>{note}"
    )


STYLE = """
:root{--bg:#fbfbfd;--panel:#fff;--ink:#16181d;--muted:#6b7280;--line:#e5e7eb;
--good:#0f8a5f;--bad:#c0392b;--accent:#2563eb;--bench:#9aa3af;--shadow:0 1px 3px rgba(0,0,0,.06)}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--panel:#161a21;--ink:#e8eaed;
--muted:#9aa3af;--line:#252b36;--good:#34d399;--bad:#f87171;--accent:#60a5fa;
--bench:#6b7280;--shadow:0 1px 3px rgba(0,0,0,.4)}}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
margin:34px 0 12px;font-weight:600}
.sub{color:var(--muted);margin:0 0 8px;font-size:13px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:16px;box-shadow:var(--shadow);overflow-x:auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:12px 14px;box-shadow:var(--shadow)}
.card .label{display:block;font-size:12px;color:var(--muted);margin-bottom:3px}
.card .value{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.value.good{color:var(--good)}.value.bad{color:var(--bad)}
.chart svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font-size:10px;font-family:inherit}
path.equity{fill:none;stroke:var(--accent);stroke-width:1.8;vector-effect:non-scaling-stroke}
path.benchmark{fill:none;stroke:var(--bench);stroke-width:1.4;stroke-dasharray:4 3;
vector-effect:non-scaling-stroke}
.dd-area{fill:var(--bad);opacity:.16}
.dd-line{fill:none;stroke:var(--bad);stroke-width:1.4;vector-effect:non-scaling-stroke}
rect.win{fill:var(--good);opacity:.75}rect.loss{fill:var(--bad);opacity:.75}
.zero{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.legend{margin-top:8px;font-size:12px;color:var(--muted)}
.key{margin-right:14px}
.key::before{content:"";display:inline-block;width:14px;height:2px;margin-right:6px;
vertical-align:middle}
.key.equity::before{background:var(--accent)}
.key.benchmark::before{background:var(--bench)}
table{border-collapse:collapse;width:100%;font-size:13px;
font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);
white-space:nowrap}
thead th{color:var(--muted);font-weight:600;font-size:12px}
.detail th{text-align:left;font-weight:500;color:var(--muted)}
.trades tbody tr.win td.pnl{color:var(--good)}
.trades tbody tr.loss td.pnl{color:var(--bad)}
.calendar th:first-child{text-align:left}
.calendar td{position:relative}
.calendar td.pos{background:color-mix(in srgb,var(--good) calc(var(--i)*55%),transparent)}
.calendar td.neg{background:color-mix(in srgb,var(--bad) calc(var(--i)*55%),transparent)}
.calendar td.total{font-weight:600;border-left:1px solid var(--line)}
.calendar td.void{background:none}
.empty,.note{color:var(--muted);font-size:13px;margin:8px 0 0}
footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);
color:var(--muted);font-size:12px}
"""


def build_html(
    *,
    title: str,
    subtitle: str,
    metrics: Metrics,
    equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    benchmark: Metrics | None = None,
    trades: pd.DataFrame | None = None,
    extra_sections: list[tuple[str, str]] | None = None,
    footnote: str = "",
) -> str:
    """Assemble a complete standalone HTML document."""
    sections = [
        ("Performance", f'<div class="cards">{_metric_cards(metrics, benchmark)}</div>'),
        ("Equity curve", f'<div class="panel">{equity_chart(equity, benchmark_equity)}</div>'),
        ("Drawdown", f'<div class="panel">{drawdown_chart(equity)}</div>'),
        ("Monthly returns %", f'<div class="panel">{monthly_table(equity)}</div>'),
        ("Trade distribution", f'<div class="panel">{trade_histogram(trades)}</div>')
        if trades is not None and len(trades)
        else None,
        ("Details", f'<div class="panel">{_detail_table(metrics, benchmark)}</div>'),
        ("Trades", f'<div class="panel">{_trades_table(trades)}</div>'),
    ]
    sections = [s for s in sections if s]
    sections.extend(extra_sections or [])

    body = "".join(
        f"<h2>{html.escape(name)}</h2>{content}" for name, content in sections
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>"
        f"<div class='wrap'><h1>{html.escape(title)}</h1>"
        f"<p class='sub'>{html.escape(subtitle)}</p>{body}"
        f"<footer>{html.escape(footnote)}</footer></div></body></html>"
    )


def write_backtest_report(result, path: str | Path, title: str | None = None) -> Path:
    """Render a :class:`~algobot.backtest.BacktestResult` to an HTML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    symbol = result.symbol or "backtest"
    meta = getattr(result, "meta", {}) or {}
    subtitle = result.strategy
    if meta:
        subtitle += (
            f"  |  {meta.get('bars', '?')} bars, fill at {meta.get('fill', '?')}"
            f"  |  {meta.get('risk', '')}"
        )

    document = build_html(
        title=title or f"{symbol} - backtest report",
        subtitle=subtitle,
        metrics=result.metrics,
        equity=result.equity,
        benchmark_equity=getattr(result, "benchmark_equity", None),
        benchmark=getattr(result, "benchmark", None),
        trades=result.trades,
        footnote=(
            "Generated by algobot. Backtested performance is not indicative of "
            "future results. Costs, slippage and survivorship assumptions all "
            "affect these numbers."
        ),
    )
    path.write_text(document, encoding="utf-8")
    return path


def write_portfolio_report(result, path: str | Path, title: str | None = None) -> Path:
    """Render a :class:`~algobot.portfolio.PortfolioResult` to an HTML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    contributions = result.contributions()
    correlation = result.correlation()
    extra = [
        (
            "Per-symbol contribution",
            f'<div class="panel">{_frame_table(contributions)}</div>',
        )
    ]
    if not correlation.empty:
        extra.append(
            (
                "Sleeve correlation",
                f'<div class="panel">{_frame_table(correlation.round(2), index=True)}</div>'
                f'<p class="note">Diversification ratio '
                f"{result.diversification_ratio():.2f} - above 1.0 means the sleeves "
                "offset each other.</p>",
            )
        )

    document = build_html(
        title=title or f"Portfolio report - {len(result.sleeves)} symbols",
        subtitle=f"{result.strategy}  |  {', '.join(result.symbols)}",
        metrics=result.metrics,
        equity=result.equity,
        trades=result.trades,
        extra_sections=extra,
        footnote=(
            "Generated by algobot. Sleeves are funded once and run independently: "
            "no shared cash, no rebalancing between symbols."
        ),
    )
    path.write_text(document, encoding="utf-8")
    return path


def _frame_table(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a plain HTML table, escaping every cell."""
    if frame is None or frame.empty:
        return "<p class='empty'>Nothing to show.</p>"

    columns = list(frame.columns)
    header = ("<th></th>" if index else "") + "".join(
        f"<th>{html.escape(str(c).replace('_', ' '))}</th>" for c in columns
    )
    rows = []
    for label, row in frame.iterrows():
        cells = "".join(
            f"<td>{html.escape(f'{v:,.2f}' if isinstance(v, (float, np.floating)) else str(v))}</td>"
            for v in row
        )
        prefix = f"<th>{html.escape(str(label))}</th>" if index else ""
        rows.append(f"<tr>{prefix}{cells}</tr>")

    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
