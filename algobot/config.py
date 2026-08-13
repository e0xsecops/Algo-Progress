"""Typed configuration objects loaded from YAML and overridable from the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "yahoo"
    symbol: str = "SPY"
    start: str | None = "2015-01-01"
    end: str | None = None
    interval: str = "1d"
    path: str | None = None


@dataclass
class StrategyConfig:
    name: str = "ema_cross"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 2.0
    fill: str = "next_open"

    def __post_init__(self) -> None:
        if self.fill not in ("next_open", "close"):
            raise ValueError(f"fill must be 'next_open' or 'close', got {self.fill!r}")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("commission_bps and slippage_bps must be non-negative")


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_position_pct: float = 1.0
    stop_loss_atr: float | None = 2.0
    take_profit_atr: float | None = None
    atr_period: int = 14
    max_drawdown_stop: float | None = 0.25

    def __post_init__(self) -> None:
        if not 0 <= self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be between 0 and 1")
        if self.max_position_pct <= 0:
            raise ValueError("max_position_pct must be positive")
        if self.atr_period < 1:
            raise ValueError("atr_period must be >= 1")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


_SECTIONS = {
    "data": DataConfig,
    "strategy": StrategyConfig,
    "backtest": BacktestConfig,
    "risk": RiskConfig,
}


def _build_section(cls: type, raw: dict[str, Any] | None) -> Any:
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in '{cls.__name__}' section; "
            f"expected any of {sorted(known)}"
        )
    return cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    """Load a :class:`Config` from YAML, falling back to the dataclass defaults."""
    if path is None:
        return Config()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a mapping, got {type(raw).__name__}")

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config section(s): {sorted(unknown)}")

    return Config(**{name: _build_section(cls, raw.get(name)) for name, cls in _SECTIONS.items()})


def apply_overrides(config: Config, **overrides: Any) -> Config:
    """Return a copy of *config* with any non-``None`` overrides applied.

    Keys are matched against ``<section>_<field>`` first (e.g. ``data_symbol``)
    and then against bare field names that are unambiguous across sections
    (e.g. ``symbol``), which is what the CLI flags map onto.
    """
    sections = {name: getattr(config, name) for name in _SECTIONS}
    updates: dict[str, dict[str, Any]] = {name: {} for name in _SECTIONS}

    for key, value in overrides.items():
        if value is None:
            continue
        target = None
        for name, section in sections.items():
            names = {f.name for f in fields(section)}
            if key.startswith(f"{name}_") and key[len(name) + 1 :] in names:
                target = (name, key[len(name) + 1 :])
                break
            if key in names:
                target = (name, key)
                break
        if target is None:
            raise ValueError(f"override {key!r} does not match any config field")
        updates[target[0]][target[1]] = value

    return Config(
        **{
            name: replace(section, **updates[name]) if updates[name] else section
            for name, section in sections.items()
        }
    )
