"""Strategy contract, registry and the EMA crossover logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.indicators import ema
from algobot.strategies import REGISTRY, available, get_strategy
from algobot.strategies.base import Strategy


def test_registry_exposes_the_shipped_strategies():
    assert available() == ["buy_hold", "ema_cross"]
    assert all(issubclass(cls, Strategy) for cls in REGISTRY.values())


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        get_strategy("moon_phase")


def test_parameters_default_and_override():
    s = get_strategy("ema_cross")
    assert s.fast == 20 and s.slow == 50

    s = get_strategy("ema_cross", fast=5, slow=15)
    assert s.fast == 5 and s.slow == 15
    assert "fast=5" in s.describe()


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown parameter"):
        get_strategy("ema_cross", lookback=10)


def test_invalid_parameter_combinations_are_rejected():
    with pytest.raises(ValueError, match="must be shorter"):
        get_strategy("ema_cross", fast=50, slow=20)
    with pytest.raises(ValueError, match=">= 1"):
        get_strategy("ema_cross", fast=0, slow=20)


def test_missing_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        get_strategy("ema_cross").nonexistent


def test_signals_are_bounded_and_aligned(bars):
    signals = get_strategy("ema_cross", fast=10, slow=30).generate_signals(bars)

    assert signals.index.equals(bars.index)
    assert set(np.unique(signals)) <= {-1.0, 0.0, 1.0}
    assert not signals.isna().any()


def test_no_position_during_warmup(bars):
    strategy = get_strategy("ema_cross", fast=10, slow=30, trend_filter=0)
    signals = strategy.generate_signals(bars)
    assert (signals.iloc[: strategy.warmup - 1] == 0).all()


def test_long_when_fast_ema_leads_slow(trending_bars):
    strategy = get_strategy("ema_cross", fast=5, slow=20, trend_filter=0)
    signals = strategy.generate_signals(trending_bars)
    indicators = strategy.indicators(trending_bars)

    bullish = (indicators["ema_fast"] > indicators["ema_slow"]) & indicators["ema_slow"].notna()
    assert (signals[bullish] == 1.0).all()
    assert (signals[~bullish] == 0.0).all()
    # The fixture trends up then down, so both states occur.
    assert bullish.any() and (~bullish).any()


def test_long_only_by_default_and_short_when_enabled(trending_bars):
    long_only = get_strategy("ema_cross", fast=5, slow=20, trend_filter=0)
    assert (long_only.generate_signals(trending_bars) >= 0).all()

    both = get_strategy("ema_cross", fast=5, slow=20, trend_filter=0, allow_short=True)
    assert (both.generate_signals(trending_bars) < 0).any()


def test_trend_filter_blocks_longs_below_the_trend_line(bars):
    unfiltered = get_strategy("ema_cross", fast=10, slow=30, trend_filter=0)
    filtered = get_strategy("ema_cross", fast=10, slow=30, trend_filter=100)

    a = unfiltered.generate_signals(bars)
    b = filtered.generate_signals(bars)

    assert (b.abs() <= a.abs()).all()  # the filter only ever removes positions
    assert b.abs().sum() < a.abs().sum()

    trend = ema(bars["close"], 100)
    assert (bars["close"][b > 0] > trend[b > 0]).all()


def test_warmup_accounts_for_every_indicator():
    assert get_strategy("ema_cross", fast=10, slow=30, trend_filter=200).warmup == 200
    assert get_strategy("ema_cross", fast=10, slow=30, trend_filter=0).warmup == 30


def test_buy_hold_is_always_long(bars):
    signals = get_strategy("buy_hold").generate_signals(bars)
    assert (signals == 1.0).all()


def test_strategies_are_causal(bars):
    """A strategy that reacts to future bars would silently invalidate everything."""
    strategy = get_strategy("ema_cross", fast=10, slow=30, trend_filter=50)
    original = strategy.generate_signals(bars)

    tampered = bars.copy()
    tampered.iloc[-100:, :] *= 4.0

    pd.testing.assert_series_equal(
        original.iloc[:-100], strategy.generate_signals(tampered).iloc[:-100]
    )
