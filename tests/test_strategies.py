"""Strategy contract, registry and the EMA crossover logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.indicators import ema
from algobot.strategies import FAMILIES, REGISTRY, available, get_strategy, register
from algobot.strategies.base import Strategy

ALL_STRATEGIES = available()


def test_registry_exposes_the_shipped_strategies():
    assert ALL_STRATEGIES == [
        "bollinger",
        "buy_hold",
        "donchian",
        "ema_cross",
        "macd",
        "rsi_reversion",
    ]
    assert all(issubclass(cls, Strategy) for cls in REGISTRY.values())
    assert set(FAMILIES) == set(REGISTRY)


def test_register_rejects_non_strategies_and_duplicate_names():
    with pytest.raises(TypeError):
        register(dict)
    with pytest.raises(ValueError, match="already registered"):

        class Clash(Strategy):
            name = "ema_cross"

            def generate_signals(self, df):
                return pd.Series(0.0, index=df.index)

        register(Clash)


# -- the contract every strategy must satisfy --------------------------------


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_signals_are_well_formed(name, bars):
    """Aligned to the input, bounded to -1/0/+1, and never NaN."""
    signals = get_strategy(name).generate_signals(bars)

    assert signals.index.equals(bars.index)
    assert set(np.unique(signals)) <= {-1.0, 0.0, 1.0}
    assert not signals.isna().any()
    assert signals.dtype == float


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_every_strategy_is_causal(name, bars):
    """Rewriting future bars must not change a single past signal.

    Parametrised over the registry on purpose: any strategy added later is
    held to this without anyone remembering to write the test.
    """
    strategy = get_strategy(name)
    original = strategy.generate_signals(bars)

    tampered = bars.copy()
    tampered.iloc[-200:, :] *= 4.0

    pd.testing.assert_series_equal(
        original.iloc[:-200], strategy.generate_signals(tampered).iloc[:-200]
    )


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_long_only_unless_shorting_is_enabled(name, bars):
    assert (get_strategy(name).generate_signals(bars) >= 0).all()


@pytest.mark.parametrize("name", [n for n in ALL_STRATEGIES if n != "buy_hold"])
def test_nothing_is_held_before_warmup(name, bars):
    strategy = get_strategy(name)
    signals = strategy.generate_signals(bars)
    assert (signals.iloc[: strategy.warmup - 1] == 0).all()


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_describe_names_the_strategy(name):
    assert get_strategy(name).describe().startswith(name)


# -- individual strategy behaviour -------------------------------------------


def frame_from(close: np.ndarray) -> pd.DataFrame:
    """Build bars from a raw price array (not a Series - that would align by index)."""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1_000.0),
        },
        index=pd.date_range("2022-01-01", periods=len(close), freq="B"),
    )


def test_rsi_reversion_buys_weakness_and_exits_on_recovery():
    # A slide into oversold, then a recovery back through the midline.
    frame = frame_from(np.concatenate([np.linspace(100, 70, 30), np.linspace(70, 105, 30)]))
    strategy = get_strategy("rsi_reversion", period=14, oversold=30, exit_level=50)

    signals = strategy.generate_signals(frame)
    values = strategy.indicators(frame)["rsi"]

    assert (signals == 1.0).any()
    # Every entry bar is genuinely oversold, and every exit bar has recovered.
    assert (values[signals.diff() > 0] < 30).all()
    assert (values[signals.diff() < 0] >= 50).all()
    assert signals.iloc[-1] == 0.0  # recovered, so the position is off


def test_donchian_enters_on_a_new_high():
    frame = frame_from(np.concatenate([np.full(25, 100.0), [120.0], np.full(10, 121.0)]))
    signals = get_strategy("donchian", entry_period=20, exit_period=10).generate_signals(frame)

    assert signals.iloc[24] == 0.0
    assert signals.iloc[25] == 1.0  # the breakout bar itself
    assert (signals.iloc[25:] == 1.0).all()


def test_donchian_rejects_an_exit_slower_than_its_entry():
    with pytest.raises(ValueError, match="should not exceed"):
        get_strategy("donchian", entry_period=10, exit_period=20)


def test_macd_follows_the_histogram_sign(bars):
    strategy = get_strategy("macd", fast=12, slow=26, signal=9)
    signals = strategy.generate_signals(bars)
    histogram = strategy.indicators(bars)["histogram"]
    warm = histogram.notna()

    assert (signals[warm & (histogram > 0)] == 1.0).all()
    assert (signals[warm & (histogram <= 0)] == 0.0).all()


def test_macd_min_histogram_reduces_activity(bars):
    plain = get_strategy("macd").generate_signals(bars)
    damped = get_strategy("macd", min_histogram=1.0).generate_signals(bars)
    assert damped.abs().sum() < plain.abs().sum()


def test_bollinger_buys_the_lower_band_and_exits_at_the_middle(bars):
    strategy = get_strategy("bollinger", period=20, num_std=2.0)
    signals = strategy.generate_signals(bars)
    ind = strategy.indicators(bars)

    entries = signals.diff() > 0
    assert (bars["close"][entries] < ind["lower"][entries]).all()
    assert (signals == 1.0).any()


@pytest.mark.parametrize(
    "name,params",
    [
        ("ema_cross", {"fast": 5, "slow": 20, "trend_filter": 0}),
        ("macd", {}),
        ("donchian", {}),
        ("rsi_reversion", {}),
        ("bollinger", {}),
    ],
)
def test_shorting_can_be_enabled_everywhere(name, params, bars):
    both = get_strategy(name, allow_short=True, **params).generate_signals(bars)
    assert (both < 0).any()


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
    """__getattr__ forwards to params, but must not swallow genuine typos."""
    with pytest.raises(AttributeError):
        get_strategy("ema_cross").nonexistent  # noqa: B018 - the access is the test


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
