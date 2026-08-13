"""Indicator correctness, including the causality property the engine relies on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.indicators import atr, ema, rsi, sma, true_range


def test_sma_matches_hand_computation():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(s, 3)
    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_uses_conventional_smoothing():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = ema(s, 2)
    # alpha = 2/(2+1); seeded on the first valid window, then recursive.
    alpha = 2 / 3
    expected = 1.0
    for value in s.iloc[1:]:
        expected = alpha * value + (1 - alpha) * expected
    assert result.iloc[-1] == pytest.approx(expected)


def test_ema_respects_warmup():
    s = pd.Series(np.arange(10.0))
    result = ema(s, 5)
    assert result.iloc[:4].isna().all()
    assert result.iloc[4:].notna().all()


def test_true_range_takes_the_widest_of_the_three_ranges():
    df = pd.DataFrame({"high": [10.0, 12.0], "low": [9.0, 11.0], "close": [9.5, 11.5]})
    tr = true_range(df)
    assert tr.iloc[0] == pytest.approx(1.0)  # no previous close, so high - low
    assert tr.iloc[1] == pytest.approx(2.5)  # high - previous close dominates


def test_atr_is_positive_and_warms_up(bars):
    result = atr(bars, 14)
    assert result.iloc[:13].isna().all()
    assert (result.dropna() > 0).all()


def test_rsi_bounds_and_extremes():
    rising = pd.Series(np.arange(1.0, 40.0))
    assert rsi(rising, 14).dropna().iloc[-1] == pytest.approx(100.0)

    falling = pd.Series(np.arange(40.0, 1.0, -1.0))
    assert rsi(falling, 14).dropna().iloc[-1] == pytest.approx(0.0)

    mixed = pd.Series(np.random.default_rng(0).normal(100, 5, 200).cumsum())
    values = rsi(mixed, 14).dropna()
    assert ((values >= 0) & (values <= 100)).all()


@pytest.mark.parametrize("fn", [lambda s: ema(s, 10), lambda s: sma(s, 10)])
def test_indicators_are_causal(fn, bars):
    """Changing the tail of a series must not change any earlier value.

    This is the property that keeps the backtester honest; if it ever breaks,
    every result the framework produces is suspect.
    """
    close = bars["close"]
    original = fn(close)

    tampered = close.copy()
    tampered.iloc[-50:] *= 3.0
    after = fn(tampered)

    pd.testing.assert_series_equal(original.iloc[:-50], after.iloc[:-50])


def test_atr_is_causal(bars):
    original = atr(bars, 14)
    tampered = bars.copy()
    tampered.iloc[-50:, :] *= 3.0
    pd.testing.assert_series_equal(original.iloc[:-50], atr(tampered, 14).iloc[:-50])


def test_invalid_periods_rejected():
    s = pd.Series([1.0, 2.0])
    for fn in (ema, sma, rsi):
        with pytest.raises(ValueError):
            fn(s, 0)
