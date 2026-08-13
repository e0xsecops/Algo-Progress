"""Indicator correctness, including the causality property the engine relies on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.indicators import (
    atr,
    bollinger,
    donchian,
    ema,
    hold_between,
    macd,
    rolling_std,
    rsi,
    sma,
    true_range,
)


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
    with pytest.raises(ValueError):
        rolling_std(s, 1)


def test_rolling_std_matches_pandas():
    s = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])
    assert rolling_std(s, 3).iloc[2] == pytest.approx(2.0)


def test_macd_histogram_is_the_gap_between_the_lines(bars):
    frame = macd(bars["close"], 12, 26, 9)
    valid = frame.dropna()

    assert list(frame.columns) == ["macd", "signal", "histogram"]
    assert (valid["histogram"] - (valid["macd"] - valid["signal"])).abs().max() < 1e-12
    assert frame["macd"].iloc[:25].isna().all()


def test_macd_rejects_inverted_periods(bars):
    with pytest.raises(ValueError, match="must be shorter"):
        macd(bars["close"], 26, 12)


def test_bollinger_bands_straddle_the_middle(bars):
    frame = bollinger(bars["close"], 20, 2.0).dropna()

    assert (frame["upper"] > frame["middle"]).all()
    assert (frame["lower"] < frame["middle"]).all()
    # The bands sit a symmetric distance either side of the mean.
    spread = frame["upper"] - frame["middle"]
    assert ((frame["middle"] - frame["lower"]) - spread).abs().max() < 1e-12


def test_donchian_excludes_the_current_bar():
    df = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 20.0],
            "low": [9.0, 8.0, 7.0, 6.0],
            "close": [9.5, 10.5, 11.5, 19.0],
        }
    )
    channel = donchian(df, 2)

    # At bar 3 the channel reflects bars 1-2 only, so the 20.0 spike does not
    # define the level it is supposed to be breaking out of.
    assert channel["upper"].iloc[3] == pytest.approx(12.0)
    assert channel["lower"].iloc[3] == pytest.approx(7.0)
    assert channel["upper"].iloc[:2].isna().all()


def test_hold_between_holds_the_position_until_the_exit_fires():
    entry = pd.Series([False, True, False, False, False, True, False])
    exit_ = pd.Series([False, False, False, True, False, False, False])

    held = hold_between(entry, exit_, 1.0)

    assert list(held) == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]


def test_hold_between_prefers_entry_when_both_fire():
    entry = pd.Series([False, True, False])
    exit_ = pd.Series([False, True, False])
    assert list(hold_between(entry, exit_, 1.0)) == [0.0, 1.0, 1.0]


def test_hold_between_is_causal():
    entry = pd.Series([False, True, False, False, False])
    exit_ = pd.Series([False, False, False, False, True])
    original = hold_between(entry, exit_, 1.0)

    tampered_exit = exit_.copy()
    tampered_exit.iloc[4] = False
    after = hold_between(entry, tampered_exit, 1.0)

    pd.testing.assert_series_equal(original.iloc[:4], after.iloc[:4])
