"""Loading and normalising bars from messy inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.data import OHLCV_COLUMNS, DataError, load, load_csv, load_synthetic
from algobot.data.loader import normalize


def test_synthetic_bars_satisfy_the_ohlcv_contract():
    df = load_synthetic(periods=100, seed=1)

    assert list(df.columns) == OHLCV_COLUMNS
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


def test_synthetic_is_reproducible():
    a = load_synthetic(periods=50, seed=42)
    b = load_synthetic(periods=50, seed=42)
    c = load_synthetic(periods=50, seed=43)

    pd.testing.assert_frame_equal(a, b)
    assert not a["close"].equals(c["close"])


def test_normalize_accepts_yahoo_style_columns():
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-03"],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Adj Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        }
    )
    df = normalize(raw)

    assert list(df.columns) == OHLCV_COLUMNS
    assert df.index[0] == pd.Timestamp("2020-01-02")
    assert df["close"].iloc[0] == 101.0


def test_normalize_flattens_multiindex_columns():
    columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["SPY"]])
    raw = pd.DataFrame(
        [[100.0, 102.0, 99.0, 101.0, 1000.0]],
        index=pd.DatetimeIndex(["2020-01-02"]),
        columns=columns,
    )
    df = normalize(raw)
    assert list(df.columns) == OHLCV_COLUMNS


def test_normalize_sorts_and_deduplicates():
    raw = pd.DataFrame(
        {
            "date": ["2020-01-03", "2020-01-02", "2020-01-03"],
            "open": [101.0, 100.0, 111.0],
            "high": [103.0, 102.0, 113.0],
            "low": [100.0, 99.0, 110.0],
            "close": [102.0, 101.0, 112.0],
        }
    )
    df = normalize(raw)

    assert len(df) == 2
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 112.0  # duplicate resolved to the last row


def test_normalize_drops_unusable_rows_and_defaults_volume():
    raw = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "open": [100.0, np.nan, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
        }
    )
    df = normalize(raw)

    assert len(df) == 2
    assert (df["volume"] == 0.0).all()


def test_normalize_repairs_bars_whose_range_excludes_the_body():
    raw = pd.DataFrame(
        {
            "date": ["2020-01-02"],
            "open": [100.0],
            "high": [99.0],  # below the open, which is impossible
            "low": [101.0],
            "close": [100.5],
        }
    )
    df = normalize(raw)

    assert df["high"].iloc[0] == 100.5
    assert df["low"].iloc[0] == 100.0


def test_normalize_rejects_empty_and_incomplete_input():
    with pytest.raises(DataError, match="no rows"):
        normalize(pd.DataFrame())

    with pytest.raises(DataError, match="missing required column"):
        normalize(pd.DataFrame({"date": ["2020-01-02"], "close": [100.0]}))

    with pytest.raises(DataError, match="no usable bars"):
        normalize(
            pd.DataFrame(
                {
                    "date": ["2020-01-02"],
                    "open": [-1.0],
                    "high": [1.0],
                    "low": [-2.0],
                    "close": [0.0],
                }
            )
        )


def test_csv_round_trip(tmp_path):
    original = load_synthetic(periods=60, seed=5)
    path = tmp_path / "bars.csv"
    original.to_csv(path)

    reloaded = load_csv(path)
    pd.testing.assert_frame_equal(original, reloaded, check_freq=False)


def test_bundled_sample_file_loads():
    df = load_csv("examples/sample_bars.csv")
    assert len(df) > 500
    assert list(df.columns) == OHLCV_COLUMNS


def test_load_dispatches_and_validates_source(tmp_path):
    df = load(source="synthetic", symbol="X", periods=30)
    assert len(df) == 30

    with pytest.raises(DataError, match="requires a path"):
        load(source="csv")

    with pytest.raises(DataError, match="unknown data source"):
        load(source="bloomberg")

    with pytest.raises(DataError, match="not found"):
        load(source="csv", path=tmp_path / "missing.csv")
