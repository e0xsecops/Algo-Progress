"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.data import load_synthetic


@pytest.fixture
def bars() -> pd.DataFrame:
    """A reproducible synthetic price series long enough to warm up indicators."""
    return load_synthetic(periods=800, seed=11)


@pytest.fixture
def trending_bars() -> pd.DataFrame:
    """A clean uptrend then downtrend, so signals are predictable by hand."""
    up = np.linspace(100.0, 200.0, 150)
    down = np.linspace(200.0, 120.0, 150)
    close = np.concatenate([up, down])
    index = pd.date_range("2021-01-01", periods=len(close), freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )
