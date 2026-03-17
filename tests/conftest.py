"""Fixtures compartilhadas para testes."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Gera DataFrame OHLCV sintetico para testes."""
    np.random.seed(42)
    n = 300
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")

    # Simular precos com random walk
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    close = np.maximum(close, 1000)  # Garantir positivo

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close + np.random.randn(n) * 100,
        "high": close + np.abs(np.random.randn(n) * 200),
        "low": close - np.abs(np.random.randn(n) * 200),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1e6) + 1e5,
    })
    # Garantir high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 1
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 1
    return df


@pytest.fixture
def sample_features(sample_ohlcv) -> tuple[np.ndarray, np.ndarray]:
    """Gera X, y sinteticos para treino."""
    np.random.seed(42)
    n = 200
    n_features = 50
    X = np.random.randn(n, n_features).astype(np.float32)
    y = np.random.randn(n).astype(np.float32) * 0.01
    return X, y
