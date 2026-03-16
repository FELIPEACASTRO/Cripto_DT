"""Testes para features da pesquisa Oriental: Regional Markets, Chart Vision, Blockchain NLP."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Cria DataFrame OHLCV sintetico para testes de features."""
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    close = np.maximum(close, 1000)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close + np.random.randn(n) * 50,
        "high": close + np.abs(np.random.randn(n) * 100),
        "low": close - np.abs(np.random.randn(n) * 100),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1e6) + 1e5,
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 1
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 1
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_return"] = df["close"].pct_change()
    df = df.dropna().reset_index(drop=True)
    return df


class TestRegionalMarketFeatures:

    def test_transform_adds_all_features(self):
        from src.features.regional_markets import RegionalMarketFeatures
        df = _make_ohlcv()
        rm = RegionalMarketFeatures()
        result = rm.transform(df)

        expected = rm.get_feature_names()
        for feat in expected:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_session_features_are_binary(self):
        from src.features.regional_markets import RegionalMarketFeatures
        df = _make_ohlcv()
        rm = RegionalMarketFeatures()
        result = rm.transform(df)

        session_cols = [c for c in result.columns if c.startswith("session_") or c == "is_weekend"]
        for col in session_cols:
            unique = set(result[col].unique())
            assert unique.issubset({0.0, 1.0}), f"{col} has non-binary values: {unique}"

    def test_no_nans_in_output(self):
        from src.features.regional_markets import RegionalMarketFeatures
        df = _make_ohlcv()
        rm = RegionalMarketFeatures()
        result = rm.transform(df)
        feature_cols = rm.get_feature_names()
        existing = [c for c in feature_cols if c in result.columns]
        # Allow some NaNs from rolling warmup
        nan_ratio = result[existing].iloc[50:].isna().mean()
        assert (nan_ratio < 0.1).all(), f"Too many NaNs: {nan_ratio[nan_ratio >= 0.1]}"

    def test_feature_names_count(self):
        from src.features.regional_markets import RegionalMarketFeatures
        rm = RegionalMarketFeatures()
        assert len(rm.get_feature_names()) == 19


class TestChartVisionFeatures:

    def test_transform_adds_features(self):
        from src.features.chart_vision import ChartVisionFeatures
        df = _make_ohlcv()
        cv = ChartVisionFeatures()
        result = cv.transform(df)

        expected = cv.get_feature_names()
        for feat in expected:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_pattern_score_range(self):
        from src.features.chart_vision import ChartVisionFeatures
        df = _make_ohlcv()
        cv = ChartVisionFeatures()
        result = cv.transform(df)

        if "chart_pattern_score" in result.columns:
            scores = result["chart_pattern_score"].dropna()
            assert scores.min() >= -2.0, "Pattern score too low"
            assert scores.max() <= 2.0, "Pattern score too high"

    def test_consecutive_candles_non_negative(self):
        from src.features.chart_vision import ChartVisionFeatures
        df = _make_ohlcv()
        cv = ChartVisionFeatures()
        result = cv.transform(df)

        for col in ["chart_consecutive_green", "chart_consecutive_red"]:
            if col in result.columns:
                assert (result[col].dropna() >= 0).all(), f"{col} has negative values"


class TestBlockchainNLPFeatures:

    def test_transform_adds_features(self):
        from src.features.blockchain_nlp import BlockchainNLPFeatures
        df = _make_ohlcv()
        bnlp = BlockchainNLPFeatures()
        result = bnlp.transform(df)

        expected = bnlp.get_feature_names()
        for feat in expected:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_no_crash_on_minimal_data(self):
        from src.features.blockchain_nlp import BlockchainNLPFeatures
        df = _make_ohlcv(n=30)
        bnlp = BlockchainNLPFeatures()
        result = bnlp.transform(df)
        assert len(result) > 0  # Should not crash or return empty

    def test_features_are_float32(self):
        from src.features.blockchain_nlp import BlockchainNLPFeatures
        df = _make_ohlcv()
        bnlp = BlockchainNLPFeatures()
        result = bnlp.transform(df)
        feature_names = bnlp.get_feature_names()
        for feat in feature_names:
            if feat in result.columns:
                assert result[feat].dtype == np.float32, f"{feat} is {result[feat].dtype}"
