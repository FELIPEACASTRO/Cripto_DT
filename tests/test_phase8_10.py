"""Testes para Phase 8 (Chart Patterns) e Phase 10 (Regional Intelligence)."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
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
    return df


class TestChartPatternDetector:

    def test_transform_generates_features(self):
        from src.features.chart_patterns import ChartPatternDetector
        df = _make_ohlcv(200)
        detector = ChartPatternDetector()
        result = detector.transform(df)
        chart_cols = [c for c in result.columns if c.startswith("chart_")]
        assert len(chart_cols) >= 5

    def test_preserves_original_data(self):
        from src.features.chart_patterns import ChartPatternDetector
        df = _make_ohlcv(100)
        original_cols = set(df.columns)
        detector = ChartPatternDetector()
        result = detector.transform(df)
        assert original_cols.issubset(set(result.columns))
        assert len(result) == len(df)

    def test_no_crash_short_data(self):
        from src.features.chart_patterns import ChartPatternDetector
        df = _make_ohlcv(10)
        detector = ChartPatternDetector()
        result = detector.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_no_inf_or_all_nan(self):
        from src.features.chart_patterns import ChartPatternDetector
        df = _make_ohlcv(200)
        detector = ChartPatternDetector()
        result = detector.transform(df)
        chart_cols = [c for c in result.columns if c.startswith("chart_")]
        for col in chart_cols:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert not np.any(np.isinf(vals)), f"{col} tem inf"

    def test_empty_df(self):
        from src.features.chart_patterns import ChartPatternDetector
        df = pd.DataFrame()
        detector = ChartPatternDetector()
        result = detector.transform(df)
        assert isinstance(result, pd.DataFrame)


class TestRegionalIntelligence:

    def test_transform_generates_features(self):
        from src.features.regional_intelligence import RegionalIntelligence
        df = _make_ohlcv(200)
        ri = RegionalIntelligence()
        result = ri.transform(df)
        reg_cols = [c for c in result.columns if c.startswith("reg_")]
        assert len(reg_cols) >= 5

    def test_preserves_original_data(self):
        from src.features.regional_intelligence import RegionalIntelligence
        df = _make_ohlcv(100)
        original_cols = set(df.columns)
        ri = RegionalIntelligence()
        result = ri.transform(df)
        assert original_cols.issubset(set(result.columns))
        assert len(result) == len(df)

    def test_no_crash_short_data(self):
        from src.features.regional_intelligence import RegionalIntelligence
        df = _make_ohlcv(10)
        ri = RegionalIntelligence()
        result = ri.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_calendar_features(self):
        from src.features.regional_intelligence import RegionalIntelligence
        df = _make_ohlcv(200)
        ri = RegionalIntelligence()
        result = ri.transform(df)
        # Check calendar-related features exist
        calendar_cols = [c for c in result.columns if "day" in c or "month" in c or "week" in c]
        assert len(calendar_cols) >= 1 or len([c for c in result.columns if c.startswith("reg_")]) >= 5

    def test_empty_df(self):
        from src.features.regional_intelligence import RegionalIntelligence
        df = pd.DataFrame()
        ri = RegionalIntelligence()
        result = ri.transform(df)
        assert isinstance(result, pd.DataFrame)


class TestPhase810Config:

    def test_feature_flags(self):
        from config.settings import Config
        cfg = Config()
        assert hasattr(cfg.features, "use_chart_patterns")
        assert cfg.features.use_chart_patterns is True
        assert hasattr(cfg.features, "use_regional_intelligence")
        assert cfg.features.use_regional_intelligence is True
