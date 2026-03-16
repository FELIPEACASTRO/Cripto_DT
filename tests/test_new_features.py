"""Testes para novas features: Volatility, Bubble, Anomaly, HAR, Whale."""

import numpy as np
import pandas as pd
import pytest


class TestVolatilityFeatures:

    def test_transform(self, sample_ohlcv):
        from src.features.volatility import VolatilityFeatures
        from src.data.preprocessor import DataPreprocessor
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        df = prep.add_returns(df)

        vf = VolatilityFeatures()
        result = vf.transform(df)
        vol_cols = [c for c in result.columns if c.startswith(("realized_vol", "parkinson", "garman", "vol_"))]
        assert len(vol_cols) >= 4


class TestBubbleDetector:

    def test_transform(self, sample_ohlcv):
        from src.features.bubble import BubbleDetector
        from src.data.preprocessor import DataPreprocessor
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        df = prep.add_returns(df)

        bd = BubbleDetector()
        result = bd.transform(df)
        bubble_cols = [c for c in result.columns if c.startswith("bubble_")]
        assert len(bubble_cols) >= 3


class TestAnomalyDetector:

    def test_transform(self, sample_ohlcv):
        from src.features.anomaly import AnomalyDetector
        from src.data.preprocessor import DataPreprocessor
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        df = prep.add_returns(df)

        ad = AnomalyDetector()
        result = ad.transform(df)
        anomaly_cols = [c for c in result.columns if c.startswith("anomaly_")]
        assert len(anomaly_cols) >= 4


class TestHARVolatility:

    def test_transform(self, sample_ohlcv):
        from src.features.har_volatility import HARVolatility
        from src.data.preprocessor import DataPreprocessor
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        df = prep.add_returns(df)

        har = HARVolatility()
        result = har.transform(df)
        har_cols = [c for c in result.columns if c.startswith("har_")]
        assert len(har_cols) >= 4


class TestWhaleMonitor:

    def test_add_whale_features(self, sample_ohlcv):
        from src.data.whale_monitor import WhaleMonitor
        from src.data.preprocessor import DataPreprocessor
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)

        wm = WhaleMonitor()
        result = wm.add_whale_features(df, "BTC")
        whale_cols = [c for c in result.columns if c.startswith("whale_")]
        assert len(whale_cols) >= 3


class TestFeatureSelection:

    def test_mutual_info_select(self, sample_features):
        from src.features.feature_selection import FeatureSelector
        X, y = sample_features
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        fs = FeatureSelector()
        selected = fs.mutual_info_select(X, y, feature_names, k=10)
        assert len(selected) == 10
        assert all(f in feature_names for f in selected)

    def test_l1_select(self, sample_features):
        from src.features.feature_selection import FeatureSelector
        X, y = sample_features
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        fs = FeatureSelector()
        selected = fs.l1_select(X, y, feature_names)
        assert isinstance(selected, list)
