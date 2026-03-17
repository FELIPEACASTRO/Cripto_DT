"""Testes para features de sentimento."""

import numpy as np
import pandas as pd
import pytest

from src.data.sentiment import SentimentCollector


class TestSentimentCollector:

    def _make_fg_df(self) -> pd.DataFrame:
        """Cria DataFrame de Fear & Greed sintetico."""
        dates = pd.date_range("2023-01-01", periods=90, freq="1D", tz="UTC")
        np.random.seed(42)
        values = np.random.randint(10, 90, size=90)
        return pd.DataFrame({
            "timestamp": dates,
            "fg_value": values,
            "fg_classification": ["Neutral"] * 90,
        })

    def test_add_sentiment_features(self, sample_ohlcv):
        collector = SentimentCollector()
        fg_df = self._make_fg_df()
        result = collector.add_sentiment_features(sample_ohlcv, fg_df)

        assert "fg_value" in result.columns
        assert "fg_normalized" in result.columns
        assert "fg_change_1d" in result.columns
        assert "fg_ma_7" in result.columns
        assert "fg_extreme_fear" in result.columns
        assert "fg_extreme_greed" in result.columns

    def test_fg_normalized_range(self, sample_ohlcv):
        collector = SentimentCollector()
        fg_df = self._make_fg_df()
        result = collector.add_sentiment_features(sample_ohlcv, fg_df)
        valid = result["fg_normalized"].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0
            assert valid.max() <= 1.0

    def test_empty_fg_returns_unchanged(self, sample_ohlcv):
        collector = SentimentCollector()
        empty = pd.DataFrame()
        result = collector.add_sentiment_features(sample_ohlcv, empty)
        assert len(result) == len(sample_ohlcv)
        assert "fg_value" not in result.columns

    def test_feature_names(self):
        names = SentimentCollector.get_feature_names()
        assert "fg_value" in names
        assert "fg_extreme_fear" in names
        assert len(names) == 9
