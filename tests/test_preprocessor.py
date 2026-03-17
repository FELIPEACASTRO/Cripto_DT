"""Testes para preprocessamento de dados."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor, FeatureScaler


class TestDataPreprocessor:

    def test_clean_removes_duplicates(self, sample_ohlcv):
        df = pd.concat([sample_ohlcv, sample_ohlcv.iloc[:5]])
        prep = DataPreprocessor()
        result = prep.clean(df)
        assert len(result) == len(sample_ohlcv)

    def test_clean_removes_negative_prices(self, sample_ohlcv):
        df = sample_ohlcv.copy()
        df.loc[0, "close"] = -100
        prep = DataPreprocessor()
        result = prep.clean(df)
        assert (result["close"] > 0).all()

    def test_add_returns_columns(self, sample_ohlcv):
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        df = prep.add_returns(df)
        assert "log_return" in df.columns
        assert "pct_return" in df.columns
        assert "direction" in df.columns
        assert df["direction"].isin([0, 1]).all()

    def test_add_target_log_return(self, sample_ohlcv):
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        df = prep.add_returns(df)
        df = prep.add_target(df, horizon=1, target_type="log_return")
        assert "target" in df.columns
        # Ultimo valor deve ser NaN (sem futuro)
        assert pd.isna(df["target"].iloc[-1])

    def test_prepare_full_pipeline(self, sample_ohlcv):
        prep = DataPreprocessor()
        df = prep.prepare(sample_ohlcv)
        assert "log_return" in df.columns
        assert "target" in df.columns
        assert len(df) > 0


class TestFeatureScaler:

    def test_fit_transform(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "feat_a": np.random.randn(100),
            "feat_b": np.random.randn(100) * 10 + 5,
        })
        scaler = FeatureScaler()
        result = scaler.fit_transform(df, ["feat_a", "feat_b"])
        # Apos escalar, media deve ser ~0 e std ~1
        assert abs(result["feat_a"].mean()) < 0.2
        assert abs(result["feat_b"].mean()) < 0.2

    def test_transform_without_fit_raises(self):
        scaler = FeatureScaler()
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError):
            scaler.transform(df)

    def test_no_data_leakage(self):
        """Scaler fitado no treino nao deve usar info do teste."""
        np.random.seed(42)
        train = pd.DataFrame({"x": np.random.randn(100)})
        test = pd.DataFrame({"x": np.random.randn(50) + 100})

        scaler = FeatureScaler()
        scaler.fit(train, ["x"])
        test_scaled = scaler.transform(test)

        # Dados de teste devem ter valores altos apos escalar (media do treino ~0)
        assert test_scaled["x"].mean() > 50
