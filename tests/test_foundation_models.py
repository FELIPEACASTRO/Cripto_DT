"""Testes para Foundation Models (Phase 5) e NLP Sentiment (Phase 6)."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Cria DataFrame OHLCV sintetico."""
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


class TestChronosModel:

    def test_init_and_name(self):
        from src.models.chronos_model import ChronosModel
        model = ChronosModel()
        assert "chronos" in model.name

    def test_fit_predict(self):
        from src.models.chronos_model import ChronosModel
        np.random.seed(42)
        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32) * 0.01
        X_val = np.random.randn(20, 10).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32) * 0.01
        X_test = np.random.randn(15, 10).astype(np.float32)

        model = ChronosModel()
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, dict)

        preds = model.predict(X_test)
        assert preds.shape == (15,)

    def test_predict_with_confidence(self):
        from src.models.chronos_model import ChronosModel
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32) * 0.01

        model = ChronosModel()
        model.fit(X[:40], y[:40])
        preds, conf = model.predict_with_confidence(X[40:])
        assert preds.shape == (10,)
        assert conf.shape == (10,)
        assert np.all(conf >= 0) and np.all(conf <= 1)

    def test_save_load(self, tmp_path):
        from src.models.chronos_model import ChronosModel
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32) * 0.01

        model = ChronosModel()
        model.fit(X[:40], y[:40])
        path = tmp_path / "chronos.pt"
        model.save(path)
        loaded = ChronosModel.load(path)
        preds1 = model.predict(X[40:])
        preds2 = loaded.predict(X[40:])
        np.testing.assert_allclose(preds1, preds2, atol=1e-5)


class TestTTMModel:

    def test_init_and_name(self):
        from src.models.ttm_model import TTMModel
        model = TTMModel()
        assert model.name == "ttm"

    def test_fit_predict(self):
        from src.models.ttm_model import TTMModel
        np.random.seed(42)
        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32) * 0.01
        X_val = np.random.randn(20, 10).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32) * 0.01
        X_test = np.random.randn(15, 10).astype(np.float32)

        model = TTMModel()
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, dict)

        preds = model.predict(X_test)
        assert preds.shape == (15,)

    def test_predict_with_confidence(self):
        from src.models.ttm_model import TTMModel
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32) * 0.01

        model = TTMModel()
        model.fit(X[:40], y[:40])
        preds, conf = model.predict_with_confidence(X[40:])
        assert preds.shape == (10,)
        assert conf.shape == (10,)

    def test_save_load(self, tmp_path):
        from src.models.ttm_model import TTMModel
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32) * 0.01

        model = TTMModel()
        model.fit(X[:40], y[:40])
        path = tmp_path / "ttm.joblib"
        model.save(path)
        loaded = TTMModel.load(path)
        preds1 = model.predict(X[40:])
        preds2 = loaded.predict(X[40:])
        np.testing.assert_allclose(preds1, preds2, atol=1e-5)


class TestMoiraiModel:

    def test_init_and_name(self):
        from src.models.moirai_model import MoiraiModel
        model = MoiraiModel()
        assert model.name == "moirai"

    def test_fit_predict(self):
        from src.models.moirai_model import MoiraiModel
        np.random.seed(42)
        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32) * 0.01
        X_val = np.random.randn(20, 10).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32) * 0.01
        X_test = np.random.randn(15, 10).astype(np.float32)

        model = MoiraiModel()
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert isinstance(metrics, dict)

        preds = model.predict(X_test)
        assert preds.shape == (15,)

    def test_predict_with_confidence(self):
        from src.models.moirai_model import MoiraiModel
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32) * 0.01

        model = MoiraiModel()
        model.fit(X[:40], y[:40])
        preds, conf = model.predict_with_confidence(X[40:])
        assert preds.shape == (10,)
        assert conf.shape == (10,)

    def test_save_load(self, tmp_path):
        from src.models.moirai_model import MoiraiModel
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32) * 0.01

        model = MoiraiModel()
        model.fit(X[:40], y[:40])
        path = tmp_path / "moirai.joblib"
        model.save(path)
        loaded = MoiraiModel.load(path)
        preds1 = model.predict(X[40:])
        preds2 = loaded.predict(X[40:])
        np.testing.assert_allclose(preds1, preds2, atol=1e-5)


class TestNLPSentimentFeatures:

    def test_transform_generates_features(self):
        from src.features.finbert_sentiment import NLPSentimentFeatures
        df = _make_ohlcv(100)
        nlp = NLPSentimentFeatures()
        result = nlp.transform(df)
        # Deve gerar features proxy a partir de OHLCV
        nlp_cols = [c for c in result.columns if c.startswith("nlp_")]
        assert len(nlp_cols) > 0

    def test_no_crash_empty_df(self):
        from src.features.finbert_sentiment import NLPSentimentFeatures
        df = pd.DataFrame()
        nlp = NLPSentimentFeatures()
        result = nlp.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self):
        from src.features.finbert_sentiment import NLPSentimentFeatures
        df = _make_ohlcv(50)
        original_cols = set(df.columns)
        nlp = NLPSentimentFeatures()
        result = nlp.transform(df)
        assert original_cols.issubset(set(result.columns))

    def test_proxy_features_range(self):
        from src.features.finbert_sentiment import NLPSentimentFeatures
        df = _make_ohlcv(100)
        nlp = NLPSentimentFeatures()
        result = nlp.transform(df)
        nlp_cols = [c for c in result.columns if c.startswith("nlp_")]
        # Proxy features should not have inf or all-NaN
        for col in nlp_cols:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert not np.any(np.isinf(vals)), f"{col} tem inf"


class TestConfigIntegration:

    def test_foundation_model_configs_exist(self):
        from config.settings import Config
        cfg = Config()
        assert hasattr(cfg, "chronos")
        assert hasattr(cfg, "ttm")
        assert hasattr(cfg, "moirai")
        assert hasattr(cfg, "nlp_sentiment")

    def test_training_flags_exist(self):
        from config.settings import Config
        cfg = Config()
        assert hasattr(cfg.training, "use_chronos")
        assert hasattr(cfg.training, "use_ttm")
        assert hasattr(cfg.training, "use_moirai")
        assert hasattr(cfg.training, "use_nlp_finbert")

    def test_feature_flag_exists(self):
        from config.settings import Config
        cfg = Config()
        assert hasattr(cfg.features, "use_finbert_sentiment")
