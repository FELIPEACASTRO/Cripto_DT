"""Testes para modelos avancados: Helformer, MDN, Conformal, EMGNN."""

import numpy as np
import pytest
import torch


class TestConformalPredictor:

    def test_calibrate_and_predict(self):
        from src.models.conformal import ConformalPredictor

        np.random.seed(42)
        n = 200
        y_true = np.random.randn(n) * 0.01
        y_pred = y_true + np.random.randn(n) * 0.003

        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(y_true, y_pred)

        lower, upper = cp.predict_intervals(y_pred[:50])
        assert lower.shape == (50,)
        assert upper.shape == (50,)
        assert (lower < upper).all()

    def test_coverage(self):
        from src.models.conformal import ConformalPredictor

        np.random.seed(42)
        n = 500
        y_true = np.random.randn(n) * 0.01
        noise = np.random.randn(n) * 0.003
        y_pred = y_true + noise

        # Calibrar com metade
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(y_true[:250], y_pred[:250])

        # Testar com outra metade
        lower, upper = cp.predict_intervals(y_pred[250:])
        y_test = y_true[250:]
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        # Coverage deve ser ~90% (alpha=0.1) com margem
        assert coverage >= 0.80

    def test_save_load(self, tmp_path):
        from src.models.conformal import ConformalPredictor

        np.random.seed(42)
        y_true = np.random.randn(100) * 0.01
        y_pred = y_true + np.random.randn(100) * 0.003

        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(y_true, y_pred)

        path = tmp_path / "conformal.joblib"
        cp.save(path)
        loaded = ConformalPredictor.load(path)
        assert loaded.alpha == 0.1
        assert loaded._q_hat is not None


class TestMDNModel:

    def test_fit_and_predict(self):
        from src.models.mdn import MDNModel

        np.random.seed(42)
        X = np.random.randn(200, 20).astype(np.float32)
        y = np.random.randn(200).astype(np.float32) * 0.01

        model = MDNModel(input_size=20)
        model.fit(X[:150], y[:150], X[150:], y[150:])

        preds = model.predict(X[150:])
        assert preds.shape == (50,)

    def test_predict_with_confidence(self):
        from src.models.mdn import MDNModel

        np.random.seed(42)
        X = np.random.randn(200, 20).astype(np.float32)
        y = np.random.randn(200).astype(np.float32) * 0.01

        model = MDNModel(input_size=20)
        model.fit(X[:150], y[:150])

        preds, conf = model.predict_with_confidence(X[150:])
        assert preds.shape == conf.shape
        assert (conf >= 0).all() and (conf <= 1).all()


class TestHelformerModel:

    def test_fit_and_predict(self):
        from src.models.helformer import HelformerModel
        from config.settings import FeatureConfig

        np.random.seed(42)
        # Use enough samples so that lookback_window=20 leaves sequences
        X = np.random.randn(200, 10).astype(np.float32)
        y = np.random.randn(200).astype(np.float32) * 0.01

        feat_cfg = FeatureConfig(lookback_window=20)
        model = HelformerModel(seq_len=20, feature_config=feat_cfg, max_epochs=5)
        model.fit(X[:150], y[:150], X[150:], y[150:])

        preds = model.predict(X[150:])
        assert len(preds) > 0

    def test_predict_with_confidence(self):
        from src.models.helformer import HelformerModel
        from config.settings import FeatureConfig

        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = np.random.randn(200).astype(np.float32) * 0.01

        feat_cfg = FeatureConfig(lookback_window=20)
        model = HelformerModel(seq_len=20, feature_config=feat_cfg, max_epochs=5, mc_dropout_samples=5)
        model.fit(X[:150], y[:150])

        preds, conf = model.predict_with_confidence(X[150:])
        assert len(preds) == len(conf)


class TestDecomposition:

    def test_transform(self, sample_ohlcv):
        try:
            from src.features.decomposition import SignalDecomposer
        except ImportError:
            pytest.skip("PyEMD not installed")

        decomposer = SignalDecomposer(n_imfs=4)
        result = decomposer.transform(sample_ohlcv)
        # Deve ter features de IMF
        imf_cols = [c for c in result.columns if c.startswith("imf_")]
        assert len(imf_cols) > 0


class TestRegimeDetector:

    def test_transform(self, sample_ohlcv):
        try:
            from src.features.regime import RegimeDetector
        except ImportError:
            pytest.skip("hmmlearn not installed")

        from src.data.preprocessor import DataPreprocessor
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        df = prep.add_returns(df)

        detector = RegimeDetector(n_regimes=3)
        detector.fit(df["log_return"].dropna().values)
        result = detector.transform(df)
        assert "regime_state" in result.columns
