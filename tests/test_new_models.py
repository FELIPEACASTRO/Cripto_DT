"""Testes para novos modelos: LightGBM, Random Forest, SVM, CNN-LSTM, TCN, BNN."""

import numpy as np
import pytest
import torch


class TestLightGBMRegressor:

    def test_fit_and_predict(self, sample_features):
        from src.models.lightgbm_model import LightGBMRegressor
        X, y = sample_features
        model = LightGBMRegressor()
        model.fit(X[:150], y[:150], X[150:], y[150:])
        preds = model.predict(X[150:])
        assert preds.shape == (50,)

    def test_predict_with_confidence(self, sample_features):
        from src.models.lightgbm_model import LightGBMRegressor
        X, y = sample_features
        model = LightGBMRegressor()
        model.fit(X[:150], y[:150])
        preds, conf = model.predict_with_confidence(X[150:])
        assert preds.shape == conf.shape
        assert (conf >= 0).all() and (conf <= 1).all()


class TestRandomForestRegressor:

    def test_fit_and_predict(self, sample_features):
        from src.models.random_forest import RandomForestRegressorModel
        X, y = sample_features
        model = RandomForestRegressorModel()
        model.fit(X[:150], y[:150], X[150:], y[150:])
        preds = model.predict(X[150:])
        assert preds.shape == (50,)

    def test_predict_with_confidence(self, sample_features):
        from src.models.random_forest import RandomForestRegressorModel
        X, y = sample_features
        model = RandomForestRegressorModel()
        model.fit(X[:150], y[:150])
        preds, conf = model.predict_with_confidence(X[150:])
        assert preds.shape == conf.shape
        assert (conf >= 0).all() and (conf <= 1).all()


class TestSVMRegressor:

    def test_fit_and_predict(self, sample_features):
        from src.models.svm_model import SVMRegressor
        X, y = sample_features
        model = SVMRegressor()
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        assert preds.shape == (50,)

    def test_predict_with_confidence(self, sample_features):
        from src.models.svm_model import SVMRegressor
        X, y = sample_features
        model = SVMRegressor()
        model.fit(X[:150], y[:150])
        preds, conf = model.predict_with_confidence(X[150:])
        assert preds.shape == conf.shape


class TestCNNLSTMModel:

    def test_fit_and_predict(self):
        from config.settings import FeatureConfig
        from src.models.cnn_lstm import CNNLSTMModel
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = np.random.randn(200).astype(np.float32) * 0.01
        feat_cfg = FeatureConfig(lookback_window=20)
        model = CNNLSTMModel(max_epochs=3, feature_config=feat_cfg)
        model.fit(X[:150], y[:150], X[150:], y[150:])
        preds = model.predict(X[150:])
        assert len(preds) > 0


class TestTCNModel:

    def test_fit_and_predict(self):
        from config.settings import FeatureConfig
        from src.models.tcn import TCNModel
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = np.random.randn(200).astype(np.float32) * 0.01
        feat_cfg = FeatureConfig(lookback_window=20)
        model = TCNModel(max_epochs=3, feature_config=feat_cfg)
        model.fit(X[:150], y[:150], X[150:], y[150:])
        preds = model.predict(X[150:])
        assert len(preds) > 0


class TestBNNModel:

    def test_fit_and_predict(self):
        from src.models.bnn import BNNModel
        np.random.seed(42)
        X = np.random.randn(200, 20).astype(np.float32)
        y = np.random.randn(200).astype(np.float32) * 0.01
        model = BNNModel(input_size=20, max_epochs=5)
        model.fit(X[:150], y[:150], X[150:], y[150:])
        preds = model.predict(X[150:])
        assert preds.shape == (50,)

    def test_predict_with_confidence(self):
        from src.models.bnn import BNNModel
        np.random.seed(42)
        X = np.random.randn(200, 20).astype(np.float32)
        y = np.random.randn(200).astype(np.float32) * 0.01
        model = BNNModel(input_size=20, max_epochs=5, n_samples=5)
        model.fit(X[:150], y[:150])
        preds, conf = model.predict_with_confidence(X[150:])
        assert preds.shape == conf.shape
        assert (conf >= 0).all() and (conf <= 1).all()
