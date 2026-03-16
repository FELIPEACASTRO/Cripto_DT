"""Testes para modelos de ML."""

import numpy as np
import pytest
import torch

from src.models.losses import MADLLoss, DirectionalHuberLoss
from src.models.xgboost_model import XGBoostRegressor, XGBoostClassifier
from src.models.ensemble import EnsembleModel


class TestMADLLoss:

    def test_correct_direction_lower_loss(self):
        """Previsoes na direcao correta devem ter loss menor."""
        loss_fn = MADLLoss(alpha=2.0)
        y_true = torch.tensor([0.01, -0.01, 0.02])

        # Direcao correta
        y_pred_correct = torch.tensor([0.005, -0.005, 0.01])
        loss_correct = loss_fn(y_pred_correct, y_true)

        # Direcao errada (mesma magnitude)
        y_pred_wrong = torch.tensor([-0.005, 0.005, -0.01])
        loss_wrong = loss_fn(y_pred_wrong, y_true)

        assert loss_wrong > loss_correct

    def test_zero_alpha_equals_mae(self):
        """Com alpha=0, MADL deve ser equivalente a MAE."""
        madl = MADLLoss(alpha=0.0, base_loss="mae")
        mae = torch.nn.L1Loss()
        y_true = torch.tensor([0.01, -0.02, 0.03])
        y_pred = torch.tensor([-0.005, 0.01, -0.02])
        assert abs(madl(y_pred, y_true).item() - mae(y_pred, y_true).item()) < 1e-6

    def test_output_is_scalar(self):
        loss_fn = MADLLoss()
        y_true = torch.randn(32)
        y_pred = torch.randn(32)
        result = loss_fn(y_pred, y_true)
        assert result.dim() == 0


class TestDirectionalHuberLoss:

    def test_output_is_scalar(self):
        loss_fn = DirectionalHuberLoss()
        y_true = torch.randn(32)
        y_pred = torch.randn(32)
        result = loss_fn(y_pred, y_true)
        assert result.dim() == 0

    def test_gradient_flows(self):
        loss_fn = DirectionalHuberLoss()
        y_pred = torch.randn(16, requires_grad=True)
        y_true = torch.randn(16)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None


class TestXGBoostRegressor:

    def test_fit_and_predict(self, sample_features):
        X, y = sample_features
        X_train, y_train = X[:150], y[:150]
        X_val, y_val = X[150:], y[150:]

        model = XGBoostRegressor()
        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert "train_rmse" in metrics
        assert "val_rmse" in metrics

        preds = model.predict(X_val)
        assert preds.shape == (50,)

    def test_predict_with_confidence(self, sample_features):
        X, y = sample_features
        model = XGBoostRegressor()
        model.fit(X[:150], y[:150])
        preds, conf = model.predict_with_confidence(X[150:])
        assert preds.shape == conf.shape
        assert (conf >= 0).all() and (conf <= 1).all()


class TestXGBoostClassifier:

    def test_fit_and_predict(self, sample_features):
        X, y = sample_features
        model = XGBoostClassifier()
        metrics = model.fit(X[:150], y[:150], X[150:], y[150:])
        assert "train_acc" in metrics

        preds = model.predict(X[150:])
        assert preds.shape == (50,)
        # Probabilidades devem estar entre 0 e 1
        assert (preds >= 0).all() and (preds <= 1).all()


class TestEnsembleModel:

    def test_stacking(self):
        np.random.seed(42)
        n = 100
        # Simular 3 modelos base
        y_true = np.random.randn(n) * 0.01
        preds = np.column_stack([
            y_true + np.random.randn(n) * 0.005,
            y_true + np.random.randn(n) * 0.008,
            y_true + np.random.randn(n) * 0.003,
        ])

        model = EnsembleModel(method="stacking")
        metrics = model.fit(preds[:70], y_true[:70], preds[70:], y_true[70:])
        assert "train_rmse" in metrics

        result = model.predict(preds[70:])
        assert result.shape == (30,)

    def test_confidence_range(self):
        np.random.seed(42)
        n = 50
        preds = np.random.randn(n, 3) * 0.01
        y = np.random.randn(n) * 0.01

        model = EnsembleModel(method="stacking")
        model.fit(preds, y)
        result, conf = model.predict_with_confidence(preds)
        assert (conf >= 0).all() and (conf <= 1).all()
