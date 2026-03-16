"""Ensemble: combina previsoes de multiplos modelos."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Combina previsoes de modelos base via stacking ou media ponderada."""

    def __init__(self, method: str = "stacking"):
        """
        Args:
            method: "stacking" (Ridge meta-learner) ou "weighted" (media ponderada)
        """
        self.method = method
        self.meta_model: Ridge | None = None
        self.weights: np.ndarray | None = None
        self.n_models: int = 0

    @property
    def name(self) -> str:
        return f"ensemble_{self.method}"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Treina o ensemble.

        X_train: shape (n_samples, n_models) — previsoes dos modelos base
        y_train: shape (n_samples,) — targets reais
        """
        self.n_models = X_train.shape[1]

        if self.method == "stacking":
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(X_train, y_train)
            train_pred = self.meta_model.predict(X_train)
        else:
            # Media ponderada otimizada via minimos quadrados
            self.weights = self._optimize_weights(X_train, y_train)
            train_pred = X_train @ self.weights

        metrics = {
            "train_rmse": float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
            metrics["val_dir_acc"] = float(
                np.mean(np.sign(val_pred) == np.sign(y_val))
            )

        if self.method == "stacking" and self.meta_model is not None:
            logger.info(f"  Ensemble coeficientes: {self.meta_model.coef_}")
        elif self.weights is not None:
            logger.info(f"  Ensemble pesos: {self.weights}")

        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    def _optimize_weights(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Encontra pesos otimos via minimos quadrados com restricao de soma=1."""
        # Resolver via pseudoinversa com regularizacao
        n_models = predictions.shape[1]
        # Adicionar restricao suave de soma = 1
        reg = 0.01 * np.eye(n_models)
        weights = np.linalg.solve(
            predictions.T @ predictions + reg,
            predictions.T @ targets,
        )
        # Normalizar para somar 1
        weights = np.abs(weights)
        weights = weights / weights.sum()
        return weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: shape (n_samples, n_models) — previsoes dos modelos base."""
        if self.method == "stacking" and self.meta_model is not None:
            return self.meta_model.predict(X)
        elif self.weights is not None:
            return X @ self.weights
        raise RuntimeError("Ensemble nao treinado")

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Confianca baseada na concordancia entre modelos."""
        preds = self.predict(X)

        # Concordancia: quao parecidas sao as previsoes dos modelos
        model_std = X.std(axis=1)  # std entre modelos para cada sample
        max_std = model_std.max() if model_std.max() > 0 else 1.0
        confidence = 1.0 - np.clip(model_std / max_std, 0, 1)

        # Boost de confianca quando todos os modelos concordam na direcao
        all_same_sign = np.all(X > 0, axis=1) | np.all(X < 0, axis=1)
        confidence[all_same_sign] = np.clip(
            confidence[all_same_sign] + 0.1, 0, 1
        )

        return preds, confidence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "method": self.method,
            "meta_model": self.meta_model,
            "weights": self.weights,
            "n_models": self.n_models,
        }, path)
        logger.info(f"Ensemble salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        data = joblib.load(path)
        instance = cls(method=data["method"])
        instance.meta_model = data["meta_model"]
        instance.weights = data["weights"]
        instance.n_models = data["n_models"]
        return instance
