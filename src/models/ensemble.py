"""Ensemble: combina previsoes de multiplos modelos."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

# Minimum directional accuracy to include a model in the ensemble.
# Models at or below 0.52 are barely better than random and add noise.
MIN_DIR_ACCURACY = 0.52


class EnsembleModel(BaseModel):
    """Combina previsoes de modelos base via stacking ou media ponderada.

    Supports three methods:
    - "stacking": Ridge meta-learner (low regularization)
    - "weighted": Least-squares optimized weights (sum to 1)
    - "performance_weighted": Weights proportional to validation directional
      accuracy, with weak model filtering (recommended for directional tasks)
    """

    def __init__(self, method: str = "performance_weighted"):
        """
        Args:
            method: "stacking", "weighted", or "performance_weighted"
        """
        self.method = method
        self.meta_model: Ridge | None = None
        self.weights: np.ndarray | None = None
        self.n_models: int = 0
        self.model_names: list[str] = []
        self.selected_mask: np.ndarray | None = None
        self.val_dir_accuracies: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"ensemble_{self.method}"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        model_names: list[str] | None = None,
        val_dir_accuracies: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Treina o ensemble.

        X_train: shape (n_samples, n_models) -- previsoes dos modelos base
        y_train: shape (n_samples,) -- targets reais
        model_names: optional list of model names (for logging)
        val_dir_accuracies: per-model directional accuracy on validation set.
            Used by "performance_weighted" to filter and weight models.
            If not provided, it is computed from X_train and y_train.
        """
        self.n_models = X_train.shape[1]
        self.model_names = model_names or [f"model_{i}" for i in range(self.n_models)]

        # Compute val directional accuracies if not provided
        if val_dir_accuracies is not None:
            self.val_dir_accuracies = np.asarray(val_dir_accuracies)
        else:
            # Estimate from the training matrix (which is val predictions for stacking)
            self.val_dir_accuracies = np.array([
                float(np.mean(np.sign(X_train[:, i]) == np.sign(y_train)))
                for i in range(self.n_models)
            ])

        # Log per-model directional accuracy
        for i, mname in enumerate(self.model_names):
            logger.info(f"    {mname}: val_dir_acc={self.val_dir_accuracies[i]:.4f}")

        # Filter out weak models
        self.selected_mask = self.val_dir_accuracies >= MIN_DIR_ACCURACY
        n_selected = int(self.selected_mask.sum())

        if n_selected == 0:
            # Fallback: keep all models if none pass the threshold
            logger.warning(
                "  No models pass MIN_DIR_ACCURACY threshold "
                f"({MIN_DIR_ACCURACY:.2f}). Keeping all models."
            )
            self.selected_mask = np.ones(self.n_models, dtype=bool)
            n_selected = self.n_models

        excluded = [
            self.model_names[i]
            for i in range(self.n_models)
            if not self.selected_mask[i]
        ]
        if excluded:
            logger.info(
                f"  Ensemble: excluding weak models: {excluded}"
            )
        logger.info(f"  Ensemble: using {n_selected}/{self.n_models} models")

        # Work only with selected models
        X_train_sel = X_train[:, self.selected_mask]
        X_val_sel = X_val[:, self.selected_mask] if X_val is not None else None
        sel_accuracies = self.val_dir_accuracies[self.selected_mask]

        if self.method == "stacking":
            # Use low regularization so the meta-learner can properly weight
            self.meta_model = Ridge(alpha=0.01)
            self.meta_model.fit(X_train_sel, y_train)
            train_pred = self.meta_model.predict(X_train_sel)

        elif self.method == "performance_weighted":
            self.weights = self._performance_weights(sel_accuracies)
            train_pred = X_train_sel @ self.weights

        else:
            # Least-squares optimized weights
            self.weights = self._optimize_weights(X_train_sel, y_train)
            train_pred = X_train_sel @ self.weights

        metrics = {
            "train_rmse": float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
        }

        if X_val_sel is not None and y_val is not None:
            val_pred = self._predict_internal(X_val_sel)
            metrics["val_rmse"] = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
            metrics["val_dir_acc"] = float(
                np.mean(np.sign(val_pred) == np.sign(y_val))
            )

        if self.method == "stacking" and self.meta_model is not None:
            logger.info(f"  Ensemble coeficientes: {self.meta_model.coef_}")
        elif self.weights is not None:
            weight_info = {
                self.model_names[j]: f"{self.weights[k]:.4f}"
                for k, j in enumerate(np.where(self.selected_mask)[0])
            }
            logger.info(f"  Ensemble pesos: {weight_info}")

        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    @staticmethod
    def _performance_weights(dir_accuracies: np.ndarray) -> np.ndarray:
        """Compute weights proportional to (dir_accuracy - 0.5)^2.

        Using the squared excess over 0.5 strongly favors models with higher
        directional accuracy and penalizes near-random models.
        """
        # Excess accuracy above random
        excess = np.clip(dir_accuracies - 0.5, 0.0, None)
        # Square to give more weight to the best models
        raw_weights = excess ** 2
        total = raw_weights.sum()
        if total == 0:
            # Uniform fallback
            return np.ones(len(dir_accuracies)) / len(dir_accuracies)
        return raw_weights / total

    def _optimize_weights(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Encontra pesos otimos via minimos quadrados com restricao de soma=1."""
        n_models = predictions.shape[1]
        reg = 0.001 * np.eye(n_models)
        weights = np.linalg.solve(
            predictions.T @ predictions + reg,
            predictions.T @ targets,
        )
        # Normalizar para somar 1
        weights = np.abs(weights)
        total = weights.sum()
        if total == 0:
            return np.ones(n_models) / n_models
        return weights / total

    def _predict_internal(self, X_sel: np.ndarray) -> np.ndarray:
        """Predict using already-selected columns."""
        if self.method == "stacking" and self.meta_model is not None:
            return self.meta_model.predict(X_sel)
        elif self.weights is not None:
            return X_sel @ self.weights
        raise RuntimeError("Ensemble nao treinado")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: shape (n_samples, n_models) -- previsoes dos modelos base."""
        if self.selected_mask is not None:
            X_sel = X[:, self.selected_mask]
        else:
            X_sel = X
        return self._predict_internal(X_sel)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Confianca baseada na concordancia entre modelos."""
        if self.selected_mask is not None:
            X_sel = X[:, self.selected_mask]
        else:
            X_sel = X

        preds = self._predict_internal(X_sel)

        # Concordancia: quao parecidas sao as previsoes dos modelos
        model_std = X_sel.std(axis=1)
        max_std = model_std.max() if model_std.max() > 0 else 1.0
        confidence = 1.0 - np.clip(model_std / max_std, 0, 1)

        # Boost de confianca quando todos os modelos concordam na direcao
        all_same_sign = np.all(X_sel > 0, axis=1) | np.all(X_sel < 0, axis=1)
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
            "model_names": self.model_names,
            "selected_mask": self.selected_mask,
            "val_dir_accuracies": self.val_dir_accuracies,
        }, path)
        logger.info(f"Ensemble salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        data = joblib.load(path)
        instance = cls(method=data["method"])
        instance.meta_model = data["meta_model"]
        instance.weights = data["weights"]
        instance.n_models = data["n_models"]
        instance.model_names = data.get("model_names", [])
        instance.selected_mask = data.get("selected_mask")
        instance.val_dir_accuracies = data.get("val_dir_accuracies")
        return instance
