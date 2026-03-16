"""Modelo SVM para regressao e classificacao direcional."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC, SVR

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class SVMRegressor(BaseModel):
    """SVM para prever retorno (regressao)."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.01,
        gamma: str = "scale",
    ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma

        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
        )

    @property
    def name(self) -> str:
        return "svm_regressor"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        self.model.fit(X_train, y_train)

        # Metricas de treino
        train_pred = self.model.predict(X_train)
        metrics = {
            "train_rmse": float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
        }
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
            metrics["val_dir_acc"] = float(np.mean(np.sign(val_pred) == np.sign(y_val)))

        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Confianca baseada na distancia ao hiperplano normalizada."""
        preds = self.predict(X)

        # Para SVR, usamos a magnitude da previsao relativa ao epsilon-tube
        # SVR nao tem decision_function — usamos abs(pred) como proxy de confianca
        abs_preds = np.abs(preds)
        max_pred = abs_preds.max() if abs_preds.max() > 0 else 1.0
        confidence = np.clip(abs_preds / max_pred, 0.1, 1.0)

        return preds, confidence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Modelo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "SVMRegressor":
        instance = cls()
        instance.model = joblib.load(path)
        return instance


class SVMClassifier(BaseModel):
    """SVM para prever direcao (classificacao binaria)."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        probability: bool = True,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability

        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=42,
        )

    @property
    def name(self) -> str:
        return "svm_classifier"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        # Converter para binario (0/1)
        y_train_dir = (y_train > 0).astype(int)

        self.model.fit(X_train, y_train_dir)

        metrics = {"train_acc": float(self.model.score(X_train, y_train_dir))}
        if X_val is not None and y_val is not None:
            y_val_dir = (y_val > 0).astype(int)
            metrics["val_acc"] = float(self.model.score(X_val, y_val_dir))

        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidade da classe positiva (subida)."""
        return self.model.predict_proba(X)[:, 1]

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        proba = self.model.predict_proba(X)[:, 1]
        # Converter para sinal direcional (-1, +1) baseado em 0.5
        preds = np.where(proba > 0.5, 1.0, -1.0)
        # Confianca = distancia de 0.5
        confidence = np.abs(proba - 0.5) * 2  # 0 a 1
        return preds, confidence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Modelo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "SVMClassifier":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
