"""Random Forest para regressao e classificacao de criptomoedas."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestRegressorModel(BaseModel):
    """Random Forest para prever retorno (regressao)."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    @property
    def name(self) -> str:
        return "rf_reg"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        metrics: dict[str, float] = {
            "train_rmse": float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
            metrics["val_mae"] = float(mean_absolute_error(y_val, val_pred))
            metrics["val_dir_acc"] = float(np.mean(np.sign(val_pred) == np.sign(y_val)))

        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Confianca baseada na variancia entre arvores individuais.

        Quanto menor a variancia entre as previsoes das arvores,
        maior a confianca do modelo na previsao.
        """
        # Coleta previsao de cada arvore individual
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])  # (n_estimators, n_samples)

        mean_pred = tree_predictions.mean(axis=0)
        std_pred = tree_predictions.std(axis=0)

        # Confianca inversamente proporcional a variancia
        max_std = std_pred.max() if std_pred.max() > 0 else 1.0
        confidence = 1.0 - np.clip(std_pred / max_std, 0, 1)

        return mean_pred, confidence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Modelo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "RandomForestRegressorModel":
        instance = cls()
        instance.model = joblib.load(path)
        return instance

    def get_feature_importance(self) -> dict[str, float]:
        """Retorna importancia das features."""
        return dict(enumerate(self.model.feature_importances_))


class RandomForestClassifierModel(BaseModel):
    """Random Forest para prever direcao (classificacao binaria)."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    @property
    def name(self) -> str:
        return "rf_cls"

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

        train_pred = self.model.predict(X_train)
        metrics: dict[str, float] = {
            "train_acc": float(accuracy_score(y_train_dir, train_pred)),
            "train_f1": float(f1_score(y_train_dir, train_pred, zero_division=0)),
        }

        if X_val is not None and y_val is not None:
            y_val_dir = (y_val > 0).astype(int)
            val_pred = self.model.predict(X_val)
            metrics["val_acc"] = float(accuracy_score(y_val_dir, val_pred))
            metrics["val_f1"] = float(f1_score(y_val_dir, val_pred, zero_division=0))

        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidade da classe positiva (subida)."""
        return self.model.predict_proba(X)[:, 1]

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Confianca baseada na variancia entre arvores individuais.

        Cada arvore vota em uma classe; a confianca reflete
        o grau de concordancia entre as arvores.
        """
        # Coleta previsao de cada arvore individual
        tree_predictions = np.array([
            tree.predict(X.reshape(X.shape[0], -1) if X.ndim > 2 else X)
            for tree in self.model.estimators_
        ])  # (n_estimators, n_samples)

        # Proporcao de votos para classe 1 (subida)
        proba_up = tree_predictions.mean(axis=0)

        # Converter para sinal direcional (-1, +1)
        preds = np.where(proba_up > 0.5, 1.0, -1.0)

        # Confianca = distancia de 0.5 (consenso entre arvores)
        confidence = np.abs(proba_up - 0.5) * 2  # 0 a 1

        return preds, confidence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Modelo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "RandomForestClassifierModel":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
