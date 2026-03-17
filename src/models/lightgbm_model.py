"""Modelo LightGBM para regressao e classificacao direcional."""

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMRegressor(BaseModel):
    """LightGBM para prever retorno (regressao)."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    @property
    def name(self) -> str:
        return "lightgbm_regressor"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        fit_params: dict = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ]

        self.model.fit(X_train, y_train, **fit_params)

        # Metricas de treino
        metrics = {
            "train_rmse": float(np.sqrt(np.mean((self.model.predict(X_train) - y_train) ** 2)))
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
        """Confianca baseada na variancia entre as arvores individuais."""
        preds = self.predict(X)

        # Obter previsoes individuais de cada arvore
        booster = self.model.booster_
        n_trees = booster.num_trees()
        tree_preds = np.zeros((len(X), n_trees))
        for i in range(n_trees):
            tree_preds[:, i] = booster.predict(X, start_iteration=i, num_iteration=1, raw_score=True)

        # Variancia entre arvores como medida de incerteza
        std_pred = tree_preds.std(axis=1)
        max_std = std_pred.max() if std_pred.max() > 0 else 1.0
        confidence = 1.0 - np.clip(std_pred / max_std, 0, 1)

        return preds, confidence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Modelo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "LightGBMRegressor":
        instance = cls()
        instance.model = joblib.load(path)
        return instance

    def get_feature_importance(self) -> dict[str, float]:
        """Retorna importancia das features."""
        names = self.model.feature_name_ or []
        importances = self.model.feature_importances_
        return dict(zip(names, importances))


class LightGBMClassifier(BaseModel):
    """LightGBM para prever direcao (classificacao binaria)."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    @property
    def name(self) -> str:
        return "lightgbm_classifier"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        # Converter para binario (0/1)
        y_train_dir = (y_train > 0).astype(int)

        fit_params: dict = {}
        if X_val is not None and y_val is not None:
            y_val_dir = (y_val > 0).astype(int)
            fit_params["eval_set"] = [(X_val, y_val_dir)]
            fit_params["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ]

        self.model.fit(X_train, y_train_dir, **fit_params)

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
    def load(cls, path: Path) -> "LightGBMClassifier":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
