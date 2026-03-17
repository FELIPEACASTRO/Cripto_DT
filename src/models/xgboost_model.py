"""Modelo XGBoost para regressao e classificacao direcional."""

import logging
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from config.settings import XGBoostConfig, config as default_config
from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostRegressor(BaseModel):
    """XGBoost para prever retorno (regressao)."""

    def __init__(self, xgb_config: XGBoostConfig | None = None):
        self.cfg = xgb_config or default_config.xgboost
        self.model = xgb.XGBRegressor(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            random_state=42,
            n_jobs=-1,
        )

    @property
    def name(self) -> str:
        return "xgboost_regressor"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)

        metrics = {"train_rmse": float(np.sqrt(np.mean((self.model.predict(X_train) - y_train) ** 2)))}
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
        preds = self.predict(X)
        # Confianca baseada na magnitude da previsao relativa ao historico
        abs_preds = np.abs(preds)
        max_pred = abs_preds.max() if abs_preds.max() > 0 else 1.0
        confidence = np.clip(abs_preds / max_pred, 0.1, 1.0)
        return preds, confidence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Modelo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "XGBoostRegressor":
        instance = cls()
        instance.model = joblib.load(path)
        return instance

    def get_feature_importance(self) -> dict[str, float]:
        """Retorna importancia das features."""
        return dict(zip(
            self.model.get_booster().feature_names or [],
            self.model.feature_importances_,
        ))


class XGBoostClassifier(BaseModel):
    """XGBoost para prever direcao (classificacao binaria)."""

    def __init__(self, xgb_config: XGBoostConfig | None = None):
        self.cfg = xgb_config or default_config.xgboost
        self.model = xgb.XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    @property
    def name(self) -> str:
        return "xgboost_classifier"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        # Converter para binario (0/1)
        y_train_dir = (y_train > 0).astype(int)

        fit_params = {}
        if X_val is not None and y_val is not None:
            y_val_dir = (y_val > 0).astype(int)
            fit_params["eval_set"] = [(X_val, y_val_dir)]
            fit_params["verbose"] = False

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
    def load(cls, path: Path) -> "XGBoostClassifier":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
