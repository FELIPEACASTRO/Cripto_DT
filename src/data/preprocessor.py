"""Preprocessamento de dados: limpeza, normalizacao e calculo de targets."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Limpa e prepara dados OHLCV para feature engineering."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dados OHLCV brutos.

        - Remove duplicatas
        - Forward-fill gaps
        - Remove linhas com volume zero ou precos negativos
        """
        if df.empty:
            return df

        df = df.copy()
        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Forward-fill gaps (exchange downtime)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].ffill()

        # Volume zero e ok (mercado parado), mas negativo nao
        df = df[df["close"] > 0].reset_index(drop=True)
        df["volume"] = df["volume"].clip(lower=0)

        n_removed = len(df) - len(df.dropna(subset=["close"]))
        if n_removed > 0:
            logger.warning(f"Removidas {n_removed} linhas com dados invalidos")
            df = df.dropna(subset=["close"]).reset_index(drop=True)

        return df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas de retorno."""
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["pct_return"] = df["close"].pct_change()
        df["direction"] = (df["log_return"] > 0).astype(int)
        return df

    def add_target(
        self, df: pd.DataFrame, horizon: int = 1, target_type: str = "log_return"
    ) -> pd.DataFrame:
        """Adiciona coluna target (valor futuro a prever).

        Args:
            df: DataFrame com coluna 'log_return' ou 'close'
            horizon: Passos a frente para previsao
            target_type: "log_return" ou "direction"
        """
        df = df.copy()
        if target_type == "log_return":
            df["target"] = df["log_return"].shift(-horizon)
        elif target_type == "direction":
            df["target"] = df["direction"].shift(-horizon)
        else:
            raise ValueError(f"target_type invalido: {target_type}")
        return df

    def prepare(
        self, df: pd.DataFrame, horizon: int = 1, target_type: str = "log_return"
    ) -> pd.DataFrame:
        """Pipeline completo de preprocessamento."""
        df = self.clean(df)
        df = self.add_returns(df)
        df = self.add_target(df, horizon, target_type)
        return df


class FeatureScaler:
    """Scaler que fita apenas nos dados de treino (previne data leakage)."""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.feature_columns: list[str] = []

    def fit(self, df: pd.DataFrame, feature_columns: list[str]) -> "FeatureScaler":
        """Fita o scaler nos dados de treino."""
        self.feature_columns = feature_columns
        self.scaler.fit(df[feature_columns].values)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma features (nao altera colunas non-feature)."""
        if not self._is_fitted:
            raise RuntimeError("Scaler nao foi fitado. Chame fit() primeiro.")
        df = df.copy()
        df[self.feature_columns] = self.scaler.transform(
            df[self.feature_columns].values
        )
        return df

    def fit_transform(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """Fit + transform em uma chamada."""
        self.fit(df, feature_columns)
        return self.transform(df)
