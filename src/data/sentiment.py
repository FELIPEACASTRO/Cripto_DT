"""Coleta de dados de sentimento: Fear & Greed Index."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class SentimentCollector:
    """Coleta Fear & Greed Index e deriva features de sentimento."""

    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    def fetch_fear_greed(self, days: int = 730) -> pd.DataFrame:
        """Busca historico do Fear & Greed Index.

        Args:
            days: Numero de dias de historico

        Returns:
            DataFrame com colunas [timestamp, fg_value, fg_classification]
        """
        try:
            resp = requests.get(
                self.FEAR_GREED_URL,
                params={"limit": days, "format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except requests.RequestException as e:
            logger.error(f"Erro ao buscar Fear & Greed Index: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        records = []
        for entry in data:
            records.append({
                "timestamp": pd.to_datetime(
                    int(entry["timestamp"]), unit="s", utc=True
                ),
                "fg_value": int(entry["value"]),
                "fg_classification": entry["value_classification"],
            })

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Fear & Greed: {len(df)} dias coletados")
        return df

    def add_sentiment_features(
        self, df: pd.DataFrame, fg_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge Fear & Greed com dados OHLCV e gera features derivadas.

        Args:
            df: DataFrame OHLCV com coluna 'timestamp'
            fg_df: DataFrame do Fear & Greed Index

        Returns:
            DataFrame com features de sentimento adicionadas
        """
        if fg_df.empty:
            logger.warning("Fear & Greed vazio, retornando sem sentiment features")
            return df

        df = df.copy()

        # Normalizar timestamps para date (Fear & Greed e diario)
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        fg_df = fg_df.copy()
        fg_df["_date"] = pd.to_datetime(fg_df["timestamp"]).dt.normalize()

        # Merge por data
        fg_subset = fg_df[["_date", "fg_value"]].drop_duplicates(subset=["_date"])
        df = df.merge(fg_subset, on="_date", how="left")

        # Forward-fill para gaps (finais de semana)
        df["fg_value"] = df["fg_value"].ffill()

        # Features derivadas
        df["fg_value"] = df["fg_value"].astype(float)

        # Normalizar para 0-1
        df["fg_normalized"] = df["fg_value"] / 100.0

        # Mudanca do F&G
        df["fg_change_1d"] = df["fg_value"].diff(1)
        df["fg_change_7d"] = df["fg_value"].diff(7)

        # Media movel do F&G
        df["fg_ma_7"] = df["fg_value"].rolling(7).mean()
        df["fg_ma_14"] = df["fg_value"].rolling(14).mean()

        # Desvio do F&G em relacao a media (extremos)
        df["fg_zscore"] = (
            (df["fg_value"] - df["fg_ma_14"])
            / df["fg_value"].rolling(14).std().replace(0, np.nan)
        )

        # Indicador binario de medo extremo / ganancia extrema
        df["fg_extreme_fear"] = (df["fg_value"] < 25).astype(float)
        df["fg_extreme_greed"] = (df["fg_value"] > 75).astype(float)

        df = df.drop(columns=["_date"])
        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features de sentimento."""
        return [
            "fg_value", "fg_normalized",
            "fg_change_1d", "fg_change_7d",
            "fg_ma_7", "fg_ma_14", "fg_zscore",
            "fg_extreme_fear", "fg_extreme_greed",
        ]
