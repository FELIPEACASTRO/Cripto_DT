"""Deteccao de anomalias via Z-score rolling."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detecta anomalias em series temporais financeiras usando Z-score rolling."""

    def __init__(self, window: int = 60, threshold: float = 3.0):
        """Inicializa o detector de anomalias.

        Args:
            window: Janela rolling para calculo do Z-score (padrao: 60 dias)
            threshold: Limiar de |Z-score| para marcar anomalia (padrao: 3.0)
        """
        self.window = window
        self.threshold = threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula Z-scores rolling e marca anomalias.

        Espera colunas: close (para retornos), volume, e opcionalmente
        atr ou atr_pct (para volatilidade).

        Args:
            df: DataFrame com dados OHLCV e possivelmente features tecnicas

        Returns:
            DataFrame com features de anomalia adicionadas
        """
        df = df.copy()

        # --- Z-score do retorno ---
        if "close" in df.columns:
            returns = df["close"].pct_change()
            ret_mean = returns.rolling(self.window, min_periods=10).mean()
            ret_std = returns.rolling(self.window, min_periods=10).std()
            df["anomaly_return_zscore"] = (
                (returns - ret_mean) / ret_std.replace(0, np.nan)
            )
        else:
            df["anomaly_return_zscore"] = np.nan
            logger.warning("Coluna 'close' nao encontrada para Z-score de retorno")

        # --- Z-score do volume ---
        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            vol_mean = vol.rolling(self.window, min_periods=10).mean()
            vol_std = vol.rolling(self.window, min_periods=10).std()
            df["anomaly_volume_zscore"] = (
                (vol - vol_mean) / vol_std.replace(0, np.nan)
            )
        else:
            df["anomaly_volume_zscore"] = np.nan
            logger.warning("Coluna 'volume' nao encontrada para Z-score de volume")

        # --- Z-score da volatilidade ---
        # Usar ATR percentual se disponivel, senao estimar volatilidade pelo retorno
        if "atr_pct" in df.columns:
            volatility = df["atr_pct"]
        elif "close" in df.columns:
            # Volatilidade estimada: desvio padrao rolling dos retornos
            returns = df["close"].pct_change()
            volatility = returns.rolling(5, min_periods=2).std()
        else:
            volatility = pd.Series(np.nan, index=df.index)

        vol_mean = volatility.rolling(self.window, min_periods=10).mean()
        vol_std = volatility.rolling(self.window, min_periods=10).std()
        df["anomaly_volatility_zscore"] = (
            (volatility - vol_mean) / vol_std.replace(0, np.nan)
        )

        # --- Score agregado: maximo dos valores absolutos ---
        abs_scores = pd.DataFrame({
            "r": df["anomaly_return_zscore"].abs(),
            "v": df["anomaly_volume_zscore"].abs(),
            "vol": df["anomaly_volatility_zscore"].abs(),
        })
        df["anomaly_score"] = abs_scores.max(axis=1)

        # --- Flag de anomalia ---
        df["anomaly_flag"] = (df["anomaly_score"] > self.threshold).astype(float)

        # --- Contagem de anomalias nos ultimos 7 dias ---
        df["anomaly_count_7d"] = (
            df["anomaly_flag"].rolling(7, min_periods=1).sum()
        )

        logger.info(
            f"Anomalias detectadas: {int(df['anomaly_flag'].sum())} em "
            f"{len(df)} registros ({df['anomaly_flag'].mean() * 100:.1f}%)"
        )
        return df

    def get_anomaly_report(self, df: pd.DataFrame) -> dict:
        """Gera relatorio resumido das anomalias detectadas.

        Args:
            df: DataFrame ja processado com features de anomalia

        Returns:
            Dicionario com estatisticas de anomalias
        """
        if "anomaly_flag" not in df.columns:
            logger.warning("DataFrame nao contem features de anomalia. Execute transform() primeiro.")
            return {}

        total = len(df)
        total_anomalias = int(df["anomaly_flag"].sum())
        pct_anomalos = (total_anomalias / total * 100) if total > 0 else 0.0

        # Determinar tipo mais frequente de anomalia
        tipos = {
            "retorno": (df["anomaly_return_zscore"].abs() > self.threshold).sum(),
            "volume": (df["anomaly_volume_zscore"].abs() > self.threshold).sum(),
            "volatilidade": (df["anomaly_volatility_zscore"].abs() > self.threshold).sum(),
        }
        tipo_mais_frequente = max(tipos, key=tipos.get) if any(tipos.values()) else "nenhum"

        report = {
            "total_registros": total,
            "total_anomalias": total_anomalias,
            "pct_dias_anomalos": round(pct_anomalos, 2),
            "tipo_mais_frequente": tipo_mais_frequente,
            "anomalias_por_tipo": {k: int(v) for k, v in tipos.items()},
            "max_anomaly_score": round(float(df["anomaly_score"].max()), 2)
            if not df["anomaly_score"].isna().all()
            else None,
        }

        logger.info(
            f"Relatorio de anomalias: {total_anomalias} anomalias "
            f"({pct_anomalos:.1f}%), tipo mais frequente: {tipo_mais_frequente}"
        )
        return report

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features de anomalia geradas."""
        return [
            "anomaly_return_zscore",
            "anomaly_volume_zscore",
            "anomaly_volatility_zscore",
            "anomaly_score",
            "anomaly_flag",
            "anomaly_count_7d",
        ]
