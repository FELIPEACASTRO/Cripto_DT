"""Fusao de features multi-timeframe: agrega dados intradiarios em features diarias."""

import logging

import numpy as np
import pandas as pd

from config.settings import Config, config as default_config

logger = logging.getLogger(__name__)

# Prefixos por timeframe de origem
_TF_PREFIX = {
    "1h": "tf1h",
    "4h": "tf4h",
}


class TimeframeFusion:
    """Agrega dados de timeframes menores (1h, 4h) em features diarias.

    Para cada janela de 1 dia no timeframe de alta frequencia, calcula:
      - intraday_volatility: desvio padrao dos retornos
      - intraday_range: (max - min) / close
      - intraday_volume_profile: proporcao volume 1a metade / 2a metade
      - intraday_momentum: retorno das ultimas N velas vs primeiras N
      - intraday_vwap_distance: (close - VWAP) / close
      - max_drawdown_intraday: maior drawdown intra-dia
      - n_direction_changes: quantidade de mudancas de sinal nos retornos
    """

    # Nomes-base das features geradas por timeframe
    _BASE_FEATURES = [
        "intraday_volatility",
        "intraday_range",
        "intraday_volume_first_half_ratio",
        "intraday_momentum",
        "intraday_vwap_distance",
        "max_drawdown_intraday",
        "n_direction_changes",
    ]

    def __init__(self, config: Config = default_config):
        self.config = config

    # ------------------------------------------------------------------
    # Agregacao de um unico timeframe
    # ------------------------------------------------------------------

    def aggregate_timeframe(
        self,
        df_high_freq: pd.DataFrame,
        target_tf: str = "1d",
    ) -> pd.DataFrame:
        """Agrega dados de alta frequencia em features diarias.

        Args:
            df_high_freq: DataFrame OHLCV com coluna ``timestamp``
                (DatetimeIndex ou coluna). Deve conter: open, high, low,
                close, volume.
            target_tf: Timeframe alvo (atualmente apenas ``"1d"``).

        Returns:
            DataFrame indexado por data (datetime.date normalizado) com as
            features intra-dia calculadas.
        """
        df = df_high_freq.copy()

        # Garantir que temos um DatetimeIndex
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        # Normalizar para data (agrupar por dia)
        df["_date"] = df.index.normalize()

        # Retornos intra-dia
        df["_return"] = df["close"].pct_change()

        grouped = df.groupby("_date")
        records: list[dict] = []

        for date, group in grouped:
            if len(group) < 2:
                continue

            returns = group["_return"].dropna().values
            closes = group["close"].values
            highs = group["high"].values
            lows = group["low"].values
            volumes = group["volume"].values

            record: dict = {"date": date}

            # 1. Volatilidade intradiaria
            record["intraday_volatility"] = (
                float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
            )

            # 2. Range intradiario
            day_close = closes[-1]
            if day_close > 0:
                record["intraday_range"] = float(
                    (highs.max() - lows.min()) / day_close
                )
            else:
                record["intraday_range"] = 0.0

            # 3. Perfil de volume: razao 1a metade / total
            mid = len(volumes) // 2
            vol_first = volumes[:mid].sum()
            vol_total = volumes.sum()
            if vol_total > 0:
                record["intraday_volume_first_half_ratio"] = float(
                    vol_first / vol_total
                )
            else:
                record["intraday_volume_first_half_ratio"] = 0.5

            # 4. Momentum intradiario: retorno ultima metade vs primeira metade
            mid_r = len(returns) // 2
            if mid_r > 0:
                first_half_return = float(np.sum(returns[:mid_r]))
                second_half_return = float(np.sum(returns[mid_r:]))
                record["intraday_momentum"] = second_half_return - first_half_return
            else:
                record["intraday_momentum"] = 0.0

            # 5. Distancia VWAP
            if vol_total > 0:
                typical_price = (highs + lows + closes) / 3.0
                vwap = float(np.sum(typical_price * volumes) / vol_total)
                if day_close > 0:
                    record["intraday_vwap_distance"] = float(
                        (day_close - vwap) / day_close
                    )
                else:
                    record["intraday_vwap_distance"] = 0.0
            else:
                record["intraday_vwap_distance"] = 0.0

            # 6. Maximo drawdown intradiario
            if len(closes) > 1:
                cummax = np.maximum.accumulate(closes)
                drawdowns = (closes - cummax) / np.where(cummax > 0, cummax, 1.0)
                record["max_drawdown_intraday"] = float(drawdowns.min())
            else:
                record["max_drawdown_intraday"] = 0.0

            # 7. Numero de mudancas de direcao
            if len(returns) > 1:
                signs = np.sign(returns)
                changes = np.diff(signs)
                record["n_direction_changes"] = int(np.count_nonzero(changes))
            else:
                record["n_direction_changes"] = 0

            records.append(record)

        if not records:
            logger.warning(
                "Nenhum registro gerado na agregacao de timeframe. "
                "Verifique se df_high_freq contem dados suficientes."
            )
            return pd.DataFrame(columns=["date"] + self._BASE_FEATURES)

        result = pd.DataFrame(records)
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        result = result.set_index("date").sort_index()

        logger.info(
            "Agregacao timeframe concluida: %d dias, %d features",
            len(result),
            len(result.columns),
        )
        return result

    # ------------------------------------------------------------------
    # Merge de multiplos timeframes
    # ------------------------------------------------------------------

    def merge_timeframes(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame | None = None,
        df_1h: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Combina features diarias com features agregadas de timeframes menores.

        Args:
            df_1d: DataFrame diario (base). Deve ter coluna ``timestamp``
                ou DatetimeIndex.
            df_4h: DataFrame OHLCV de 4h (opcional). Sera agregado para
                features diarias antes do merge.
            df_1h: DataFrame OHLCV de 1h (opcional). Idem.

        Returns:
            DataFrame diario com todas as features fundidas, indexado por data.
        """
        df = df_1d.copy()

        # Normalizar index para data
        if "timestamp" in df.columns:
            df["_merge_date"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
        elif isinstance(df.index, pd.DatetimeIndex):
            df["_merge_date"] = df.index.normalize()
        else:
            df["_merge_date"] = pd.to_datetime(df.index).normalize()

        tf_sources = {
            "4h": df_4h,
            "1h": df_1h,
        }

        for tf_label, tf_df in tf_sources.items():
            if tf_df is None or tf_df.empty:
                continue

            prefix = _TF_PREFIX.get(tf_label, f"tf{tf_label}")
            agg = self.aggregate_timeframe(tf_df, target_tf="1d")

            if agg.empty:
                logger.warning(
                    "Agregacao de %s retornou vazio; pulando merge.", tf_label
                )
                continue

            # Renomear colunas com prefixo do timeframe
            agg = agg.rename(
                columns={c: f"{prefix}_{c}" for c in agg.columns}
            )

            # Alinhar a chave de merge (date normalizado)
            agg["_merge_date"] = agg.index

            df = df.merge(agg, on="_merge_date", how="left")

            logger.info(
                "Merge %s: %d features adicionadas (%d linhas casadas de %d)",
                tf_label,
                len(agg.columns) - 1,  # descontar _merge_date
                df[f"{prefix}_{self._BASE_FEATURES[0]}"].notna().sum(),
                len(df),
            )

        # Limpar coluna auxiliar
        df = df.drop(columns=["_merge_date"], errors="ignore")

        # Preencher NaN das features de TF com 0 (dias sem dados intradiarios)
        tf_cols = [c for c in df.columns if c.startswith(("tf1h_", "tf4h_"))]
        if tf_cols:
            df[tf_cols] = df[tf_cols].fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # Utilitarios
    # ------------------------------------------------------------------

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna lista com todos os nomes de features geradas (ambos TFs).

        Returns:
            Lista de nomes no formato ``tf{TF}_{feature}``.
        """
        names: list[str] = []
        for tf_label, prefix in _TF_PREFIX.items():
            for feat in TimeframeFusion._BASE_FEATURES:
                names.append(f"{prefix}_{feat}")
        return names
