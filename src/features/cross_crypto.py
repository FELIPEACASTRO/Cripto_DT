"""Features cross-cryptocurrency: correlacoes e sinais entre moedas.

Baseado em estudo da SMU Singapore sobre previsibilidade cruzada
em mercados de criptomoedas.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrossCryptoFeatures:
    """Cria features cruzadas entre multiplas criptomoedas.

    Explora a estrutura de correlacao e lead-lag entre moedas para
    gerar features preditivas baseadas em retornos defasados e
    correlacoes rolling de outras moedas do universo.
    """

    def __init__(
        self,
        lag_periods: tuple[int, ...] = (1, 2),
        correlation_window: int = 20,
    ):
        self.lag_periods = lag_periods
        self.correlation_window = correlation_window

    def transform(
        self, coin_data: dict[str, pd.DataFrame], target_coin: str
    ) -> pd.DataFrame:
        """Adiciona features cruzadas ao DataFrame da moeda alvo.

        Args:
            coin_data: Dicionario {nome_moeda: DataFrame} com dados de
                todas as moedas. Cada DataFrame deve ter colunas
                'timestamp' e 'log_return' (ou 'close' para calcular).
            target_coin: Nome da moeda alvo no dicionario.

        Returns:
            DataFrame da moeda alvo com features cruzadas adicionadas.
        """
        # Validar que a moeda alvo existe no dicionario
        if target_coin not in coin_data:
            logger.error(
                "Moeda alvo '%s' nao encontrada no dicionario. "
                "Moedas disponiveis: %s",
                target_coin,
                list(coin_data.keys()),
            )
            raise ValueError(f"Moeda alvo '{target_coin}' nao encontrada.")

        df = coin_data[target_coin].copy()

        # Garantir que temos retornos para a moeda alvo
        target_returns = self._get_returns(df, target_coin)
        if target_returns is None:
            logger.warning("Nao foi possivel calcular retornos para '%s'.", target_coin)
            return df

        # Coletar moedas do universo (excluindo a alvo)
        other_coins = [c for c in coin_data.keys() if c != target_coin]

        if not other_coins:
            logger.warning("Nenhuma outra moeda disponivel para features cruzadas.")
            return df

        logger.info(
            "Gerando features cruzadas para '%s' usando %d moedas: %s",
            target_coin,
            len(other_coins),
            other_coins,
        )

        # --- Features por moeda individual ---
        lagged_returns_1 = {}  # Para calcular cross_leader_return depois
        correlations = {}

        for coin in other_coins:
            if coin not in coin_data:
                logger.debug("Moeda '%s' nao encontrada no dicionario, pulando.", coin)
                continue

            other_df = coin_data[coin]
            other_returns = self._get_returns(other_df, coin)

            if other_returns is None:
                logger.debug("Nao foi possivel obter retornos para '%s', pulando.", coin)
                continue

            # Alinhar indices pela coluna timestamp
            aligned = self._align_returns(df, target_returns, other_df, other_returns)
            if aligned is None:
                continue

            target_aligned, other_aligned = aligned

            # Retornos defasados da outra moeda
            for lag in self.lag_periods:
                col_name = f"{coin}_lag{lag}_return"
                df[col_name] = other_aligned.shift(lag).reindex(df.index)

            # Guardar lag1 para calculo do leader
            lagged_returns_1[coin] = other_aligned.shift(1).reindex(df.index)

            # Correlacao rolling com a moeda alvo
            corr_col = f"{coin}_corr_{self.correlation_window}"
            rolling_corr = target_aligned.rolling(self.correlation_window).corr(
                other_aligned
            )
            df[corr_col] = rolling_corr.reindex(df.index)
            correlations[coin] = df[corr_col]

        # --- Features agregadas ---
        df = self._add_aggregate_features(df, lagged_returns_1, correlations)

        return df

    def _get_returns(
        self, df: pd.DataFrame, coin_name: str
    ) -> pd.Series | None:
        """Obtem serie de retornos de um DataFrame.

        Tenta usar 'log_return' se disponivel, caso contrario calcula
        a partir de 'close'.

        Args:
            df: DataFrame da moeda.
            coin_name: Nome da moeda (para logging).

        Returns:
            Serie de retornos ou None se nao for possivel calcular.
        """
        if "log_return" in df.columns:
            return df["log_return"]
        elif "close" in df.columns:
            logger.debug(
                "Calculando log_return para '%s' a partir de close.", coin_name
            )
            return np.log(df["close"] / df["close"].shift(1))
        else:
            logger.warning(
                "Moeda '%s' nao tem colunas 'log_return' nem 'close'.", coin_name
            )
            return None

    def _align_returns(
        self,
        target_df: pd.DataFrame,
        target_returns: pd.Series,
        other_df: pd.DataFrame,
        other_returns: pd.Series,
    ) -> tuple[pd.Series, pd.Series] | None:
        """Alinha retornos de duas moedas pelo timestamp.

        Args:
            target_df: DataFrame da moeda alvo.
            target_returns: Retornos da moeda alvo.
            other_df: DataFrame da outra moeda.
            other_returns: Retornos da outra moeda.

        Returns:
            Tupla (target_aligned, other_aligned) ou None se impossivel alinhar.
        """
        if "timestamp" not in target_df.columns or "timestamp" not in other_df.columns:
            # Tentar usar o indice diretamente
            logger.debug("Usando indice numerico para alinhamento.")
            min_len = min(len(target_returns), len(other_returns))
            return (
                target_returns.iloc[:min_len].reset_index(drop=True),
                other_returns.iloc[:min_len].reset_index(drop=True),
            )

        # Criar series indexadas por timestamp
        target_ts = pd.Series(
            target_returns.values,
            index=pd.to_datetime(target_df["timestamp"].values),
        )
        other_ts = pd.Series(
            other_returns.values,
            index=pd.to_datetime(other_df["timestamp"].values),
        )

        # Alinhar e reindexar para o indice original do target
        combined = pd.DataFrame({"target": target_ts, "other": other_ts}).dropna()

        if len(combined) < self.correlation_window:
            logger.debug("Sobreposicao insuficiente apos alinhamento.")
            return None

        # Reindexar para o indice numerico do DataFrame original
        aligned_target = combined["target"]
        aligned_other = combined["other"]

        # Mapear de volta usando os indices do target_df
        target_mapped = pd.Series(np.nan, index=target_df.index)
        other_mapped = pd.Series(np.nan, index=target_df.index)

        # Criar mapeamento timestamp -> indice do target_df
        ts_to_idx = dict(
            zip(pd.to_datetime(target_df["timestamp"].values), target_df.index)
        )
        for ts in combined.index:
            if ts in ts_to_idx:
                idx = ts_to_idx[ts]
                target_mapped.loc[idx] = combined.loc[ts, "target"]
                other_mapped.loc[idx] = combined.loc[ts, "other"]

        return target_mapped, other_mapped

    def _add_aggregate_features(
        self,
        df: pd.DataFrame,
        lagged_returns: dict[str, pd.Series],
        correlations: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Adiciona features agregadas do universo de moedas.

        Args:
            df: DataFrame da moeda alvo.
            lagged_returns: Dicionario {moeda: retornos_lag1}.
            correlations: Dicionario {moeda: correlacao_rolling}.

        Returns:
            DataFrame com features agregadas adicionadas.
        """
        if not lagged_returns:
            logger.warning("Nenhum retorno defasado disponivel para features agregadas.")
            df["cross_mean_return"] = np.nan
            df["cross_dispersion"] = np.nan
            df["cross_leader_return"] = np.nan
            return df

        # Criar DataFrame com todos os retornos defasados
        lag_df = pd.DataFrame(lagged_returns)

        # Media dos retornos defasados de todas as moedas
        df["cross_mean_return"] = lag_df.mean(axis=1)

        # Dispersao: desvio padrao dos retornos (mede dispersao do mercado)
        df["cross_dispersion"] = lag_df.std(axis=1)

        # Retorno da moeda com maior correlacao (lider do mercado)
        if correlations:
            corr_df = pd.DataFrame(correlations)

            # Para cada linha, encontrar a moeda com maior correlacao absoluta
            # e usar o retorno defasado dela
            leader_returns = pd.Series(np.nan, index=df.index)

            for idx in df.index:
                # Obter correlacoes nesse ponto
                corr_values = corr_df.loc[idx] if idx in corr_df.index else pd.Series()
                corr_values = corr_values.dropna()

                if corr_values.empty:
                    continue

                # Moeda com maior correlacao absoluta
                leader_coin = corr_values.abs().idxmax()

                # Retorno defasado dessa moeda
                if leader_coin in lagged_returns:
                    lag_series = lagged_returns[leader_coin]
                    if idx in lag_series.index:
                        leader_returns.loc[idx] = lag_series.loc[idx]

            df["cross_leader_return"] = leader_returns
        else:
            df["cross_leader_return"] = np.nan

        return df

    def get_feature_names(self, other_coins: list[str] | None = None) -> list[str]:
        """Retorna nomes das features geradas.

        Args:
            other_coins: Lista de nomes das outras moedas. Se None,
                retorna apenas nomes das features agregadas.
        """
        names = []

        # Features por moeda
        if other_coins:
            for coin in other_coins:
                for lag in self.lag_periods:
                    names.append(f"{coin}_lag{lag}_return")
                names.append(f"{coin}_corr_{self.correlation_window}")

        # Features agregadas
        names.extend([
            "cross_mean_return",
            "cross_dispersion",
            "cross_leader_return",
        ])

        return names
