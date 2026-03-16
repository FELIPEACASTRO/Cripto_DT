"""Monitoramento de transacoes de baleias (baseado em estudo da SUTD Singapore)."""

import logging

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class WhaleMonitor:
    """Detecta e monitora atividade de baleias a partir de dados OHLCV.

    Gera features que indicam movimentacoes anomalas de volume,
    frequentemente associadas a grandes players (baleias).
    Funciona inteiramente com dados OHLCV — APIs externas sao opcionais.
    """

    # Limiares para deteccao de atividade de baleia
    WHALE_VOLUME_THRESHOLD = 2.5  # Multiplo da media para considerar "baleia"
    WHALE_SPIKE_THRESHOLD = 3.0   # Multiplo mais agressivo para spikes

    # APIs opcionais (usadas quando disponiveis)
    BLOCKCHAIN_API = "https://blockchain.info/q/24hrbtc"
    ETHERSCAN_API = "https://api.etherscan.io/api"

    def fetch_whale_data(
        self, coin: str, days: int = 30
    ) -> pd.DataFrame:
        """Busca dados de grandes transacoes (baleias) via APIs externas.

        Tenta buscar de APIs publicas (Blockchain.com para BTC,
        Etherscan para ETH). Como fallback, retorna DataFrame vazio
        e as features serao estimadas a partir dos dados OHLCV.

        Args:
            coin: Simbolo da moeda (ex: BTC, ETH)
            days: Numero de dias de historico

        Returns:
            DataFrame com dados de transacoes grandes (pode ser vazio)
        """
        coin_upper = coin.upper()

        # Tentar API do Blockchain.com para BTC
        if coin_upper == "BTC":
            return self._fetch_btc_whale_data()

        # Tentar Etherscan para ETH (requer API key)
        if coin_upper == "ETH":
            return self._fetch_eth_whale_data()

        # Outras moedas: sem API de baleias disponivel
        logger.info(
            f"Sem API de baleias para {coin} — features serao estimadas via OHLCV"
        )
        return pd.DataFrame()

    def _fetch_btc_whale_data(self) -> pd.DataFrame:
        """Tenta buscar volume BTC das ultimas 24h via Blockchain.com."""
        try:
            resp = requests.get(self.BLOCKCHAIN_API, timeout=15)
            resp.raise_for_status()
            volume_24h = float(resp.text)
            logger.info(f"Volume BTC 24h (Blockchain.com): {volume_24h:.2f}")
            return pd.DataFrame([{
                "source": "blockchain.com",
                "volume_24h": volume_24h,
            }])
        except Exception as e:
            logger.warning(f"Erro ao buscar dados BTC de baleias: {e}")
            return pd.DataFrame()

    def _fetch_eth_whale_data(self) -> pd.DataFrame:
        """Tenta buscar dados ETH via Etherscan (requer API key)."""
        # Etherscan requer API key — retornar vazio como fallback
        logger.info("Etherscan requer API key — usando estimativa OHLCV para ETH")
        return pd.DataFrame()

    def add_whale_features(
        self, df: pd.DataFrame, coin: str
    ) -> pd.DataFrame:
        """Adiciona features de atividade de baleias ao DataFrame.

        Todas as features sao calculadas a partir de dados OHLCV,
        garantindo funcionamento mesmo sem APIs externas.

        Features geradas:
            - whale_volume_ratio: volume do dia / media 30 dias
            - whale_activity: 1 se volume > 2.5x media, 0 caso contrario
            - whale_accumulation: soma de whale_activity nos ultimos 7 dias
            - whale_vol_zscore: z-score do volume vs rolling 30 dias
            - whale_large_tx_proxy: proxy de transacoes grandes via variacao
              anomala de volume intraday

        Args:
            df: DataFrame OHLCV com colunas 'volume', 'high', 'low', 'close'
            coin: Simbolo da moeda (usado para tentativa de API)

        Returns:
            DataFrame com features de baleias adicionadas
        """
        df = df.copy()

        # Verificar se coluna de volume existe
        if "volume" not in df.columns:
            logger.error("Coluna 'volume' nao encontrada — retornando sem features")
            return self._add_empty_features(df)

        volume = df["volume"].astype(float)

        # --- whale_volume_ratio ---
        # Razao entre volume do dia e media movel de 30 dias
        vol_ma30 = volume.rolling(window=30, min_periods=1).mean()
        df["whale_volume_ratio"] = (volume / vol_ma30.replace(0, np.nan)).fillna(1.0)

        # --- whale_activity ---
        # Indicador binario: volume > 2.5x a media de 30 dias
        df["whale_activity"] = (
            df["whale_volume_ratio"] > self.WHALE_VOLUME_THRESHOLD
        ).astype(float)

        # --- whale_accumulation ---
        # Soma de dias com atividade de baleia nos ultimos 7 dias
        df["whale_accumulation"] = (
            df["whale_activity"].rolling(window=7, min_periods=1).sum()
        )

        # --- whale_vol_zscore ---
        # Z-score do volume em relacao a janela de 30 dias
        vol_std30 = volume.rolling(window=30, min_periods=1).std()
        df["whale_vol_zscore"] = (
            (volume - vol_ma30) / vol_std30.replace(0, np.nan)
        ).fillna(0.0)

        # --- whale_large_tx_proxy ---
        # Proxy de transacoes grandes: combina variacao anomala de volume
        # com amplitude de preco (baleias movem preco E volume)
        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            # Amplitude relativa do dia (high-low / close)
            amplitude = ((high - low) / close.replace(0, np.nan)).fillna(0)
            amp_ma30 = amplitude.rolling(window=30, min_periods=1).mean()
            amp_ratio = (amplitude / amp_ma30.replace(0, np.nan)).fillna(1.0)

            # Proxy: produto do ratio de volume com ratio de amplitude
            # Valores altos indicam dias com volume E volatilidade anomalos
            df["whale_large_tx_proxy"] = (
                df["whale_volume_ratio"] * amp_ratio
            )

            # Normalizar com log para evitar valores extremos
            df["whale_large_tx_proxy"] = np.log1p(df["whale_large_tx_proxy"])
        else:
            # Sem dados de high/low/close — usar apenas volume
            logger.warning("Colunas high/low/close ausentes — proxy simplificado")
            df["whale_large_tx_proxy"] = np.log1p(df["whale_volume_ratio"])

        # Tentar enriquecer com dados de API (opcional)
        try:
            api_data = self.fetch_whale_data(coin)
            if not api_data.empty:
                logger.info(
                    f"Dados de baleias da API obtidos para {coin} "
                    f"({len(api_data)} registros)"
                )
        except Exception as e:
            logger.warning(f"API de baleias falhou (nao critico): {e}")

        logger.info(
            f"Features de baleias calculadas: {len(df)} linhas, "
            f"dias com atividade = {int(df['whale_activity'].sum())}"
        )

        return df

    def _add_empty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas de features de baleias com valores padrao."""
        df["whale_volume_ratio"] = 1.0
        df["whale_activity"] = 0.0
        df["whale_accumulation"] = 0.0
        df["whale_vol_zscore"] = 0.0
        df["whale_large_tx_proxy"] = 0.0
        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features de baleias."""
        return [
            "whale_volume_ratio",
            "whale_activity",
            "whale_accumulation",
            "whale_vol_zscore",
            "whale_large_tx_proxy",
        ]
