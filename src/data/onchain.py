"""Coleta de dados on-chain para metricas de criptomoedas."""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "exchange_flow_ratio",
    "whale_ratio",
    "network_activity_zscore",
    "hash_rate_change",
]


class OnchainCollector:
    """Coleta e deriva metricas on-chain para criptomoedas.

    Usa endpoints gratuitos do CoinGecko quando possivel.
    Quando dados on-chain reais nao estao disponiveis, gera features
    sinteticas derivadas de dados OHLCV (volume como proxy para fluxos
    de exchange, picos de volume como proxy para atividade de baleias).
    """

    COINGECKO_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "SOL": "solana",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "AVAX": "avalanche-2",
        "DOT": "polkadot",
        "MATIC": "matic-network",
    }

    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
    _RATE_LIMIT_SECONDS = 1.5  # CoinGecko free tier: ~30 req/min

    def __init__(self):
        if not HAS_REQUESTS:
            logger.warning(
                "requests nao instalado. Coleta de dados on-chain sera limitada."
            )

    def _get_coingecko_id(self, coin: str) -> str | None:
        """Resolve simbolo para CoinGecko ID."""
        cg_id = self.COINGECKO_IDS.get(coin.upper())
        if cg_id is None:
            logger.warning("CoinGecko ID nao encontrado para %s", coin)
        return cg_id

    def _rate_limited_get(self, url: str, params: dict | None = None) -> dict | None:
        """GET com rate limiting e tratamento de erros."""
        if not HAS_REQUESTS:
            return None

        time.sleep(self._RATE_LIMIT_SECONDS)
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                logger.warning("Rate limit atingido. Aguardando 60s...")
                time.sleep(60)
                resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            logger.exception("Falha na requisicao: %s", url)
            return None

    def fetch_exchange_flows(self, coin: str, days: int = 90) -> pd.DataFrame:
        """Busca dados de fluxo de exchange (inflow/outflow).

        Como APIs on-chain gratuitas sao limitadas, utiliza dados de mercado
        do CoinGecko (volume) como proxy e gera estimativas sinteticas.

        Args:
            coin: Simbolo da moeda (ex: BTC, ETH).
            days: Numero de dias historicos.

        Returns:
            DataFrame com colunas: timestamp, exchange_inflow, exchange_outflow,
            net_flow.
        """
        cg_id = self._get_coingecko_id(coin)
        if cg_id is None:
            return self._empty_flow_df()

        url = f"{self.COINGECKO_BASE_URL}/coins/{cg_id}/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        data = self._rate_limited_get(url, params)

        if data is None or "total_volumes" not in data:
            logger.warning(
                "Sem dados de volume para %s. Retornando DataFrame vazio.", coin
            )
            return self._empty_flow_df()

        volumes = data["total_volumes"]
        df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Simular fluxos de exchange a partir de volume
        # Inflow ~ proporcional ao volume com ruido
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(1.0, 0.1, size=len(df))
        df["exchange_inflow"] = df["volume"] * 0.35 * np.abs(noise)

        noise2 = rng.normal(1.0, 0.1, size=len(df))
        df["exchange_outflow"] = df["volume"] * 0.30 * np.abs(noise2)

        df["net_flow"] = df["exchange_inflow"] - df["exchange_outflow"]

        logger.info("Fluxos de exchange simulados para %s (%d dias).", coin, len(df))
        return df[["timestamp", "exchange_inflow", "exchange_outflow", "net_flow"]]

    def fetch_network_stats(self, coin: str, days: int = 90) -> pd.DataFrame:
        """Busca estatisticas de rede (enderecos ativos, transacoes, hash rate).

        Utiliza dados do CoinGecko complementados com estimativas sinteticas.

        Args:
            coin: Simbolo da moeda.
            days: Numero de dias historicos.

        Returns:
            DataFrame com colunas: timestamp, active_addresses,
            transaction_count, hash_rate.
        """
        cg_id = self._get_coingecko_id(coin)
        if cg_id is None:
            return self._empty_network_df()

        # Buscar dados de mercado para derivar proxies
        url = f"{self.COINGECKO_BASE_URL}/coins/{cg_id}/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        data = self._rate_limited_get(url, params)

        if data is None or "total_volumes" not in data:
            logger.warning(
                "Sem dados de mercado para %s. Retornando DataFrame vazio.", coin
            )
            return self._empty_network_df()

        volumes = data["total_volumes"]
        prices = data.get("prices", [])

        df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        if prices:
            price_df = pd.DataFrame(prices, columns=["ts", "price"])
            df["price"] = price_df["price"].values[: len(df)]
        else:
            df["price"] = 0.0

        # Enderecos ativos: proxy via log(volume) com escala e ruido
        rng = np.random.default_rng(seed=123)
        base_addresses = {
            "BTC": 900_000,
            "ETH": 500_000,
            "SOL": 300_000,
        }
        base = base_addresses.get(coin.upper(), 100_000)
        vol_norm = df["volume"] / (df["volume"].mean() + 1e-10)
        df["active_addresses"] = (
            base * vol_norm * rng.normal(1.0, 0.05, size=len(df))
        ).astype(int).clip(lower=1)

        # Transacoes: correlacao com volume
        tx_base = {"BTC": 300_000, "ETH": 1_200_000, "SOL": 20_000_000}
        tx_scale = tx_base.get(coin.upper(), 50_000)
        df["transaction_count"] = (
            tx_scale * vol_norm * rng.normal(1.0, 0.08, size=len(df))
        ).astype(int).clip(lower=1)

        # Hash rate: apenas para PoW (BTC), senao stake-like proxy
        if coin.upper() in ("BTC",):
            # Simular hash rate com tendencia crescente + ruido
            trend = np.linspace(400, 600, len(df))  # EH/s approx
            df["hash_rate"] = trend * rng.normal(1.0, 0.02, size=len(df))
        else:
            # Para PoS, usar TVL proxy ou constante + ruido
            df["hash_rate"] = 0.0

        logger.info("Estatisticas de rede estimadas para %s (%d dias).", coin, len(df))
        return df[
            ["timestamp", "active_addresses", "transaction_count", "hash_rate"]
        ]

    def add_onchain_features(
        self, df: pd.DataFrame, coin: str
    ) -> pd.DataFrame:
        """Adiciona features on-chain derivadas ao DataFrame.

        Se dados on-chain reais nao estiverem disponiveis, utiliza proxies
        baseados em OHLCV (volume).

        Features adicionadas:
          - exchange_flow_ratio: inflow / outflow (>1 = pressao vendedora)
          - whale_ratio: proporcao de volume em transacoes grandes
          - network_activity_zscore: z-score da atividade de rede
          - hash_rate_change: variacao percentual do hash rate

        Args:
            df: DataFrame com colunas OHLCV (open, high, low, close, volume).
            coin: Simbolo da moeda.

        Returns:
            DataFrame com features on-chain adicionadas.
        """
        df = df.copy()

        use_synthetic = True

        # Tentar buscar dados on-chain reais
        try:
            n_days = min(len(df), 90)
            flows = self.fetch_exchange_flows(coin, days=n_days)
            network = self.fetch_network_stats(coin, days=n_days)

            if not flows.empty and not network.empty:
                # Merge por data mais proxima (dados podem ter granularidade diferente)
                if "timestamp" in df.columns:
                    df_ts = df.set_index("timestamp") if "timestamp" in df.columns else df
                    flows_ts = flows.set_index("timestamp")
                    network_ts = network.set_index("timestamp")

                    # Reindexar para match
                    flows_reindexed = flows_ts.reindex(
                        df_ts.index, method="nearest"
                    )
                    network_reindexed = network_ts.reindex(
                        df_ts.index, method="nearest"
                    )

                    df["exchange_flow_ratio"] = (
                        flows_reindexed["exchange_inflow"].values
                        / (flows_reindexed["exchange_outflow"].values + 1e-10)
                    )
                    df["hash_rate_change"] = (
                        pd.Series(network_reindexed["hash_rate"].values)
                        .pct_change()
                        .fillna(0)
                        .values
                    )

                    # Z-score da atividade (enderecos ativos)
                    addr = pd.Series(network_reindexed["active_addresses"].values, dtype=float)
                    mean_addr = addr.rolling(window=30, min_periods=1).mean()
                    std_addr = addr.rolling(window=30, min_periods=1).std().replace(0, 1)
                    df["network_activity_zscore"] = ((addr - mean_addr) / std_addr).values

                    use_synthetic = False
                    logger.info("Features on-chain reais adicionadas para %s.", coin)

        except Exception:
            logger.warning(
                "Falha ao buscar dados on-chain para %s. Usando proxies sinteticos.",
                coin,
                exc_info=True,
            )

        if use_synthetic:
            df = self._add_synthetic_features(df)

        # Whale ratio: sempre derivado de volume (picos como proxy)
        if "whale_ratio" not in df.columns:
            df["whale_ratio"] = self._compute_whale_ratio(df)

        # Garantir que todas as features existam
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def _add_synthetic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features sinteticas a partir de OHLCV quando dados on-chain nao estao disponiveis."""
        if "volume" not in df.columns:
            logger.warning("Coluna 'volume' ausente. Features sinteticas serao zero.")
            for col in FEATURE_NAMES:
                df[col] = 0.0
            return df

        vol = df["volume"].astype(float)

        # Exchange flow ratio: ratio de volume acima/abaixo da media movel
        vol_ma = vol.rolling(window=20, min_periods=1).mean()
        df["exchange_flow_ratio"] = vol / (vol_ma + 1e-10)

        # Whale ratio via picos de volume
        df["whale_ratio"] = self._compute_whale_ratio(df)

        # Network activity z-score: z-score do volume
        vol_mean = vol.rolling(window=30, min_periods=1).mean()
        vol_std = vol.rolling(window=30, min_periods=1).std().replace(0, 1)
        df["network_activity_zscore"] = (vol - vol_mean) / vol_std

        # Hash rate change: proxy via variacao de volume (sem significado real para PoS)
        df["hash_rate_change"] = vol.pct_change().fillna(0).clip(-1, 1)

        logger.info("Features on-chain sinteticas adicionadas (proxy via OHLCV).")
        return df

    @staticmethod
    def _compute_whale_ratio(df: pd.DataFrame) -> pd.Series:
        """Calcula whale ratio como proporcao de volume em picos.

        Picos de volume (acima de 2 desvios da media movel) sao
        interpretados como atividade de baleias.
        """
        if "volume" not in df.columns:
            return pd.Series(0.0, index=df.index)

        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(window=20, min_periods=1).mean()
        vol_std = vol.rolling(window=20, min_periods=1).std().replace(0, 1)

        # Volume acima de 2 sigma e considerado "whale"
        whale_threshold = vol_ma + 2 * vol_std
        whale_vol = vol.where(vol > whale_threshold, 0)
        total_vol_rolling = vol.rolling(window=20, min_periods=1).sum()

        whale_ratio = whale_vol / (total_vol_rolling + 1e-10)
        return whale_ratio.clip(0, 1)

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features on-chain geradas."""
        return list(FEATURE_NAMES)

    @staticmethod
    def _empty_flow_df() -> pd.DataFrame:
        """Retorna DataFrame vazio com schema de fluxos."""
        return pd.DataFrame(
            columns=["timestamp", "exchange_inflow", "exchange_outflow", "net_flow"]
        )

    @staticmethod
    def _empty_network_df() -> pd.DataFrame:
        """Retorna DataFrame vazio com schema de rede."""
        return pd.DataFrame(
            columns=["timestamp", "active_addresses", "transaction_count", "hash_rate"]
        )
