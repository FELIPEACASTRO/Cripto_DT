"""Coleta de dados OHLCV via CCXT (Binance) e CoinGecko."""

import time
import logging
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import requests

from config.settings import Config, config as default_config

logger = logging.getLogger(__name__)


class DataCollector:
    """Coleta dados historicos de criptomoedas."""

    # Mapeamento de simbolo para CoinGecko ID
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

    def __init__(self, config: Config = default_config):
        self.config = config
        self.exchange = getattr(ccxt, config.data.exchange)({
            "enableRateLimit": True,
        })

    def fetch_ohlcv(
        self, coin: str, timeframe: str, since_days: int | None = None
    ) -> pd.DataFrame:
        """Busca dados OHLCV do exchange via CCXT.

        Args:
            coin: Simbolo da moeda (ex: "BTC")
            timeframe: Intervalo (ex: "1h", "4h", "1d")
            since_days: Dias de historico. Se None, usa config.

        Returns:
            DataFrame com colunas [timestamp, open, high, low, close, volume]
        """
        if since_days is None:
            since_days = self.config.data.history_days

        symbol = f"{coin}/{self.config.data.quote_currency}"
        since_ms = int(
            (datetime.utcnow() - timedelta(days=since_days)).timestamp() * 1000
        )

        logger.info(f"Coletando {symbol} {timeframe} ({since_days} dias)...")

        all_candles = []
        limit = 1000  # max por request na Binance

        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=limit
                )
            except ccxt.BaseError as e:
                logger.error(f"Erro ao buscar {symbol}: {e}")
                break

            if not candles:
                break

            all_candles.extend(candles)
            since_ms = candles[-1][0] + 1  # proximo ms apos ultimo candle

            if len(candles) < limit:
                break

            time.sleep(self.exchange.rateLimit / 1000)

        if not all_candles:
            logger.warning(f"Nenhum dado retornado para {symbol} {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df.reset_index(drop=True)

        logger.info(f"  {symbol} {timeframe}: {len(df)} candles coletados")
        return df

    def fetch_market_data(self, coin: str) -> dict:
        """Busca dados de mercado do CoinGecko (market cap, supply, etc).

        Args:
            coin: Simbolo da moeda (ex: "BTC")

        Returns:
            Dicionario com dados de mercado
        """
        cg_id = self.COINGECKO_IDS.get(coin)
        if not cg_id:
            logger.warning(f"CoinGecko ID nao encontrado para {coin}")
            return {}

        url = f"{self.config.data.coingecko_base_url}/coins/{cg_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            time.sleep(self.config.data.rate_limit_sleep)

            return {
                "market_cap": data.get("market_data", {}).get("market_cap", {}).get("usd"),
                "total_volume": data.get("market_data", {}).get("total_volume", {}).get("usd"),
                "circulating_supply": data.get("market_data", {}).get("circulating_supply"),
                "market_cap_rank": data.get("market_cap_rank"),
            }
        except requests.RequestException as e:
            logger.error(f"Erro CoinGecko para {coin}: {e}")
            return {}

    def collect_all(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Coleta dados OHLCV para todas as moedas e timeframes configurados.

        Returns:
            Dict aninhado: {coin: {timeframe: DataFrame}}
        """
        result = {}
        for coin in self.config.data.coins:
            result[coin] = {}
            for tf in self.config.data.timeframes:
                df = self.fetch_ohlcv(coin, tf)
                if not df.empty:
                    result[coin][tf] = df
        return result
