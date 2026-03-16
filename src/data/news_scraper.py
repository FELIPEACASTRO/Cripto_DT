"""Coleta de noticias de criptomoedas de multiplas fontes gratuitas."""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class NewsScraper:
    """Coleta noticias de criptomoedas de fontes gratuitas para alimentar pipeline de sentimento NLP."""

    CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    COINGECKO_STATUS_URL = "https://api.coingecko.com/api/v3/status_updates"
    REDDIT_SUBS = ["cryptocurrency", "bitcoin"]
    RSS_FEEDS = {
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "CoinTelegraph": "https://cointelegraph.com/rss",
    }

    # Mapeamento de simbolos de moedas para palavras-chave de busca
    COIN_KEYWORDS = {
        "BTC": ["bitcoin", "btc", "satoshi"],
        "ETH": ["ethereum", "eth", "ether", "vitalik"],
        "SOL": ["solana", "sol"],
        "BNB": ["binance", "bnb"],
        "XRP": ["ripple", "xrp"],
        "ADA": ["cardano", "ada"],
        "DOGE": ["dogecoin", "doge"],
        "DOT": ["polkadot", "dot"],
        "MATIC": ["polygon", "matic"],
        "AVAX": ["avalanche", "avax"],
        "LINK": ["chainlink", "link"],
        "LTC": ["litecoin", "ltc"],
    }

    # Palavras-chave para analise de sentimento simples
    POSITIVE_WORDS = [
        "bullish", "surge", "rally", "gain", "soar", "breakout", "pump",
        "moon", "profit", "growth", "adoption", "upgrade", "success",
        "record", "high", "boost", "recover", "positive", "optimistic",
    ]
    NEGATIVE_WORDS = [
        "bearish", "crash", "dump", "plunge", "drop", "fear", "sell",
        "ban", "hack", "scam", "fraud", "loss", "decline", "risk",
        "warning", "lawsuit", "regulation", "crackdown", "negative",
    ]

    def __init__(self, sources=None, max_articles=50, cache_hours=1):
        """Inicializa o scraper de noticias.

        Args:
            sources: Lista de fontes para buscar (None = todas).
                     Opcoes: 'cryptocompare', 'reddit', 'rss'
            max_articles: Numero maximo de artigos a retornar
            cache_hours: Horas para manter cache em memoria
        """
        self.sources = sources or ["cryptocompare", "reddit", "rss"]
        self.max_articles = max_articles
        self.cache_hours = cache_hours
        self._cache = {}  # {chave: (timestamp, dados)}

    def _is_cache_valid(self, key: str) -> bool:
        """Verifica se o cache para uma chave ainda e valido."""
        if key not in self._cache:
            return False
        cached_time, _ = self._cache[key]
        return (datetime.now(timezone.utc) - cached_time) < timedelta(hours=self.cache_hours)

    def _get_cached(self, key: str) -> list[dict] | None:
        """Retorna dados do cache se validos, senao None."""
        if self._is_cache_valid(key):
            _, data = self._cache[key]
            logger.debug(f"Cache hit para '{key}': {len(data)} artigos")
            return data
        return None

    def _set_cache(self, key: str, data: list[dict]) -> None:
        """Armazena dados no cache com timestamp atual."""
        self._cache[key] = (datetime.now(timezone.utc), data)

    def fetch_all_news(self, coin: str = "BTC") -> list[dict]:
        """Busca noticias de todas as fontes configuradas.

        Args:
            coin: Simbolo da moeda para filtrar relevancia (ex: BTC, ETH)

        Returns:
            Lista de dicts com chaves: title, text, source, timestamp, coin_relevance
        """
        all_articles = []

        if "cryptocompare" in self.sources:
            try:
                articles = self._fetch_cryptocompare()
                all_articles.extend(articles)
                logger.info(f"CryptoCompare: {len(articles)} artigos coletados")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Erro ao buscar CryptoCompare: {e}")

        if "reddit" in self.sources:
            for subreddit in self.REDDIT_SUBS:
                try:
                    articles = self._fetch_reddit(subreddit)
                    all_articles.extend(articles)
                    logger.info(f"Reddit r/{subreddit}: {len(articles)} posts coletados")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Erro ao buscar Reddit r/{subreddit}: {e}")

        if "rss" in self.sources:
            for source_name, url in self.RSS_FEEDS.items():
                try:
                    articles = self._fetch_rss(url, source_name)
                    all_articles.extend(articles)
                    logger.info(f"RSS {source_name}: {len(articles)} artigos coletados")
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Erro ao buscar RSS {source_name}: {e}")

        # Filtrar por relevancia da moeda
        filtered = self._filter_by_coin(all_articles, coin)

        # Ordenar por timestamp (mais recente primeiro) e limitar
        filtered.sort(key=lambda x: x.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        filtered = filtered[: self.max_articles]

        logger.info(
            f"Total de noticias: {len(all_articles)} coletadas, "
            f"{len(filtered)} relevantes para {coin}"
        )
        return filtered

    def _fetch_cryptocompare(self) -> list[dict]:
        """Busca noticias da API gratuita do CryptoCompare.

        Returns:
            Lista de artigos normalizados
        """
        cached = self._get_cached("cryptocompare")
        if cached is not None:
            return cached

        try:
            resp = requests.get(self.CRYPTOCOMPARE_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("Data", [])
        except requests.RequestException as e:
            logger.error(f"Erro na requisicao CryptoCompare: {e}")
            return []

        articles = []
        for item in data:
            articles.append({
                "title": item.get("title", ""),
                "text": item.get("body", ""),
                "source": f"CryptoCompare/{item.get('source', 'unknown')}",
                "timestamp": datetime.fromtimestamp(
                    item.get("published_on", 0), tz=timezone.utc
                ),
                "coin_relevance": 0.0,
            })

        self._set_cache("cryptocompare", articles)
        return articles

    def _fetch_reddit(self, subreddit: str) -> list[dict]:
        """Busca posts do Reddit via API publica JSON.

        Args:
            subreddit: Nome do subreddit (sem r/)

        Returns:
            Lista de artigos normalizados
        """
        cache_key = f"reddit_{subreddit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        url = f"https://www.reddit.com/r/{subreddit}/hot.json"
        headers = {"User-Agent": "CriptoDT/1.0 (news scraper)"}

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Erro na requisicao Reddit r/{subreddit}: {e}")
            return []

        articles = []
        posts = data.get("data", {}).get("children", [])
        for post in posts:
            post_data = post.get("data", {})
            # Ignorar posts fixados e stickied
            if post_data.get("stickied", False):
                continue
            articles.append({
                "title": post_data.get("title", ""),
                "text": post_data.get("selftext", ""),
                "source": f"Reddit/r/{subreddit}",
                "timestamp": datetime.fromtimestamp(
                    post_data.get("created_utc", 0), tz=timezone.utc
                ),
                "coin_relevance": 0.0,
            })

        self._set_cache(cache_key, articles)
        return articles

    def _fetch_rss(self, url: str, source_name: str) -> list[dict]:
        """Busca e parseia feed RSS usando xml.etree.ElementTree.

        Args:
            url: URL do feed RSS
            source_name: Nome da fonte para identificacao

        Returns:
            Lista de artigos normalizados
        """
        cache_key = f"rss_{source_name}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Erro ao buscar RSS {source_name}: {e}")
            return []

        articles = []
        try:
            root = ET.fromstring(resp.content)

            # Suporte para RSS 2.0 e Atom
            # Tentar RSS 2.0 primeiro (channel/item)
            items = root.findall(".//item")

            # Se nao encontrar, tentar Atom (entry)
            if not items:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                items = root.findall(".//atom:entry", ns)

            for item in items:
                title = self._get_xml_text(item, "title") or ""
                description = (
                    self._get_xml_text(item, "description")
                    or self._get_xml_text(item, "summary")
                    or ""
                )
                pub_date_str = (
                    self._get_xml_text(item, "pubDate")
                    or self._get_xml_text(item, "published")
                    or ""
                )

                timestamp = self._parse_rss_date(pub_date_str)

                articles.append({
                    "title": title,
                    "text": description,
                    "source": source_name,
                    "timestamp": timestamp,
                    "coin_relevance": 0.0,
                })

        except ET.ParseError as e:
            logger.error(f"Erro ao parsear XML do RSS {source_name}: {e}")
            return []

        self._set_cache(cache_key, articles)
        return articles

    @staticmethod
    def _get_xml_text(element, tag: str) -> str | None:
        """Extrai texto de um sub-elemento XML de forma segura."""
        child = element.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None

    @staticmethod
    def _parse_rss_date(date_str: str) -> datetime:
        """Tenta parsear data de feed RSS em varios formatos.

        Args:
            date_str: String de data do feed

        Returns:
            datetime com timezone UTC
        """
        if not date_str:
            return datetime.now(timezone.utc)

        formats = [
            "%a, %d %b %Y %H:%M:%S %z",   # RFC 822 (RSS 2.0)
            "%a, %d %b %Y %H:%M:%S %Z",   # RFC 822 com timezone nome
            "%Y-%m-%dT%H:%M:%S%z",          # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",           # ISO 8601 UTC
            "%Y-%m-%d %H:%M:%S",            # Formato simples
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        logger.debug(f"Nao foi possivel parsear data RSS: {date_str}")
        return datetime.now(timezone.utc)

    def _filter_by_coin(self, articles: list[dict], coin: str) -> list[dict]:
        """Filtra artigos relevantes para uma moeda especifica usando palavras-chave.

        Args:
            articles: Lista de artigos para filtrar
            coin: Simbolo da moeda (ex: BTC, ETH)

        Returns:
            Lista de artigos com coin_relevance atualizada (> 0)
        """
        coin = coin.upper()
        keywords = self.COIN_KEYWORDS.get(coin, [coin.lower()])

        filtered = []
        for article in articles:
            text_lower = (
                article.get("title", "") + " " + article.get("text", "")
            ).lower()

            # Calcular relevancia baseada em ocorrencias de palavras-chave
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                relevance = min(matches / len(keywords), 1.0)
                article = article.copy()
                article["coin_relevance"] = relevance
                filtered.append(article)

        return filtered

    def _compute_sentiment(self, text: str) -> float:
        """Calcula sentimento simples baseado em palavras-chave.

        Args:
            text: Texto para analisar

        Returns:
            Score de sentimento entre -1.0 (negativo) e 1.0 (positivo)
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        pos_count = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)
        neg_count = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total

    def add_news_features(self, df: pd.DataFrame, coin: str = "BTC") -> pd.DataFrame:
        """Adiciona features agregadas de noticias ao DataFrame.

        Args:
            df: DataFrame OHLCV com coluna 'timestamp'
            coin: Simbolo da moeda para filtrar noticias

        Returns:
            DataFrame com features de noticias adicionadas:
            news_count_24h, news_sentiment_mean, news_sentiment_std, news_buzz_ratio
        """
        df = df.copy()

        try:
            articles = self.fetch_all_news(coin=coin)
        except Exception as e:
            logger.error(f"Erro ao buscar noticias para features: {e}")
            for feat in self.get_feature_names():
                df[feat] = np.nan
            return df

        if not articles:
            logger.warning("Nenhuma noticia encontrada, retornando features como NaN")
            for feat in self.get_feature_names():
                df[feat] = np.nan
            return df

        # Calcular sentimento para cada artigo
        for article in articles:
            article["sentiment"] = self._compute_sentiment(
                article.get("title", "") + " " + article.get("text", "")
            )

        # Converter artigos para DataFrame auxiliar
        news_df = pd.DataFrame(articles)
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], utc=True)
        news_df["_date"] = news_df["timestamp"].dt.normalize()

        # Agregar por dia
        daily_news = news_df.groupby("_date").agg(
            news_count=("sentiment", "count"),
            sentiment_mean=("sentiment", "mean"),
            sentiment_std=("sentiment", "std"),
        ).reset_index()
        daily_news["sentiment_std"] = daily_news["sentiment_std"].fillna(0.0)

        # Calcular media movel de 7 dias para buzz ratio
        daily_news = daily_news.sort_values("_date").reset_index(drop=True)
        daily_news["news_count_ma7"] = (
            daily_news["news_count"].rolling(7, min_periods=1).mean()
        )
        daily_news["buzz_ratio"] = (
            daily_news["news_count"]
            / daily_news["news_count_ma7"].replace(0, np.nan)
        )

        # Merge com DataFrame principal
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()

        merge_cols = daily_news[["_date", "news_count", "sentiment_mean", "sentiment_std", "buzz_ratio"]]
        merge_cols = merge_cols.drop_duplicates(subset=["_date"])

        df = df.merge(merge_cols, on="_date", how="left")

        # Renomear para nomes padrao das features
        df = df.rename(columns={
            "news_count": "news_count_24h",
            "sentiment_mean": "news_sentiment_mean",
            "sentiment_std": "news_sentiment_std",
            "buzz_ratio": "news_buzz_ratio",
        })

        # Preencher NaN com valores neutros
        df["news_count_24h"] = df["news_count_24h"].fillna(0)
        df["news_sentiment_mean"] = df["news_sentiment_mean"].fillna(0.0)
        df["news_sentiment_std"] = df["news_sentiment_std"].fillna(0.0)
        df["news_buzz_ratio"] = df["news_buzz_ratio"].fillna(1.0)

        df = df.drop(columns=["_date"])

        logger.info("Features de noticias adicionadas com sucesso")
        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features de noticias geradas."""
        return [
            "news_count_24h",
            "news_sentiment_mean",
            "news_sentiment_std",
            "news_buzz_ratio",
        ]
