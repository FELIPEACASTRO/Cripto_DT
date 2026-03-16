"""Crawler avancado de noticias cripto inspirado em padroes Crawl4AI.

Complementa o news_scraper.py existente com requisicoes assincronas,
logica de retry com backoff exponencial, rate limiting por dominio,
rotacao de user agents e deduplicacao de conteudo.
"""

import asyncio
import hashlib
import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests

# Importacao graceful do aiohttp - fallback para requests sincrono
try:
    import aiohttp

    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constantes e configuracoes
# ──────────────────────────────────────────────────────────────────────

# Pool de User-Agents para rotacao anti-bloqueio
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# Mapeamento de simbolos de moedas para palavras-chave de busca
_COIN_KEYWORDS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "satoshi", "sats"],
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

# Listas de palavras-chave de sentimento (positivo / negativo)
_POSITIVE_WORDS = [
    "bullish", "surge", "rally", "gain", "soar", "breakout", "pump",
    "moon", "profit", "growth", "adoption", "upgrade", "success",
    "record", "high", "boost", "recover", "positive", "optimistic",
    "accumulate", "institutional", "etf", "approval", "launch",
]
_NEGATIVE_WORDS = [
    "bearish", "crash", "dump", "plunge", "drop", "fear", "sell",
    "ban", "hack", "scam", "fraud", "loss", "decline", "risk",
    "warning", "lawsuit", "regulation", "crackdown", "negative",
    "liquidation", "exploit", "rug", "ponzi", "sec",
]


# ──────────────────────────────────────────────────────────────────────
# Utilitarios internos
# ──────────────────────────────────────────────────────────────────────

def _rotate_ua() -> str:
    """Retorna um User-Agent aleatorio do pool (deterministico por segundo para testes)."""
    idx = int(time.time()) % len(_USER_AGENTS)
    return _USER_AGENTS[idx]


def _clean_html(raw: str) -> str:
    """Remove tags HTML e decodifica entidades, retornando texto limpo."""
    if not raw:
        return ""
    # Decodificar entidades HTML
    text = html.unescape(raw)
    # Remover tags HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalizar espacos em branco
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _title_hash(title: str) -> str:
    """Gera hash de titulo normalizado para deduplicacao."""
    normalized = re.sub(r"[^a-z0-9]", "", title.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def _compute_sentiment(text: str) -> float:
    """Calcula sentimento simples baseado em palavras-chave.

    Retorna score entre -1.0 (muito negativo) e 1.0 (muito positivo).
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _extract_sentiment_keywords(text: str) -> list[str]:
    """Extrai palavras-chave de sentimento encontradas no texto."""
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for w in _POSITIVE_WORDS:
        if w in text_lower:
            found.append(f"+{w}")
    for w in _NEGATIVE_WORDS:
        if w in text_lower:
            found.append(f"-{w}")
    return found


def _parse_rss_date(date_str: str) -> datetime:
    """Tenta parsear data de feed RSS em varios formatos comuns."""
    if not date_str:
        return datetime.now(timezone.utc)

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    logger.debug(f"Nao foi possivel parsear data: {date_str}")
    return datetime.now(timezone.utc)


# ──────────────────────────────────────────────────────────────────────
# Rate limiter simples por dominio (em memoria)
# ──────────────────────────────────────────────────────────────────────

class _DomainRateLimiter:
    """Controla intervalo minimo entre requisicoes para cada dominio."""

    def __init__(self, default_delay: float = 1.5):
        # Mapa dominio -> timestamp da ultima requisicao
        self._last_request: dict[str, float] = {}
        self._default_delay = default_delay
        # Delays customizados por dominio (segundos)
        self._domain_delays: dict[str, float] = {
            "reddit.com": 2.0,
            "api.coingecko.com": 1.5,
            "min-api.cryptocompare.com": 1.0,
            "alternative.me": 1.0,
        }

    def _get_delay(self, domain: str) -> float:
        """Retorna delay configurado para o dominio."""
        for key, delay in self._domain_delays.items():
            if key in domain:
                return delay
        return self._default_delay

    def wait_sync(self, domain: str) -> None:
        """Espera o tempo necessario antes de fazer requisicao (sincrono)."""
        delay = self._get_delay(domain)
        last = self._last_request.get(domain, 0.0)
        elapsed = time.time() - last
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request[domain] = time.time()

    async def wait_async(self, domain: str) -> None:
        """Espera o tempo necessario antes de fazer requisicao (assincrono)."""
        delay = self._get_delay(domain)
        last = self._last_request.get(domain, 0.0)
        elapsed = time.time() - last
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_request[domain] = time.time()


# ──────────────────────────────────────────────────────────────────────
# Cache em memoria com TTL
# ──────────────────────────────────────────────────────────────────────

class _TTLCache:
    """Cache simples em memoria com tempo de expiracao."""

    def __init__(self, ttl_hours: float = 1.0):
        self._ttl = timedelta(hours=ttl_hours)
        self._store: dict[str, tuple[datetime, Any]] = {}

    def get(self, key: str) -> Any | None:
        """Retorna valor do cache se ainda valido, senao None."""
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if (datetime.now(timezone.utc) - ts) > self._ttl:
            del self._store[key]
            return None
        logger.debug(f"Cache hit: {key}")
        return value

    def set(self, key: str, value: Any) -> None:
        """Armazena valor no cache com timestamp atual."""
        self._store[key] = (datetime.now(timezone.utc), value)

    def clear(self) -> None:
        """Limpa todo o cache."""
        self._store.clear()


# ──────────────────────────────────────────────────────────────────────
# Classe principal: AdvancedNewsCrawler
# ──────────────────────────────────────────────────────────────────────

class AdvancedNewsCrawler:
    """Crawler avancado de noticias cripto com suporte assincrono,
    retry com backoff exponencial, rate limiting e deduplicacao.

    Pode ser usado como substituto ou complemento ao NewsScraper existente.
    """

    # Endpoints de fontes de dados
    CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    COINGECKO_NEWS_URL = "https://api.coingecko.com/api/v3/status_updates"
    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"

    REDDIT_SUBS = ["cryptocurrency", "bitcoin", "ethtrader", "solana"]

    RSS_FEEDS: dict[str, str] = {
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "Decrypt": "https://decrypt.co/feed",
        "TheBlock": "https://www.theblock.co/rss.xml",
    }

    def __init__(
        self,
        max_articles: int = 100,
        cache_ttl_hours: float = 1.0,
        max_retries: int = 3,
    ):
        """Inicializa o crawler avancado.

        Args:
            max_articles: Numero maximo de artigos a retornar apos deduplicacao
            cache_ttl_hours: Tempo de vida do cache em horas
            max_retries: Numero maximo de tentativas por requisicao
        """
        self.max_articles = max_articles
        self.max_retries = max_retries
        self._cache = _TTLCache(ttl_hours=cache_ttl_hours)
        self._rate_limiter = _DomainRateLimiter()
        # Conjunto de hashes de titulos para deduplicacao
        self._seen_hashes: set[str] = set()

    # ──────────────────────────────────────────────────────────────
    # Metodos publicos
    # ──────────────────────────────────────────────────────────────

    def fetch_all(self, coin: str = "BTC") -> list[dict]:
        """Busca noticias de todas as fontes (wrapper sincrono).

        Se aiohttp estiver disponivel, executa a versao assincrona
        internamente. Caso contrario, usa requests de forma sincrona.

        Args:
            coin: Simbolo da moeda (ex: BTC, ETH, SOL)

        Returns:
            Lista de dicts com chaves:
                title, text, source, timestamp, coin_relevance, sentiment_keywords
        """
        if _HAS_AIOHTTP:
            # Verificar se ja existe um event loop rodando
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                # Ja dentro de um loop async - criar tarefa futura
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.fetch_all_async(coin))
                    return future.result(timeout=120)
            else:
                return asyncio.run(self.fetch_all_async(coin))
        else:
            logger.info("aiohttp nao disponivel, usando fallback sincrono")
            return self._fetch_all_sync(coin)

    async def fetch_all_async(self, coin: str = "BTC") -> list[dict]:
        """Busca noticias de todas as fontes de forma assincrona.

        Executa todas as fontes em paralelo usando aiohttp.

        Args:
            coin: Simbolo da moeda

        Returns:
            Lista de artigos normalizados e deduplicados
        """
        if not _HAS_AIOHTTP:
            logger.warning("aiohttp nao disponivel, delegando para fallback sincrono")
            return self._fetch_all_sync(coin)

        # Verificar cache global
        cache_key = f"all_news_{coin}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        all_articles: list[dict] = []
        self._seen_hashes.clear()

        # Criar sessao aiohttp com timeout global
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Preparar tarefas para todas as fontes
            tasks = []

            # CryptoCompare
            tasks.append(self._async_fetch_cryptocompare(session))

            # CoinGecko
            tasks.append(self._async_fetch_coingecko(session))

            # Reddit - uma tarefa por subreddit
            for sub in self.REDDIT_SUBS:
                tasks.append(self._async_fetch_reddit(session, sub))

            # RSS feeds - uma tarefa por feed
            for name, url in self.RSS_FEEDS.items():
                tasks.append(self._async_fetch_rss(session, url, name))

            # Executar todas as tarefas em paralelo
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Erro em tarefa de coleta: {result}")
                    continue
                if isinstance(result, list):
                    all_articles.extend(result)

        # Filtrar por relevancia da moeda
        filtered = self._filter_by_coin(all_articles, coin)

        # Deduplicar por titulo
        deduplicated = self._deduplicate(filtered)

        # Ordenar por timestamp (mais recente primeiro)
        deduplicated.sort(
            key=lambda x: x.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )

        # Limitar quantidade
        deduplicated = deduplicated[: self.max_articles]

        # Enriquecer com palavras-chave de sentimento
        for article in deduplicated:
            combined_text = f"{article.get('title', '')} {article.get('text', '')}"
            article["sentiment_keywords"] = _extract_sentiment_keywords(combined_text)

        logger.info(
            f"Crawler avancado: {len(all_articles)} coletados, "
            f"{len(filtered)} relevantes para {coin}, "
            f"{len(deduplicated)} apos deduplicacao"
        )

        self._cache.set(cache_key, deduplicated)
        return deduplicated

    def get_fear_greed_index(self) -> dict:
        """Busca o indice Fear & Greed da alternative.me.

        Returns:
            Dict com chaves: value (int), classification (str), timestamp (datetime)
        """
        # Verificar cache
        cached = self._cache.get("fear_greed")
        if cached is not None:
            return cached

        result = {
            "value": 50,
            "classification": "Neutral",
            "timestamp": datetime.now(timezone.utc),
        }

        try:
            self._rate_limiter.wait_sync("alternative.me")
            resp = requests.get(
                self.FEAR_GREED_URL,
                headers={"User-Agent": _rotate_ua()},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if data:
                entry = data[0]
                result = {
                    "value": int(entry.get("value", 50)),
                    "classification": entry.get("value_classification", "Neutral"),
                    "timestamp": datetime.fromtimestamp(
                        int(entry.get("timestamp", time.time())), tz=timezone.utc
                    ),
                }
                logger.info(
                    f"Fear & Greed Index: {result['value']} ({result['classification']})"
                )
        except Exception as e:
            logger.error(f"Erro ao buscar Fear & Greed Index: {e}")

        self._cache.set("fear_greed", result)
        return result

    def add_crawler_features(self, df: pd.DataFrame, coin: str = "BTC") -> pd.DataFrame:
        """Adiciona features do crawler avancado ao DataFrame de precos.

        Features geradas:
            - crawler_news_count: quantidade de noticias coletadas
            - crawler_sentiment_mean: sentimento medio dos artigos
            - crawler_fear_greed: valor do indice Fear & Greed
            - crawler_buzz_score: intensidade relativa de noticias
            - crawler_source_diversity: numero de fontes distintas

        Args:
            df: DataFrame OHLCV com coluna 'timestamp'
            coin: Simbolo da moeda

        Returns:
            DataFrame com features adicionadas
        """
        df = df.copy()

        # Valores padrao (NaN) para quando coleta falhar
        default_features = {feat: np.nan for feat in self.get_feature_names()}

        try:
            articles = self.fetch_all(coin=coin)
        except Exception as e:
            logger.error(f"Erro ao buscar noticias para features: {e}")
            for feat, val in default_features.items():
                df[feat] = val
            return df

        # Buscar Fear & Greed Index
        try:
            fng = self.get_fear_greed_index()
            fear_greed_value = fng["value"]
        except Exception:
            fear_greed_value = np.nan

        if not articles:
            logger.warning("Nenhuma noticia coletada, features serao NaN")
            for feat, val in default_features.items():
                df[feat] = val
            df["crawler_fear_greed"] = fear_greed_value
            return df

        # Calcular sentimento de cada artigo
        sentiments = []
        for article in articles:
            combined = f"{article.get('title', '')} {article.get('text', '')}"
            sentiments.append(_compute_sentiment(combined))

        # Contagem de fontes unicas
        unique_sources = set()
        for article in articles:
            # Extrair fonte base (ex: "Reddit/r/bitcoin" -> "Reddit")
            source = article.get("source", "unknown").split("/")[0]
            unique_sources.add(source)

        # Converter artigos para DataFrame auxiliar para agregacao diaria
        news_df = pd.DataFrame(articles)
        news_df["sentiment"] = sentiments
        news_df["timestamp"] = pd.to_datetime(news_df["timestamp"], utc=True).dt.tz_localize(None)
        news_df["_date"] = news_df["timestamp"].dt.normalize()

        # Agregar por dia
        daily = news_df.groupby("_date").agg(
            news_count=("sentiment", "count"),
            sentiment_mean=("sentiment", "mean"),
        ).reset_index()

        # Calcular buzz score (razao entre contagem do dia e media movel de 7 dias)
        daily = daily.sort_values("_date").reset_index(drop=True)
        daily["news_count_ma7"] = daily["news_count"].rolling(7, min_periods=1).mean()
        daily["buzz_score"] = (
            daily["news_count"] / daily["news_count_ma7"].replace(0, np.nan)
        )

        # Preparar merge com DataFrame principal
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()

        merge_cols = daily[["_date", "news_count", "sentiment_mean", "buzz_score"]]
        merge_cols = merge_cols.drop_duplicates(subset=["_date"])

        df = df.merge(merge_cols, on="_date", how="left")

        # Renomear colunas para nomes padrao
        df = df.rename(columns={
            "news_count": "crawler_news_count",
            "sentiment_mean": "crawler_sentiment_mean",
            "buzz_score": "crawler_buzz_score",
        })

        # Adicionar features fixas (nao variam por dia neste batch)
        df["crawler_fear_greed"] = fear_greed_value
        df["crawler_source_diversity"] = len(unique_sources)

        # Preencher NaN com valores neutros
        df["crawler_news_count"] = df["crawler_news_count"].fillna(0)
        df["crawler_sentiment_mean"] = df["crawler_sentiment_mean"].fillna(0.0)
        df["crawler_buzz_score"] = df["crawler_buzz_score"].fillna(1.0)
        df["crawler_source_diversity"] = df["crawler_source_diversity"].fillna(0)

        df = df.drop(columns=["_date"])

        logger.info("Features do crawler avancado adicionadas com sucesso")
        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features geradas pelo crawler."""
        return [
            "crawler_news_count",
            "crawler_sentiment_mean",
            "crawler_fear_greed",
            "crawler_buzz_score",
            "crawler_source_diversity",
        ]

    # ──────────────────────────────────────────────────────────────
    # Metodos internos - busca assincrona (aiohttp)
    # ──────────────────────────────────────────────────────────────

    async def _async_request(
        self,
        session: "aiohttp.ClientSession",
        url: str,
        domain: str,
        headers: dict | None = None,
    ) -> dict | str | None:
        """Faz requisicao HTTP assincrona com retry e backoff exponencial.

        Args:
            session: Sessao aiohttp ativa
            url: URL alvo
            domain: Dominio para rate limiting
            headers: Headers adicionais

        Returns:
            Resposta parseada (JSON dict ou texto str), ou None em caso de falha
        """
        default_headers = {"User-Agent": _rotate_ua()}
        if headers:
            default_headers.update(headers)

        for attempt in range(1, self.max_retries + 1):
            try:
                await self._rate_limiter.wait_async(domain)
                async with session.get(url, headers=default_headers) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get("Content-Type", "")
                        if "json" in content_type or "javascript" in content_type:
                            return await resp.json(content_type=None)
                        else:
                            return await resp.text()
                    elif resp.status == 429:
                        # Rate limited - esperar mais antes de tentar novamente
                        wait_time = 2 ** attempt * 2
                        logger.warning(
                            f"Rate limited ({url}), aguardando {wait_time}s "
                            f"(tentativa {attempt}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(
                            f"HTTP {resp.status} em {url} "
                            f"(tentativa {attempt}/{self.max_retries})"
                        )
                        if attempt < self.max_retries:
                            await asyncio.sleep(2 ** attempt)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout em {url} (tentativa {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Erro em {url}: {e} (tentativa {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

        logger.error(f"Todas as {self.max_retries} tentativas falharam para {url}")
        return None

    async def _async_fetch_cryptocompare(self, session: "aiohttp.ClientSession") -> list[dict]:
        """Busca noticias do CryptoCompare de forma assincrona."""
        cached = self._cache.get("cryptocompare")
        if cached is not None:
            return cached

        data = await self._async_request(
            session, self.CRYPTOCOMPARE_URL, "min-api.cryptocompare.com"
        )
        if not isinstance(data, dict):
            return []

        articles = []
        for item in data.get("Data", []):
            articles.append({
                "title": _clean_html(item.get("title", "")),
                "text": _clean_html(item.get("body", "")),
                "source": f"CryptoCompare/{item.get('source', 'unknown')}",
                "timestamp": datetime.fromtimestamp(
                    item.get("published_on", 0), tz=timezone.utc
                ),
                "coin_relevance": 0.0,
                "sentiment_keywords": [],
            })

        logger.info(f"CryptoCompare (async): {len(articles)} artigos coletados")
        self._cache.set("cryptocompare", articles)
        return articles

    async def _async_fetch_coingecko(self, session: "aiohttp.ClientSession") -> list[dict]:
        """Busca atualizacoes de status do CoinGecko de forma assincrona."""
        cached = self._cache.get("coingecko")
        if cached is not None:
            return cached

        data = await self._async_request(
            session, self.COINGECKO_NEWS_URL, "api.coingecko.com"
        )
        if not isinstance(data, dict):
            return []

        articles = []
        for item in data.get("status_updates", []):
            description = _clean_html(item.get("description", ""))
            # CoinGecko retorna atualizacoes de projetos, nao exatamente noticias
            project = item.get("project", {})
            project_name = project.get("name", "unknown") if isinstance(project, dict) else "unknown"

            # Parsear data ISO 8601
            created = item.get("created_at", "")
            timestamp = _parse_rss_date(created)

            articles.append({
                "title": f"{project_name}: {description[:80]}",
                "text": description,
                "source": f"CoinGecko/{project_name}",
                "timestamp": timestamp,
                "coin_relevance": 0.0,
                "sentiment_keywords": [],
            })

        logger.info(f"CoinGecko (async): {len(articles)} atualizacoes coletadas")
        self._cache.set("coingecko", articles)
        return articles

    async def _async_fetch_reddit(
        self, session: "aiohttp.ClientSession", subreddit: str
    ) -> list[dict]:
        """Busca posts do Reddit via API publica JSON de forma assincrona."""
        cache_key = f"reddit_{subreddit}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
        headers = {"User-Agent": f"CriptoDT-AdvancedCrawler/2.0 (subreddit:{subreddit})"}

        data = await self._async_request(session, url, "reddit.com", headers=headers)
        if not isinstance(data, dict):
            return []

        articles = []
        posts = data.get("data", {}).get("children", [])
        for post in posts:
            pd_ = post.get("data", {})
            # Ignorar posts fixados
            if pd_.get("stickied", False):
                continue
            title = pd_.get("title", "")
            selftext = _clean_html(pd_.get("selftext", ""))

            articles.append({
                "title": title,
                "text": selftext,
                "source": f"Reddit/r/{subreddit}",
                "timestamp": datetime.fromtimestamp(
                    pd_.get("created_utc", 0), tz=timezone.utc
                ),
                "coin_relevance": 0.0,
                "sentiment_keywords": [],
            })

        logger.info(f"Reddit r/{subreddit} (async): {len(articles)} posts coletados")
        self._cache.set(cache_key, articles)
        return articles

    async def _async_fetch_rss(
        self, session: "aiohttp.ClientSession", url: str, source_name: str
    ) -> list[dict]:
        """Busca e parseia feed RSS de forma assincrona."""
        cache_key = f"rss_{source_name}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Extrair dominio da URL para rate limiting
        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        text = await self._async_request(session, url, domain)
        if not isinstance(text, str):
            return []

        articles = []
        try:
            root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)

            # Suporte para RSS 2.0 (channel/item) e Atom (entry)
            items = root.findall(".//item")
            if not items:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                items = root.findall(".//atom:entry", ns)

            for item in items:
                title = _clean_html(
                    _get_xml_text(item, "title") or ""
                )
                description = _clean_html(
                    _get_xml_text(item, "description")
                    or _get_xml_text(item, "summary")
                    or _get_xml_text(item, "content")
                    or ""
                )
                pub_date_str = (
                    _get_xml_text(item, "pubDate")
                    or _get_xml_text(item, "published")
                    or ""
                )

                articles.append({
                    "title": title,
                    "text": description,
                    "source": source_name,
                    "timestamp": _parse_rss_date(pub_date_str),
                    "coin_relevance": 0.0,
                    "sentiment_keywords": [],
                })

        except ET.ParseError as e:
            logger.error(f"Erro ao parsear XML do RSS {source_name}: {e}")
            return []

        logger.info(f"RSS {source_name} (async): {len(articles)} artigos coletados")
        self._cache.set(cache_key, articles)
        return articles

    # ──────────────────────────────────────────────────────────────
    # Metodos internos - busca sincrona (fallback sem aiohttp)
    # ──────────────────────────────────────────────────────────────

    def _sync_request(
        self, url: str, domain: str, headers: dict | None = None
    ) -> requests.Response | None:
        """Faz requisicao HTTP sincrona com retry e backoff exponencial."""
        default_headers = {"User-Agent": _rotate_ua()}
        if headers:
            default_headers.update(headers)

        for attempt in range(1, self.max_retries + 1):
            try:
                self._rate_limiter.wait_sync(domain)
                resp = requests.get(url, headers=default_headers, timeout=30)

                if resp.status_code == 200:
                    return resp
                elif resp.status_code == 429:
                    wait_time = 2 ** attempt * 2
                    logger.warning(
                        f"Rate limited ({url}), aguardando {wait_time}s "
                        f"(tentativa {attempt}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"HTTP {resp.status_code} em {url} "
                        f"(tentativa {attempt}/{self.max_retries})"
                    )
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
            except requests.RequestException as e:
                logger.error(f"Erro em {url}: {e} (tentativa {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)

        logger.error(f"Todas as {self.max_retries} tentativas falharam para {url}")
        return None

    def _fetch_all_sync(self, coin: str) -> list[dict]:
        """Busca noticias de forma sincrona (fallback quando aiohttp nao esta disponivel)."""
        cache_key = f"all_news_{coin}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        all_articles: list[dict] = []
        self._seen_hashes.clear()

        # CryptoCompare
        try:
            all_articles.extend(self._sync_fetch_cryptocompare())
        except Exception as e:
            logger.error(f"Erro no CryptoCompare (sync): {e}")

        # CoinGecko
        try:
            all_articles.extend(self._sync_fetch_coingecko())
        except Exception as e:
            logger.error(f"Erro no CoinGecko (sync): {e}")

        # Reddit
        for sub in self.REDDIT_SUBS:
            try:
                all_articles.extend(self._sync_fetch_reddit(sub))
            except Exception as e:
                logger.error(f"Erro no Reddit r/{sub} (sync): {e}")

        # RSS feeds
        for name, url in self.RSS_FEEDS.items():
            try:
                all_articles.extend(self._sync_fetch_rss(url, name))
            except Exception as e:
                logger.error(f"Erro no RSS {name} (sync): {e}")

        # Filtrar, deduplicar, ordenar, limitar
        filtered = self._filter_by_coin(all_articles, coin)
        deduplicated = self._deduplicate(filtered)
        deduplicated.sort(
            key=lambda x: x.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )
        deduplicated = deduplicated[: self.max_articles]

        # Enriquecer com palavras-chave de sentimento
        for article in deduplicated:
            combined_text = f"{article.get('title', '')} {article.get('text', '')}"
            article["sentiment_keywords"] = _extract_sentiment_keywords(combined_text)

        logger.info(
            f"Crawler avancado (sync): {len(all_articles)} coletados, "
            f"{len(filtered)} relevantes para {coin}, "
            f"{len(deduplicated)} apos deduplicacao"
        )

        self._cache.set(cache_key, deduplicated)
        return deduplicated

    def _sync_fetch_cryptocompare(self) -> list[dict]:
        """Busca noticias do CryptoCompare de forma sincrona."""
        cached = self._cache.get("cryptocompare")
        if cached is not None:
            return cached

        resp = self._sync_request(self.CRYPTOCOMPARE_URL, "min-api.cryptocompare.com")
        if resp is None:
            return []

        articles = []
        for item in resp.json().get("Data", []):
            articles.append({
                "title": _clean_html(item.get("title", "")),
                "text": _clean_html(item.get("body", "")),
                "source": f"CryptoCompare/{item.get('source', 'unknown')}",
                "timestamp": datetime.fromtimestamp(
                    item.get("published_on", 0), tz=timezone.utc
                ),
                "coin_relevance": 0.0,
                "sentiment_keywords": [],
            })

        logger.info(f"CryptoCompare (sync): {len(articles)} artigos coletados")
        self._cache.set("cryptocompare", articles)
        return articles

    def _sync_fetch_coingecko(self) -> list[dict]:
        """Busca atualizacoes do CoinGecko de forma sincrona."""
        cached = self._cache.get("coingecko")
        if cached is not None:
            return cached

        resp = self._sync_request(self.COINGECKO_NEWS_URL, "api.coingecko.com")
        if resp is None:
            return []

        articles = []
        for item in resp.json().get("status_updates", []):
            description = _clean_html(item.get("description", ""))
            project = item.get("project", {})
            project_name = project.get("name", "unknown") if isinstance(project, dict) else "unknown"
            created = item.get("created_at", "")

            articles.append({
                "title": f"{project_name}: {description[:80]}",
                "text": description,
                "source": f"CoinGecko/{project_name}",
                "timestamp": _parse_rss_date(created),
                "coin_relevance": 0.0,
                "sentiment_keywords": [],
            })

        logger.info(f"CoinGecko (sync): {len(articles)} atualizacoes coletadas")
        self._cache.set("coingecko", articles)
        return articles

    def _sync_fetch_reddit(self, subreddit: str) -> list[dict]:
        """Busca posts do Reddit de forma sincrona."""
        cache_key = f"reddit_{subreddit}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
        headers = {"User-Agent": f"CriptoDT-AdvancedCrawler/2.0 (subreddit:{subreddit})"}

        resp = self._sync_request(url, "reddit.com", headers=headers)
        if resp is None:
            return []

        articles = []
        posts = resp.json().get("data", {}).get("children", [])
        for post in posts:
            pd_ = post.get("data", {})
            if pd_.get("stickied", False):
                continue

            articles.append({
                "title": pd_.get("title", ""),
                "text": _clean_html(pd_.get("selftext", "")),
                "source": f"Reddit/r/{subreddit}",
                "timestamp": datetime.fromtimestamp(
                    pd_.get("created_utc", 0), tz=timezone.utc
                ),
                "coin_relevance": 0.0,
                "sentiment_keywords": [],
            })

        logger.info(f"Reddit r/{subreddit} (sync): {len(articles)} posts coletados")
        self._cache.set(cache_key, articles)
        return articles

    def _sync_fetch_rss(self, url: str, source_name: str) -> list[dict]:
        """Busca e parseia feed RSS de forma sincrona."""
        cache_key = f"rss_{source_name}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        from urllib.parse import urlparse
        domain = urlparse(url).netloc

        resp = self._sync_request(url, domain)
        if resp is None:
            return []

        articles = []
        try:
            root = ET.fromstring(resp.content)

            items = root.findall(".//item")
            if not items:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                items = root.findall(".//atom:entry", ns)

            for item in items:
                title = _clean_html(_get_xml_text(item, "title") or "")
                description = _clean_html(
                    _get_xml_text(item, "description")
                    or _get_xml_text(item, "summary")
                    or _get_xml_text(item, "content")
                    or ""
                )
                pub_date_str = (
                    _get_xml_text(item, "pubDate")
                    or _get_xml_text(item, "published")
                    or ""
                )

                articles.append({
                    "title": title,
                    "text": description,
                    "source": source_name,
                    "timestamp": _parse_rss_date(pub_date_str),
                    "coin_relevance": 0.0,
                    "sentiment_keywords": [],
                })

        except ET.ParseError as e:
            logger.error(f"Erro ao parsear XML do RSS {source_name}: {e}")
            return []

        logger.info(f"RSS {source_name} (sync): {len(articles)} artigos coletados")
        self._cache.set(cache_key, articles)
        return articles

    # ──────────────────────────────────────────────────────────────
    # Metodos internos - utilidades
    # ──────────────────────────────────────────────────────────────

    def _filter_by_coin(self, articles: list[dict], coin: str) -> list[dict]:
        """Filtra artigos por relevancia para a moeda especificada.

        Utiliza palavras-chave associadas ao simbolo da moeda para
        calcular score de relevancia. Artigos sem nenhuma correspondencia
        sao descartados.
        """
        coin = coin.upper()
        keywords = _COIN_KEYWORDS.get(coin, [coin.lower()])

        filtered = []
        for article in articles:
            text_lower = (
                article.get("title", "") + " " + article.get("text", "")
            ).lower()

            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                article = article.copy()
                article["coin_relevance"] = min(matches / len(keywords), 1.0)
                filtered.append(article)

        return filtered

    def _deduplicate(self, articles: list[dict]) -> list[dict]:
        """Remove artigos duplicados baseado em hash do titulo.

        Utiliza hash MD5 do titulo normalizado (apenas alfanumericos
        minusculos) para identificar duplicatas de forma eficiente.
        """
        unique = []
        for article in articles:
            title = article.get("title", "")
            if not title:
                continue
            h = _title_hash(title)
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                unique.append(article)
        return unique


# ──────────────────────────────────────────────────────────────────────
# Funcao auxiliar para XML (fora da classe para reutilizacao)
# ──────────────────────────────────────────────────────────────────────

def _get_xml_text(element: ET.Element, tag: str) -> str | None:
    """Extrai texto de um sub-elemento XML de forma segura."""
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return None
