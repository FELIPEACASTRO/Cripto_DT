"""Analise de sentimento NLP com Sentence-BERT (baseado em Tokyo U of Science)."""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Tentativa de importar sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False
    logger.warning("sentence-transformers nao disponivel")

# Tentativa de importar TextBlob como fallback
try:
    from textblob import TextBlob

    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False
    logger.warning("textblob nao disponivel — fallback de sentimento desabilitado")


class NLPSentimentAnalyzer:
    """Analisa sentimento de noticias cripto usando Sentence-BERT.

    Usa sentence-transformers para gerar embeddings e classificar sentimento.
    Fallback com TextBlob se sentence-transformers nao estiver disponivel.
    """

    # URL da API de noticias CryptoCompare (gratuita)
    CRYPTO_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"

    # Modelo SBERT para embeddings
    _DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        """Inicializa o analisador de sentimento.

        Args:
            model_name: Nome do modelo sentence-transformers (opcional)
        """
        self._model = None
        self._model_name = model_name or self._DEFAULT_MODEL

        if _SBERT_AVAILABLE:
            try:
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Modelo SBERT carregado: {self._model_name}")
            except Exception as e:
                logger.error(f"Erro ao carregar modelo SBERT: {e}")

    def analyze_text(self, texts: list[str]) -> np.ndarray:
        """Analisa sentimento de uma lista de textos.

        Usa sentence-transformers para gerar embeddings e classificar
        sentimento. Fallback com TextBlob se SBERT nao disponivel.

        Args:
            texts: Lista de textos para analisar

        Returns:
            Array de scores de sentimento no intervalo [-1, 1]
        """
        if not texts:
            return np.array([])

        # Caminho 1: Sentence-BERT disponivel
        if self._model is not None:
            return self._analyze_with_sbert(texts)

        # Caminho 2: Fallback com TextBlob
        if _TEXTBLOB_AVAILABLE:
            return self._analyze_with_textblob(texts)

        # Caminho 3: Nenhum disponivel — retornar zeros (neutro)
        logger.warning("Nenhum backend de sentimento disponivel — retornando neutro")
        return np.zeros(len(texts))

    def _analyze_with_sbert(self, texts: list[str]) -> np.ndarray:
        """Analisa sentimento usando Sentence-BERT.

        Gera embeddings e usa heuristica baseada em similaridade com
        frases de referencia positivas e negativas.
        """
        try:
            # Frases de referencia para sentimento
            positive_refs = [
                "price surge bullish rally gains profit growth",
                "adoption breakthrough innovation positive momentum",
                "strong support uptrend buying opportunity",
            ]
            negative_refs = [
                "crash bearish dump losses decline risk",
                "hack fraud regulation ban negative selloff",
                "weak resistance downtrend selling pressure",
            ]

            # Gerar embeddings para textos e referencias
            text_embeddings = self._model.encode(texts, show_progress_bar=False)
            pos_embeddings = self._model.encode(positive_refs, show_progress_bar=False)
            neg_embeddings = self._model.encode(negative_refs, show_progress_bar=False)

            # Media dos embeddings de referencia
            pos_center = np.mean(pos_embeddings, axis=0)
            neg_center = np.mean(neg_embeddings, axis=0)

            # Similaridade cosseno com cada polo
            scores = []
            for emb in text_embeddings:
                sim_pos = np.dot(emb, pos_center) / (
                    np.linalg.norm(emb) * np.linalg.norm(pos_center) + 1e-8
                )
                sim_neg = np.dot(emb, neg_center) / (
                    np.linalg.norm(emb) * np.linalg.norm(neg_center) + 1e-8
                )
                # Score normalizado entre -1 e 1
                score = float(sim_pos - sim_neg)
                scores.append(np.clip(score, -1.0, 1.0))

            logger.info(f"Sentimento SBERT calculado para {len(texts)} textos")
            return np.array(scores)

        except Exception as e:
            logger.error(f"Erro na analise SBERT: {e}")
            if _TEXTBLOB_AVAILABLE:
                return self._analyze_with_textblob(texts)
            return np.zeros(len(texts))

    def _analyze_with_textblob(self, texts: list[str]) -> np.ndarray:
        """Fallback: analisa sentimento com TextBlob."""
        try:
            scores = []
            for text in texts:
                blob = TextBlob(text)
                # TextBlob retorna polarity em [-1, 1]
                scores.append(float(blob.sentiment.polarity))
            logger.info(f"Sentimento TextBlob calculado para {len(texts)} textos")
            return np.array(scores)
        except Exception as e:
            logger.error(f"Erro na analise TextBlob: {e}")
            return np.zeros(len(texts))

    def fetch_crypto_news(
        self, coin: str, days: int = 7
    ) -> list[dict]:
        """Busca noticias recentes de criptomoedas via CryptoCompare.

        Args:
            coin: Simbolo da moeda (ex: BTC, ETH)
            days: Numero de dias de historico

        Returns:
            Lista de dicts com title, body, published_on
        """
        try:
            resp = requests.get(
                self.CRYPTO_NEWS_URL,
                params={"lang": "EN", "categories": coin},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("Data", [])
        except requests.RequestException as e:
            logger.error(f"Erro ao buscar noticias para {coin}: {e}")
            return []

        if not data:
            logger.warning(f"Nenhuma noticia encontrada para {coin}")
            return []

        # Filtrar por periodo
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff_ts = cutoff.timestamp()

        news = []
        for item in data:
            published_on = item.get("published_on", 0)
            if published_on >= cutoff_ts:
                news.append({
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "published_on": pd.to_datetime(published_on, unit="s", utc=True),
                })

        logger.info(f"Noticias coletadas para {coin}: {len(news)} nos ultimos {days} dias")
        return news

    def add_nlp_features(
        self, df: pd.DataFrame, coin: str, days: int = 7
    ) -> pd.DataFrame:
        """Busca noticias e adiciona features de sentimento NLP ao DataFrame.

        Features geradas:
            - nlp_sentiment_mean: media do sentimento das noticias recentes
            - nlp_sentiment_std: variacao do sentimento
            - nlp_news_count: numero de noticias
            - nlp_sentiment_momentum: mudanca do sentimento vs dia anterior

        Args:
            df: DataFrame OHLCV com coluna 'timestamp'
            coin: Simbolo da moeda (ex: BTC, ETH)
            days: Dias de noticias para buscar

        Returns:
            DataFrame com features NLP adicionadas
        """
        df = df.copy()

        # Buscar noticias
        try:
            news = self.fetch_crypto_news(coin, days=days)
        except Exception as e:
            logger.error(f"Falha ao buscar noticias: {e}")
            return self._add_empty_features(df)

        if not news:
            logger.warning("Sem noticias — adicionando features vazias")
            return self._add_empty_features(df)

        # Analisar sentimento dos titulos + corpo
        texts = [f"{n['title']}. {n['body'][:200]}" for n in news]
        try:
            sentiments = self.analyze_text(texts)
        except Exception as e:
            logger.error(f"Erro na analise de sentimento: {e}")
            return self._add_empty_features(df)

        # Criar DataFrame de noticias com sentimento por data
        news_df = pd.DataFrame(news)
        news_df["sentiment"] = sentiments
        news_df["_date"] = news_df["published_on"].dt.normalize()

        # Agregar por data
        daily_sent = news_df.groupby("_date").agg(
            nlp_sentiment_mean=("sentiment", "mean"),
            nlp_sentiment_std=("sentiment", "std"),
            nlp_news_count=("sentiment", "count"),
        ).reset_index()
        daily_sent["nlp_sentiment_std"] = daily_sent["nlp_sentiment_std"].fillna(0)

        # Calcular momentum (diferenca dia a dia)
        daily_sent = daily_sent.sort_values("_date")
        daily_sent["nlp_sentiment_momentum"] = daily_sent["nlp_sentiment_mean"].diff(1)
        daily_sent["nlp_sentiment_momentum"] = (
            daily_sent["nlp_sentiment_momentum"].fillna(0)
        )

        # Merge com DataFrame principal
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        df = df.merge(daily_sent, on="_date", how="left")

        # Preencher dias sem noticias com zero (neutro)
        for col in self.get_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
            else:
                df[col] = 0.0

        df = df.drop(columns=["_date"])
        return df

    def _add_empty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas de features NLP com valores zero."""
        for col in self.get_feature_names():
            df[col] = 0.0
        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features de sentimento NLP."""
        return [
            "nlp_sentiment_mean",
            "nlp_sentiment_std",
            "nlp_news_count",
            "nlp_sentiment_momentum",
        ]
