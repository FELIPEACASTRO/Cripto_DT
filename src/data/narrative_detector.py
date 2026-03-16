"""Deteccao de narrativas cripto — inspirado em aixbt by Virtuals (aiagentstore.ai).

Mercados cripto se movem por narrativas (DeFi Summer, AI Tokens, RWA, Memecoins,
L2 Season, etc). Detectar narrativas emergentes antes que se tornem mainstream = alpha.

Este modulo analisa artigos de noticias para identificar e pontuar narrativas
dominantes, calcular momentum e gerar sinais contrarios.
"""

import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NarrativeDetector:
    """Detecta e pontua narrativas dominantes no mercado cripto.

    Analisa noticias para identificar quais temas estao ganhando forca,
    calcula momentum de narrativas e gera features para modelos de ML.
    """

    # Categorias de narrativas com palavras-chave associadas
    NARRATIVE_KEYWORDS: dict[str, list[str]] = {
        "defi": [
            "defi", "yield", "farming", "liquidity", "dex", "amm",
            "lending", "borrowing",
        ],
        "ai_crypto": [
            "ai", "artificial intelligence", "machine learning", "gpt",
            "llm", "neural",
        ],
        "rwa": [
            "real world assets", "rwa", "tokenization", "treasury", "bonds",
        ],
        "memecoins": [
            "meme", "doge", "shib", "pepe", "bonk", "frog",
        ],
        "l2_scaling": [
            "layer 2", "l2", "rollup", "zk", "optimistic", "scaling",
        ],
        "regulation": [
            "sec", "regulation", "compliance", "lawsuit", "enforcement",
        ],
        "institutional": [
            "blackrock", "etf", "institutional", "custody", "grayscale",
        ],
        "gaming_nft": [
            "gaming", "nft", "metaverse", "play to earn", "gamefi",
        ],
    }

    # Mapeamento de moedas para narrativas mais relevantes (alinhamento)
    COIN_NARRATIVE_ALIGNMENT: dict[str, list[str]] = {
        "BTC": ["institutional", "regulation"],
        "ETH": ["defi", "l2_scaling", "institutional"],
        "SOL": ["defi", "memecoins", "gaming_nft"],
        "BNB": ["defi", "regulation"],
        "AVAX": ["defi", "gaming_nft", "institutional"],
        "MATIC": ["l2_scaling", "gaming_nft", "defi"],
        "LINK": ["defi", "rwa", "institutional"],
        "DOGE": ["memecoins", "institutional"],
        "ADA": ["defi", "rwa"],
        "DOT": ["l2_scaling", "defi"],
        "ARB": ["l2_scaling", "defi"],
        "OP": ["l2_scaling", "defi"],
        "RNDR": ["ai_crypto", "gaming_nft"],
        "FET": ["ai_crypto"],
        "AGIX": ["ai_crypto"],
    }

    # Fator de decaimento exponencial para recencia (por dia)
    RECENCY_DECAY = 0.85

    def __init__(self, news_scraper=None):
        """Inicializa o detector de narrativas.

        Args:
            news_scraper: Instancia de NewsScraper para coleta de noticias.
                          Se None, cria uma instancia basica internamente.
        """
        if news_scraper is not None:
            self._scraper = news_scraper
        else:
            # Importacao tardia para evitar dependencia circular
            try:
                from src.data.news_scraper import NewsScraper
                self._scraper = NewsScraper(max_articles=100)
                logger.info("NewsScraper criado internamente pelo NarrativeDetector")
            except ImportError:
                self._scraper = None
                logger.warning(
                    "NewsScraper nao disponivel — "
                    "NarrativeDetector funcionara apenas com artigos fornecidos"
                )

        # Cache de scores historicos para calculo de momentum
        # Formato: {data_str: {narrativa: score}}
        self._history: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def detect_narratives(self, articles: list[dict]) -> dict[str, float]:
        """Pontua cada categoria de narrativa a partir dos artigos fornecidos.

        Cada artigo deve conter ao menos 'title' e opcionalmente 'text'/'body'
        e 'timestamp'. A pontuacao e ponderada pela recencia do artigo.

        Args:
            articles: Lista de dicts com chaves title, text/body, timestamp

        Returns:
            Dicionario {nome_narrativa: score_normalizado}
        """
        if not articles:
            logger.warning("Nenhum artigo fornecido para deteccao de narrativas")
            return {cat: 0.0 for cat in self.NARRATIVE_KEYWORDS}

        scores: dict[str, float] = {cat: 0.0 for cat in self.NARRATIVE_KEYWORDS}
        now = datetime.now(timezone.utc)

        for article in articles:
            # Extrair texto completo do artigo (suporta ambos os formatos)
            title = article.get("title", "")
            body = article.get("text", "") or article.get("body", "")
            full_text = f"{title} {body}".lower()

            if not full_text.strip():
                continue

            # Calcular peso de recencia baseado na idade do artigo
            recency_weight = self._compute_recency_weight(article, now)

            # Contar ocorrencias de keywords para cada narrativa
            for category, keywords in self.NARRATIVE_KEYWORDS.items():
                keyword_hits = 0
                for kw in keywords:
                    # Contar ocorrencias da keyword no texto
                    count = full_text.count(kw)
                    if count > 0:
                        # Bonus por keyword no titulo (mais relevante)
                        title_bonus = 1.5 if kw in title.lower() else 1.0
                        keyword_hits += count * title_bonus

                # Ponderar pelo peso de recencia
                scores[category] += keyword_hits * recency_weight

        # Normalizar scores para intervalo [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {cat: score / max_score for cat, score in scores.items()}

        logger.info(
            f"Narrativas detectadas em {len(articles)} artigos — "
            f"top: {max(scores, key=scores.get)} ({scores[max(scores, key=scores.get)]:.2f})"
        )
        return scores

    def get_dominant_narrative(self, articles: list[dict]) -> tuple[str, float]:
        """Identifica a narrativa dominante nos artigos fornecidos.

        Args:
            articles: Lista de artigos para analisar

        Returns:
            Tupla (nome_narrativa, score) da narrativa mais forte
        """
        scores = self.detect_narratives(articles)

        if not scores:
            return ("none", 0.0)

        dominant = max(scores, key=scores.get)
        return (dominant, scores[dominant])

    def add_narrative_features(self, df: pd.DataFrame, coin: str) -> pd.DataFrame:
        """Adiciona features de narrativa ao DataFrame de precos.

        Busca noticias via scraper (se disponivel) e calcula:
        - narrative_dominant_score: forca da narrativa dominante
        - narrative_diversity: entropia das narrativas (diversidade)
        - narrative_momentum_7d: variacao da narrativa dominante em 7 dias
        - narrative_alignment: relevancia da narrativa para a moeda
        - narrative_contrarian: divergencia entre preco e narrativa

        Args:
            df: DataFrame OHLCV com coluna 'timestamp' e 'close'
            coin: Simbolo da moeda (ex: BTC, ETH)

        Returns:
            DataFrame com features de narrativa adicionadas
        """
        df = df.copy()
        coin = coin.upper()

        # Buscar artigos via scraper se disponivel
        articles = self._fetch_articles(coin)

        if not articles:
            logger.warning(
                f"Sem artigos para {coin} — adicionando features de narrativa vazias"
            )
            return self._add_empty_features(df)

        # Agrupar artigos por data para calculo de features diarias
        articles_by_date = self._group_articles_by_date(articles)

        # Calcular scores de narrativa por data
        daily_narrative_data = []
        for date_str, day_articles in sorted(articles_by_date.items()):
            scores = self.detect_narratives(day_articles)

            # Armazenar no historico para momentum
            self._history[date_str] = scores

            dominant_name, dominant_score = max(
                scores.items(), key=lambda x: x[1]
            )

            # Calcular entropia (diversidade de narrativas)
            diversity = self._compute_entropy(scores)

            # Calcular momentum comparando com 7 dias atras
            momentum_7d = self._compute_momentum(date_str, days=7)

            # Calcular alinhamento narrativa-moeda
            alignment = self._compute_alignment(scores, coin)

            daily_narrative_data.append({
                "_date": pd.Timestamp(date_str, tz="UTC"),
                "narrative_dominant_score": dominant_score,
                "narrative_diversity": diversity,
                "narrative_momentum_7d": momentum_7d,
                "narrative_alignment": alignment,
                "_dominant_name": dominant_name,
            })

        if not daily_narrative_data:
            return self._add_empty_features(df)

        narrative_df = pd.DataFrame(daily_narrative_data)
        narrative_df["_date"] = narrative_df["_date"].dt.normalize()
        narrative_df = narrative_df.drop_duplicates(subset=["_date"])

        # Merge com DataFrame principal
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        df = df.merge(narrative_df, on="_date", how="left")

        # Calcular sinal contrario (divergencia preco vs narrativa)
        df = self._compute_contrarian_signal(df)

        # Remover colunas auxiliares
        df = df.drop(columns=["_date", "_dominant_name"], errors="ignore")

        # Preencher NaN com valores neutros
        for feat in self.get_feature_names():
            if feat in df.columns:
                df[feat] = df[feat].fillna(0.0)
            else:
                df[feat] = 0.0

        logger.info(f"Features de narrativa adicionadas para {coin}")
        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features geradas pelo detector de narrativas.

        Returns:
            Lista com nomes das features:
            - narrative_dominant_score: score da narrativa dominante
            - narrative_diversity: quantas narrativas ativas (entropia)
            - narrative_momentum_7d: mudanca na narrativa dominante em 7d
            - narrative_alignment: alinhamento narrativa-moeda (relevancia)
            - narrative_contrarian: divergencia preco vs narrativa
        """
        return [
            "narrative_dominant_score",
            "narrative_diversity",
            "narrative_momentum_7d",
            "narrative_alignment",
            "narrative_contrarian",
        ]

    # ------------------------------------------------------------------
    # Metodos internos
    # ------------------------------------------------------------------

    def _compute_recency_weight(self, article: dict, now: datetime) -> float:
        """Calcula peso de recencia para um artigo usando decaimento exponencial.

        Artigos mais recentes recebem peso maior. O fator de decaimento
        RECENCY_DECAY e aplicado por dia de idade.

        Args:
            article: Dict do artigo com campo 'timestamp' (opcional)
            now: Datetime de referencia (agora)

        Returns:
            Peso entre 0 e 1 (1 = artigo de hoje)
        """
        timestamp = article.get("timestamp")
        if timestamp is None:
            # Sem timestamp, assume peso maximo
            return 1.0

        # Converter para datetime se necessario
        if isinstance(timestamp, (int, float)):
            ts = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        elif isinstance(timestamp, str):
            try:
                ts = pd.to_datetime(timestamp, utc=True).to_pydatetime()
            except Exception:
                return 1.0
        elif isinstance(timestamp, pd.Timestamp):
            ts = timestamp.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        elif isinstance(timestamp, datetime):
            ts = timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        else:
            return 1.0

        # Dias de diferenca
        age_days = max((now - ts).total_seconds() / 86400, 0)

        # Decaimento exponencial
        return self.RECENCY_DECAY ** age_days

    def _compute_entropy(self, scores: dict[str, float]) -> float:
        """Calcula entropia de Shannon das narrativas para medir diversidade.

        Entropia alta = muitas narrativas ativas (mercado disperso).
        Entropia baixa = uma narrativa domina (mercado focado).

        Args:
            scores: Dicionario {narrativa: score_normalizado}

        Returns:
            Entropia normalizada entre 0 e 1
        """
        values = list(scores.values())
        total = sum(values)

        if total <= 0:
            return 0.0

        # Normalizar para distribuicao de probabilidade
        probs = [v / total for v in values if v > 0]

        if not probs:
            return 0.0

        # Entropia de Shannon
        entropy = -sum(p * math.log2(p) for p in probs)

        # Normalizar pela entropia maxima (log2 do numero de categorias)
        max_entropy = math.log2(len(self.NARRATIVE_KEYWORDS))
        if max_entropy > 0:
            entropy /= max_entropy

        return entropy

    def _compute_momentum(self, current_date_str: str, days: int = 7) -> float:
        """Calcula momentum da narrativa dominante ao longo de um periodo.

        Momentum positivo = narrativa dominante esta ganhando forca.
        Momentum negativo = narrativa dominante esta perdendo forca.

        Args:
            current_date_str: Data atual no formato YYYY-MM-DD
            days: Janela de dias para comparacao

        Returns:
            Variacao do score da narrativa dominante (delta)
        """
        current_scores = self._history.get(current_date_str)
        if not current_scores:
            return 0.0

        # Encontrar data de referencia
        try:
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
            past_date = current_date - timedelta(days=days)
            past_date_str = past_date.strftime("%Y-%m-%d")
        except ValueError:
            return 0.0

        past_scores = self._history.get(past_date_str)
        if not past_scores:
            # Tentar encontrar a data mais proxima no historico dentro da janela
            past_scores = self._find_closest_history(past_date_str, tolerance_days=3)
            if not past_scores:
                return 0.0

        # Calcular delta da narrativa dominante atual
        dominant = max(current_scores, key=current_scores.get)
        current_val = current_scores.get(dominant, 0.0)
        past_val = past_scores.get(dominant, 0.0)

        return current_val - past_val

    def _find_closest_history(
        self, target_date_str: str, tolerance_days: int = 3
    ) -> dict[str, float] | None:
        """Busca o historico mais proximo de uma data alvo dentro da tolerancia.

        Args:
            target_date_str: Data alvo no formato YYYY-MM-DD
            tolerance_days: Tolerancia em dias para busca

        Returns:
            Scores da data mais proxima ou None
        """
        try:
            target = datetime.strptime(target_date_str, "%Y-%m-%d")
        except ValueError:
            return None

        best_match = None
        best_diff = float("inf")

        for date_str in self._history:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                diff = abs((dt - target).days)
                if diff <= tolerance_days and diff < best_diff:
                    best_diff = diff
                    best_match = date_str
            except ValueError:
                continue

        if best_match:
            return self._history[best_match]
        return None

    def _compute_alignment(self, scores: dict[str, float], coin: str) -> float:
        """Calcula alinhamento entre narrativas ativas e a moeda especifica.

        Score alto = as narrativas dominantes sao relevantes para a moeda.
        Score baixo = a moeda nao se beneficia das narrativas atuais.

        Args:
            scores: Scores de narrativa atuais
            coin: Simbolo da moeda

        Returns:
            Score de alinhamento entre 0 e 1
        """
        aligned_narratives = self.COIN_NARRATIVE_ALIGNMENT.get(
            coin, list(self.NARRATIVE_KEYWORDS.keys())
        )

        if not aligned_narratives:
            return 0.0

        # Media dos scores das narrativas alinhadas com a moeda
        aligned_scores = [
            scores.get(narr, 0.0) for narr in aligned_narratives
        ]

        if not aligned_scores:
            return 0.0

        # Media ponderada — narrativas mais relevantes tem peso maior
        weights = [1.0 / (i + 1) for i in range(len(aligned_scores))]
        total_weight = sum(weights)

        alignment = sum(s * w for s, w in zip(aligned_scores, weights))
        return alignment / total_weight if total_weight > 0 else 0.0

    def _compute_contrarian_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula sinal contrario: divergencia entre preco e narrativa.

        Quando o preco sobe mas a narrativa enfraquece (ou vice-versa),
        isso pode indicar uma reversao iminente.

        - Valor positivo: preco subindo com narrativa fraca (possivel topo)
        - Valor negativo: preco caindo com narrativa forte (possivel fundo)

        Args:
            df: DataFrame com colunas 'close' e 'narrative_dominant_score'

        Returns:
            DataFrame com coluna 'narrative_contrarian' adicionada
        """
        if "close" not in df.columns:
            df["narrative_contrarian"] = 0.0
            return df

        # Retorno percentual do preco (7 dias)
        price_returns = df["close"].pct_change(periods=7).fillna(0.0)

        # Normalizar retornos para [-1, 1] usando tanh
        price_signal = np.tanh(price_returns * 10)  # Escalar para sensibilidade

        # Score de narrativa (ja entre 0 e 1)
        narrative_score = df.get("narrative_dominant_score", pd.Series(0.0, index=df.index))
        narrative_score = narrative_score.fillna(0.0)

        # Divergencia: preco vs narrativa
        # Positivo = preco subindo sem suporte de narrativa
        # Negativo = preco caindo apesar de narrativa forte
        df["narrative_contrarian"] = price_signal - narrative_score

        # Clipar para [-1, 1]
        df["narrative_contrarian"] = df["narrative_contrarian"].clip(-1.0, 1.0)

        return df

    def _fetch_articles(self, coin: str) -> list[dict]:
        """Busca artigos de noticias via scraper interno.

        Args:
            coin: Simbolo da moeda

        Returns:
            Lista de artigos ou lista vazia se scraper nao disponivel
        """
        if self._scraper is None:
            logger.warning("Scraper nao disponivel — retornando lista vazia")
            return []

        try:
            articles = self._scraper.fetch_all_news(coin=coin)
            logger.info(f"Artigos coletados para narrativa de {coin}: {len(articles)}")
            return articles
        except Exception as e:
            logger.error(f"Erro ao buscar artigos para {coin}: {e}")
            return []

    @staticmethod
    def _group_articles_by_date(articles: list[dict]) -> dict[str, list[dict]]:
        """Agrupa artigos por data (YYYY-MM-DD).

        Args:
            articles: Lista de artigos com campo 'timestamp'

        Returns:
            Dicionario {data_str: [artigos_do_dia]}
        """
        grouped: dict[str, list[dict]] = {}

        for article in articles:
            timestamp = article.get("timestamp")
            if timestamp is None:
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            elif isinstance(timestamp, datetime):
                date_str = timestamp.strftime("%Y-%m-%d")
            elif isinstance(timestamp, pd.Timestamp):
                date_str = timestamp.strftime("%Y-%m-%d")
            elif isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%d")
            elif isinstance(timestamp, str):
                try:
                    dt = pd.to_datetime(timestamp, utc=True)
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            else:
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            if date_str not in grouped:
                grouped[date_str] = []
            grouped[date_str].append(article)

        return grouped

    def _add_empty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas de features de narrativa com valores zero.

        Args:
            df: DataFrame original

        Returns:
            DataFrame com features zeradas
        """
        for col in self.get_feature_names():
            df[col] = 0.0
        return df
