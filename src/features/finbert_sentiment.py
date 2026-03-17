"""Features de sentimento NLP usando FinBERT e Twitter-RoBERTa (Phase 6).

Utiliza modelos pre-treinados de NLP para extrair sentimento de textos
financeiros (manchetes, tweets, noticias). Quando os modelos nao estao
disponiveis, gera features proxy a partir de dados OHLCV.

Modelos:
    - ProsusAI/finbert: sentimento financeiro (positivo/negativo/neutro)
    - cardiffnlp/twitter-roberta-base-sentiment-latest: sentimento social
"""

import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Importacao segura do transformers
# ---------------------------------------------------------------------------

_HAS_TRANSFORMERS = False
_pipeline_fn = None
_AutoTokenizer = None
_AutoModelForSequenceClassification = None

try:
    from transformers import (
        pipeline as _pipeline_fn,
        AutoTokenizer as _AutoTokenizer,
        AutoModelForSequenceClassification as _AutoModelForSequenceClassification,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    logger.debug(
        "transformers nao disponivel (instale com: pip install transformers torch). "
        "Features NLP serao geradas como proxies a partir de OHLCV."
    )


# ---------------------------------------------------------------------------
# Wrapper FinBERT
# ---------------------------------------------------------------------------

class FinBERTWrapper:
    """Wrapper para o modelo ProsusAI/finbert de sentimento financeiro.

    Realiza lazy-loading do modelo e cache de resultados.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self._pipe = None
        self._cache: dict[str, dict[str, float]] = {}

    def _load_model(self) -> None:
        """Carrega o modelo sob demanda."""
        if self._pipe is not None:
            return
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers nao instalado. Instale com: pip install transformers torch"
            )
        logger.info("Carregando modelo FinBERT (%s)...", self.MODEL_NAME)
        self._pipe = _pipeline_fn(
            "sentiment-analysis",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            device=self.device,
            truncation=True,
            max_length=512,
        )
        logger.info("Modelo FinBERT carregado com sucesso.")

    def _cache_key(self, text: str) -> str:
        """Gera chave de cache para um texto."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Gera scores de sentimento para uma lista de textos.

        Parameters
        ----------
        texts : list[str]
            Lista de textos financeiros (manchetes, noticias).

        Returns
        -------
        list[dict[str, float]]
            Lista de dicts com chaves 'positive', 'negative', 'neutral', 'sentiment'.
            'sentiment' eh um score composto em [-1, 1].
        """
        self._load_model()

        results: list[dict[str, float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Verificar cache
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append({})  # placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Processar textos nao cacheados em batches
        if uncached_texts:
            logger.info(
                "FinBERT: processando %d textos (%d do cache).",
                len(uncached_texts),
                len(texts) - len(uncached_texts),
            )
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[batch_start : batch_start + self.batch_size]
                try:
                    raw_outputs = self._pipe(batch)
                except Exception as e:
                    logger.error("Erro no FinBERT ao processar batch: %s", e)
                    # Fallback: neutro
                    raw_outputs = [
                        {"label": "neutral", "score": 1.0} for _ in batch
                    ]

                for j, output in enumerate(raw_outputs):
                    idx = uncached_indices[batch_start + j]
                    scores = self._parse_finbert_output(output)
                    key = self._cache_key(texts[idx])
                    self._cache[key] = scores
                    results[idx] = scores

        return results

    @staticmethod
    def _parse_finbert_output(output: dict) -> dict[str, float]:
        """Converte saida do pipeline FinBERT em scores normalizados."""
        label = output.get("label", "neutral").lower()
        score = output.get("score", 0.0)

        positive = score if label == "positive" else 0.0
        negative = score if label == "negative" else 0.0
        neutral = score if label == "neutral" else 0.0

        # Score composto: positivo - negativo, ponderado pela confianca
        sentiment = positive - negative

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "sentiment": sentiment,
        }


# ---------------------------------------------------------------------------
# Wrapper Twitter-RoBERTa
# ---------------------------------------------------------------------------

class TwitterRoBERTaWrapper:
    """Wrapper para cardiffnlp/twitter-roberta-base-sentiment-latest.

    Modelo otimizado para sentimento em textos de redes sociais (tweets).
    Realiza lazy-loading e cache de resultados.
    """

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self._pipe = None
        self._cache: dict[str, dict[str, float]] = {}

    def _load_model(self) -> None:
        """Carrega o modelo sob demanda."""
        if self._pipe is not None:
            return
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers nao instalado. Instale com: pip install transformers torch"
            )
        logger.info("Carregando modelo Twitter-RoBERTa (%s)...", self.MODEL_NAME)
        self._pipe = _pipeline_fn(
            "sentiment-analysis",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            device=self.device,
            truncation=True,
            max_length=512,
            top_k=3,
        )
        logger.info("Modelo Twitter-RoBERTa carregado com sucesso.")

    def _cache_key(self, text: str) -> str:
        """Gera chave de cache para um texto."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Gera scores de sentimento para uma lista de textos.

        Parameters
        ----------
        texts : list[str]
            Lista de textos sociais (tweets, posts).

        Returns
        -------
        list[dict[str, float]]
            Lista de dicts com chaves 'positive', 'negative', 'neutral', 'sentiment'.
        """
        self._load_model()

        results: list[dict[str, float]] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append({})
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            logger.info(
                "Twitter-RoBERTa: processando %d textos (%d do cache).",
                len(uncached_texts),
                len(texts) - len(uncached_texts),
            )
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[batch_start : batch_start + self.batch_size]
                try:
                    raw_outputs = self._pipe(batch)
                except Exception as e:
                    logger.error("Erro no Twitter-RoBERTa ao processar batch: %s", e)
                    raw_outputs = [
                        [
                            {"label": "neutral", "score": 1.0},
                            {"label": "positive", "score": 0.0},
                            {"label": "negative", "score": 0.0},
                        ]
                        for _ in batch
                    ]

                for j, output in enumerate(raw_outputs):
                    idx = uncached_indices[batch_start + j]
                    scores = self._parse_roberta_output(output)
                    key = self._cache_key(texts[idx])
                    self._cache[key] = scores
                    results[idx] = scores

        return results

    @staticmethod
    def _parse_roberta_output(output) -> dict[str, float]:
        """Converte saida do pipeline RoBERTa em scores normalizados."""
        positive = 0.0
        negative = 0.0
        neutral = 0.0

        if isinstance(output, list):
            for item in output:
                label = item.get("label", "").lower()
                score = item.get("score", 0.0)
                if "positive" in label:
                    positive = score
                elif "negative" in label:
                    negative = score
                elif "neutral" in label:
                    neutral = score
        elif isinstance(output, dict):
            label = output.get("label", "neutral").lower()
            score = output.get("score", 0.0)
            if "positive" in label:
                positive = score
            elif "negative" in label:
                negative = score
            else:
                neutral = score

        sentiment = positive - negative

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "sentiment": sentiment,
        }


# ---------------------------------------------------------------------------
# Ensemble de sentimento
# ---------------------------------------------------------------------------

class SentimentEnsemble:
    """Combina FinBERT e Twitter-RoBERTa em um score ensemble.

    Pesos padrao: FinBERT 0.6 (especializado financeiro), RoBERTa 0.4 (social).
    """

    def __init__(
        self,
        finbert_weight: float = 0.6,
        roberta_weight: float = 0.4,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.finbert_weight = finbert_weight
        self.roberta_weight = roberta_weight
        self.finbert = FinBERTWrapper(device=device, batch_size=batch_size)
        self.roberta = TwitterRoBERTaWrapper(device=device, batch_size=batch_size)

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Gera scores ensemble para uma lista de textos.

        Returns
        -------
        list[dict[str, float]]
            Cada dict contem:
            - finbert_sentiment: score FinBERT [-1, 1]
            - finbert_positive: probabilidade positiva
            - finbert_negative: probabilidade negativa
            - twitter_sentiment: score RoBERTa [-1, 1]
            - ensemble_sentiment: media ponderada dos modelos
            - forward_looking_score: indicador de perspectiva futura
        """
        finbert_results = self.finbert.predict(texts)
        roberta_results = self.roberta.predict(texts)

        ensemble_results = []
        for fb, rb in zip(finbert_results, roberta_results):
            ensemble_sent = (
                fb["sentiment"] * self.finbert_weight
                + rb["sentiment"] * self.roberta_weight
            )

            # Forward-looking score: combina confianca e concordancia dos modelos.
            # Alta concordancia + alta confianca = forte sinal direcional.
            agreement = 1.0 - abs(fb["sentiment"] - rb["sentiment"])
            avg_confidence = (
                max(fb["positive"], fb["negative"], fb["neutral"])
                + max(rb["positive"], rb["negative"], rb["neutral"])
            ) / 2.0
            forward_looking = ensemble_sent * agreement * avg_confidence

            ensemble_results.append({
                "finbert_sentiment": fb["sentiment"],
                "finbert_positive": fb["positive"],
                "finbert_negative": fb["negative"],
                "twitter_sentiment": rb["sentiment"],
                "ensemble_sentiment": ensemble_sent,
                "forward_looking_score": forward_looking,
            })

        return ensemble_results


# ---------------------------------------------------------------------------
# Feature transformer principal
# ---------------------------------------------------------------------------

class NLPSentimentFeatures:
    """Gera features de sentimento NLP para o pipeline de features.

    Se dados de texto estiverem disponiveis no DataFrame (colunas como
    'headline', 'tweet', 'news_text'), utiliza FinBERT e Twitter-RoBERTa.
    Caso contrario, gera 8 features proxy a partir de dados OHLCV.

    Features geradas (8 no total):
        1. nlp_finbert_sentiment     - Score FinBERT composto [-1, 1]
        2. nlp_finbert_positive      - Probabilidade de sentimento positivo
        3. nlp_finbert_negative      - Probabilidade de sentimento negativo
        4. nlp_twitter_sentiment     - Score Twitter-RoBERTa [-1, 1]
        5. nlp_ensemble_sentiment    - Media ponderada dos modelos
        6. nlp_forward_looking_score - Indicador de perspectiva futura
        7. nlp_sentiment_proxy       - Proxy baseado em momentum de preco
        8. nlp_volatility_sentiment  - Proxy baseado em volatilidade (medo/ganancia)
    """

    # Colunas de texto reconhecidas pelo transformer
    TEXT_COLUMNS = [
        "headline", "headlines", "news_text", "news_title",
        "tweet", "tweets", "text", "title", "description",
    ]

    # Nomes de todas as features geradas
    FEATURE_NAMES = [
        "nlp_finbert_sentiment",
        "nlp_finbert_positive",
        "nlp_finbert_negative",
        "nlp_twitter_sentiment",
        "nlp_ensemble_sentiment",
        "nlp_forward_looking_score",
        "nlp_sentiment_proxy",
        "nlp_volatility_sentiment",
    ]

    def __init__(
        self,
        lookback: int = 20,
        device: str = "cpu",
        batch_size: int = 32,
        finbert_weight: float = 0.6,
        roberta_weight: float = 0.4,
    ):
        self.lookback = lookback
        self.device = device
        self.batch_size = batch_size
        self._ensemble: Optional[SentimentEnsemble] = None
        self._finbert_weight = finbert_weight
        self._roberta_weight = roberta_weight

    def _get_ensemble(self) -> SentimentEnsemble:
        """Lazy-load do ensemble de modelos."""
        if self._ensemble is None:
            self._ensemble = SentimentEnsemble(
                finbert_weight=self._finbert_weight,
                roberta_weight=self._roberta_weight,
                device=self.device,
                batch_size=self.batch_size,
            )
        return self._ensemble

    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Encontra a primeira coluna de texto disponivel no DataFrame."""
        for col in self.TEXT_COLUMNS:
            if col in df.columns:
                return col
        return None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de sentimento NLP ao DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame com dados OHLCV e opcionalmente colunas de texto.
            Colunas esperadas para proxy: 'close', 'volume', 'log_return'.

        Returns
        -------
        pd.DataFrame
            Copia do DataFrame com 8 features NLP adicionadas.
        """
        df = df.copy()

        text_col = self._find_text_column(df)

        if text_col is not None and _HAS_TRANSFORMERS:
            df = self._generate_nlp_features(df, text_col)
        else:
            if text_col is not None and not _HAS_TRANSFORMERS:
                logger.warning(
                    "Coluna de texto '%s' encontrada, mas transformers nao instalado. "
                    "Gerando features proxy a partir de OHLCV.",
                    text_col,
                )
            else:
                logger.info(
                    "Nenhuma coluna de texto encontrada. "
                    "Gerando features proxy de sentimento a partir de OHLCV."
                )
            df = self._generate_proxy_features(df)

        return df

    # ------------------------------------------------------------------
    # Features via modelos NLP
    # ------------------------------------------------------------------

    def _generate_nlp_features(
        self, df: pd.DataFrame, text_col: str
    ) -> pd.DataFrame:
        """Gera features usando FinBERT e Twitter-RoBERTa.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contendo a coluna de texto.
        text_col : str
            Nome da coluna com textos para analise.
        """
        texts = df[text_col].fillna("").astype(str).tolist()

        # Filtrar textos vazios - substituir por texto neutro
        processed_texts = [
            t if t.strip() else "neutral market conditions"
            for t in texts
        ]

        try:
            ensemble = self._get_ensemble()
            results = ensemble.predict(processed_texts)

            df["nlp_finbert_sentiment"] = np.array(
                [r["finbert_sentiment"] for r in results], dtype=np.float32
            )
            df["nlp_finbert_positive"] = np.array(
                [r["finbert_positive"] for r in results], dtype=np.float32
            )
            df["nlp_finbert_negative"] = np.array(
                [r["finbert_negative"] for r in results], dtype=np.float32
            )
            df["nlp_twitter_sentiment"] = np.array(
                [r["twitter_sentiment"] for r in results], dtype=np.float32
            )
            df["nlp_ensemble_sentiment"] = np.array(
                [r["ensemble_sentiment"] for r in results], dtype=np.float32
            )
            df["nlp_forward_looking_score"] = np.array(
                [r["forward_looking_score"] for r in results], dtype=np.float32
            )

            # Proxy features complementares a partir de OHLCV
            df = self._add_ohlcv_sentiment_proxies(df)

            logger.info(
                "Features NLP geradas com sucesso via FinBERT + Twitter-RoBERTa "
                "(%d textos processados).",
                len(processed_texts),
            )

        except Exception as e:
            logger.error(
                "Erro ao gerar features NLP: %s. Usando features proxy.", e
            )
            df = self._generate_proxy_features(df)

        return df

    # ------------------------------------------------------------------
    # Features proxy a partir de OHLCV
    # ------------------------------------------------------------------

    def _generate_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera 8 features proxy de sentimento a partir de dados OHLCV.

        Quando modelos NLP nao estao disponiveis, utiliza sinais de preco,
        volatilidade e volume para aproximar sentimento do mercado.
        """
        lb = self.lookback

        # Garantir que temos as colunas necessarias
        if "log_return" not in df.columns:
            if "close" in df.columns:
                df["log_return"] = np.log(
                    df["close"] / df["close"].shift(1)
                )
                logger.info("Calculando log_return a partir de close.")
            else:
                logger.warning(
                    "Colunas 'close' e 'log_return' nao encontradas. "
                    "Preenchendo features NLP com zeros."
                )
                for feat in self.FEATURE_NAMES:
                    df[feat] = np.float32(0.0)
                return df

        ret = df["log_return"]

        # 1. nlp_sentiment_proxy: momentum de preco como proxy de sentimento
        #    Retornos positivos acumulados = sentimento positivo
        momentum = ret.rolling(lb).sum()
        momentum_std = momentum.rolling(lb * 2, min_periods=lb).std()
        df["nlp_sentiment_proxy"] = np.where(
            momentum_std > 0,
            (momentum / momentum_std.replace(0, np.nan)).clip(-3, 3) / 3.0,
            0.0,
        ).astype(np.float32)

        # 2. nlp_volatility_sentiment: volatilidade como proxy medo/ganancia
        #    Alta volatilidade = medo (-1), baixa volatilidade = ganancia (+1)
        realized_vol = ret.rolling(lb).std()
        vol_ma = realized_vol.rolling(lb * 2, min_periods=lb).mean()
        vol_std = realized_vol.rolling(lb * 2, min_periods=lb).std()
        vol_z = np.where(
            vol_std > 0,
            (realized_vol - vol_ma) / vol_std.replace(0, np.nan),
            0.0,
        )
        # Inverter: alta vol = medo (negativo), baixa vol = ganancia (positivo)
        df["nlp_volatility_sentiment"] = (
            np.clip(-vol_z, -1, 1)
        ).astype(np.float32)

        # 3. nlp_finbert_sentiment: proxy via tendencia suavizada
        #    EMA dos retornos simula avaliacao de sentimento financeiro
        ema_fast = ret.ewm(span=lb // 2, min_periods=1).mean()
        ema_slow = ret.ewm(span=lb, min_periods=1).mean()
        finbert_proxy = (ema_fast + ema_slow) / 2.0
        finbert_std = finbert_proxy.rolling(lb * 2, min_periods=lb).std()
        df["nlp_finbert_sentiment"] = np.where(
            finbert_std > 0,
            (finbert_proxy / finbert_std.replace(0, np.nan)).clip(-1, 1),
            0.0,
        ).astype(np.float32)

        # 4. nlp_finbert_positive: proporcao de retornos positivos na janela
        positive_ratio = (ret > 0).astype(float).rolling(lb).mean()
        df["nlp_finbert_positive"] = positive_ratio.astype(np.float32)

        # 5. nlp_finbert_negative: proporcao de retornos negativos na janela
        negative_ratio = (ret < 0).astype(float).rolling(lb).mean()
        df["nlp_finbert_negative"] = negative_ratio.astype(np.float32)

        # 6. nlp_twitter_sentiment: proxy via volume e momentum conjunto
        #    Volume spike + retorno positivo = sentimento social positivo
        if "volume" in df.columns:
            vol = df["volume"]
            vol_ma = vol.rolling(lb).mean()
            vol_ratio = np.where(
                vol_ma > 0,
                vol / vol_ma.replace(0, np.nan),
                1.0,
            )
            # Sentimento social: direcao do retorno ponderada pela surpresa de volume
            vol_surprise = np.clip(np.log1p(np.maximum(vol_ratio - 1.0, 0.0)), 0, 2)
            twitter_raw = np.sign(ret) * vol_surprise
            twitter_smooth = pd.Series(twitter_raw, index=df.index).rolling(
                lb // 2, min_periods=1
            ).mean()
            twitter_std = twitter_smooth.rolling(lb * 2, min_periods=lb).std()
            df["nlp_twitter_sentiment"] = np.where(
                twitter_std > 0,
                (twitter_smooth / twitter_std.replace(0, np.nan)).clip(-1, 1),
                0.0,
            ).astype(np.float32)
        else:
            # Sem volume, usar momentum puro
            df["nlp_twitter_sentiment"] = df["nlp_sentiment_proxy"].copy()

        # 7. nlp_ensemble_sentiment: media ponderada dos proxies
        df["nlp_ensemble_sentiment"] = (
            df["nlp_finbert_sentiment"] * self._finbert_weight
            + df["nlp_twitter_sentiment"] * (1 - self._finbert_weight)
        ).astype(np.float32)

        # 8. nlp_forward_looking_score: aceleracao do sentimento
        #    Mudanca no sentimento ensemble indica perspectiva futura
        ens = df["nlp_ensemble_sentiment"]
        ens_diff = ens.diff(max(lb // 4, 1))
        ens_diff_std = ens_diff.rolling(lb * 2, min_periods=lb).std()
        df["nlp_forward_looking_score"] = np.where(
            ens_diff_std > 0,
            (ens_diff / ens_diff_std.replace(0, np.nan)).clip(-1, 1),
            0.0,
        ).astype(np.float32)

        logger.info(
            "Features proxy de sentimento NLP geradas a partir de dados OHLCV "
            "(%d registros).",
            len(df),
        )

        return df

    def _add_ohlcv_sentiment_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features proxy complementares quando NLP esta ativo.

        Mesmo com modelos NLP, geramos proxies de OHLCV para as features
        7 e 8, pois capturam sinais complementares.
        """
        lb = self.lookback

        if "log_return" not in df.columns:
            if "close" in df.columns:
                df["log_return"] = np.log(
                    df["close"] / df["close"].shift(1)
                )
            else:
                df["nlp_sentiment_proxy"] = np.float32(0.0)
                df["nlp_volatility_sentiment"] = np.float32(0.0)
                return df

        ret = df["log_return"]

        # nlp_sentiment_proxy: momentum normalizado
        momentum = ret.rolling(lb).sum()
        momentum_std = momentum.rolling(lb * 2, min_periods=lb).std()
        df["nlp_sentiment_proxy"] = np.where(
            momentum_std > 0,
            (momentum / momentum_std.replace(0, np.nan)).clip(-3, 3) / 3.0,
            0.0,
        ).astype(np.float32)

        # nlp_volatility_sentiment: inverso da volatilidade normalizada
        realized_vol = ret.rolling(lb).std()
        vol_ma = realized_vol.rolling(lb * 2, min_periods=lb).mean()
        vol_std = realized_vol.rolling(lb * 2, min_periods=lb).std()
        vol_z = np.where(
            vol_std > 0,
            (realized_vol - vol_ma) / vol_std.replace(0, np.nan),
            0.0,
        )
        df["nlp_volatility_sentiment"] = (
            np.clip(-vol_z, -1, 1)
        ).astype(np.float32)

        return df

    def get_feature_names(self) -> list[str]:
        """Retorna os nomes de todas as features geradas por este modulo."""
        return list(self.FEATURE_NAMES)
