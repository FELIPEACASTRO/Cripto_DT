"""Analise de sentimento financeiro multilingual para mercados cripto globais.

Suporte a Chines, Coreano, Japones, Vietnamita, Indonesio, Arabe e Ingles
usando modelos HuggingFace especializados por idioma com carregamento lazy.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tentativa de importar transformers
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers nao disponivel — sentimento multilingual retornara scores neutros"
    )

# Mapeamento de idiomas para modelos HuggingFace
LANGUAGE_MODELS: dict[str, dict[str, str]] = {
    "zh": {
        "model": "yiyanghkust/finbert-tone-chinese",
        "license": "Apache-2.0",
        "type": "finbert",
    },
    "ko": {
        "model": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        "license": "Apache-2.0",
        "type": "xlm-roberta",
    },
    "ja": {
        "model": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        "license": "Apache-2.0",
        "type": "xlm-roberta",
    },
    "vi": {
        "model": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        "license": "Apache-2.0",
        "type": "xlm-roberta",
        "primary": "vinai/phobert-base-v2",
    },
    "id": {
        "model": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        "license": "Apache-2.0",
        "type": "xlm-roberta",
        "primary": "indobenchmark/indobert-base-p2",
    },
    "ar": {
        "model": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        "license": "Apache-2.0",
        "type": "xlm-roberta",
    },
    "en": {
        "model": "ProsusAI/finbert",
        "license": "Apache-2.0",
        "type": "finbert",
    },
}

# Pesos regionais para calculo do sentimento global ponderado
REGIONAL_WEIGHTS: dict[str, float] = {
    "zh": 0.25,
    "ko": 0.15,
    "ja": 0.15,
    "vi": 0.05,
    "id": 0.05,
    "ar": 0.10,
    "en": 0.25,
}

# Mapeamento de labels de sentimento para valores numericos
_LABEL_MAP_FINBERT: dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}

_LABEL_MAP_XLM: dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    # Alguns modelos usam indices numericos
    "LABEL_0": -1.0,  # negative
    "LABEL_1": 0.0,  # neutral
    "LABEL_2": 1.0,  # positive
}


def _neutral_result(language: str = "unknown") -> dict[str, Any]:
    """Retorna resultado neutro para fallback."""
    return {
        "sentiment": 0.0,
        "confidence": 0.0,
        "language": language,
    }


class MultilingualSentimentAnalyzer:
    """Analisa sentimento financeiro em multiplos idiomas para mercados cripto.

    Utiliza modelos HuggingFace especializados por idioma com carregamento lazy.
    Suporta Chines, Coreano, Japones, Vietnamita, Indonesio, Arabe e Ingles.

    Se transformers nao estiver disponivel, retorna scores neutros (degradacao graciosa).
    """

    def __init__(self, device: str = "cpu", max_length: int = 512):
        """Inicializa o analisador multilingual.

        Args:
            device: Dispositivo para inferencia ('cpu', 'cuda', 'mps').
            max_length: Comprimento maximo de tokens por texto.
        """
        self._device = device
        self._max_length = max_length
        self._pipelines: dict[str, Any] = {}
        self._loaded_models: set[str] = set()

    def _get_pipeline(self, language: str) -> Any | None:
        """Carrega e retorna o pipeline de sentimento para o idioma especificado.

        O modelo e carregado apenas na primeira chamada (lazy loading).

        Args:
            language: Codigo do idioma (ex: 'zh', 'ko', 'en').

        Returns:
            Pipeline de sentimento ou None se nao disponivel.
        """
        if not _TRANSFORMERS_AVAILABLE:
            return None

        if language in self._pipelines:
            return self._pipelines[language]

        lang_config = LANGUAGE_MODELS.get(language, LANGUAGE_MODELS["en"])
        model_name = lang_config["model"]

        # Para idiomas com modelo primario (vi, id), tentar primario primeiro
        primary_model = lang_config.get("primary")
        if primary_model and primary_model not in self._loaded_models:
            try:
                logger.info(
                    f"Carregando modelo primario para {language}: {primary_model}"
                )
                pipe = pipeline(
                    "sentiment-analysis",
                    model=primary_model,
                    tokenizer=primary_model,
                    device=self._device if self._device != "cpu" else -1,
                    truncation=True,
                    max_length=self._max_length,
                )
                self._pipelines[language] = pipe
                self._loaded_models.add(primary_model)
                logger.info(f"Modelo primario carregado para {language}: {primary_model}")
                return pipe
            except Exception as e:
                logger.warning(
                    f"Falha ao carregar modelo primario {primary_model} para {language}: {e}. "
                    f"Usando fallback multilingual: {model_name}"
                )

        # Carregar modelo principal (ou fallback multilingual)
        if model_name in self._loaded_models:
            # Reusar pipeline ja carregado para mesmo modelo (ex: xlm-roberta compartilhado)
            for lang, pipe in self._pipelines.items():
                existing_config = LANGUAGE_MODELS.get(lang, LANGUAGE_MODELS["en"])
                if existing_config["model"] == model_name:
                    self._pipelines[language] = pipe
                    return pipe

        try:
            logger.info(f"Carregando modelo para {language}: {model_name}")
            pipe = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=self._device if self._device != "cpu" else -1,
                truncation=True,
                max_length=self._max_length,
            )
            self._pipelines[language] = pipe
            self._loaded_models.add(model_name)
            logger.info(f"Modelo carregado para {language}: {model_name}")
            return pipe
        except Exception as e:
            logger.error(f"Erro ao carregar modelo para {language} ({model_name}): {e}")
            self._pipelines[language] = None
            return None

    def _parse_result(self, result: dict[str, Any], language: str) -> dict[str, Any]:
        """Converte resultado do pipeline para formato padronizado.

        Args:
            result: Resultado bruto do pipeline HuggingFace.
            language: Codigo do idioma.

        Returns:
            Dicionario com sentiment (-1 a 1), confidence (0 a 1) e language.
        """
        label = result.get("label", "neutral").lower()
        score = result.get("score", 0.0)

        lang_config = LANGUAGE_MODELS.get(language, LANGUAGE_MODELS["en"])
        model_type = lang_config.get("type", "finbert")

        if model_type == "finbert":
            label_map = _LABEL_MAP_FINBERT
        else:
            label_map = _LABEL_MAP_XLM

        sentiment_value = label_map.get(label, 0.0)
        # Escalar pelo score de confianca do modelo
        sentiment = sentiment_value * score

        return {
            "sentiment": float(np.clip(sentiment, -1.0, 1.0)),
            "confidence": float(np.clip(score, 0.0, 1.0)),
            "language": language,
        }

    def _detect_language(self, text: str) -> str:
        """Deteccao simples de idioma baseada em caracteres Unicode.

        Args:
            text: Texto para detectar idioma.

        Returns:
            Codigo do idioma detectado.
        """
        if not text or not text.strip():
            return "en"

        # Contar caracteres por range Unicode
        cjk_count = 0
        hangul_count = 0
        hiragana_katakana_count = 0
        arabic_count = 0
        vietnamese_marks = 0
        latin_count = 0

        for char in text:
            cp = ord(char)
            if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                cjk_count += 1
            elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
                hangul_count += 1
            elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
                hiragana_katakana_count += 1
            elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
                arabic_count += 1
            elif cp in (0x01A0, 0x01A1, 0x01AF, 0x01B0, 0x0110, 0x0111):
                vietnamese_marks += 1
            elif 0x0041 <= cp <= 0x007A:
                latin_count += 1

        total = max(len(text.strip()), 1)

        # Priorizar deteccao por scripts nao-latinos
        if hangul_count / total > 0.1:
            return "ko"
        if hiragana_katakana_count / total > 0.1:
            return "ja"
        if cjk_count / total > 0.1:
            return "zh"
        if arabic_count / total > 0.1:
            return "ar"
        if vietnamese_marks > 0:
            return "vi"

        # Heuristica para indonesio: palavras-chave comuns
        indonesian_keywords = {"yang", "dan", "untuk", "dengan", "dari", "ini", "itu"}
        words = set(text.lower().split())
        if len(words & indonesian_keywords) >= 2:
            return "id"

        return "en"

    def analyze(self, text: str, language: str = "auto") -> dict[str, Any]:
        """Analisa sentimento de um texto.

        Args:
            text: Texto para analise de sentimento.
            language: Codigo do idioma ou 'auto' para deteccao automatica.

        Returns:
            Dicionario com 'sentiment' (-1 a 1), 'confidence' (0 a 1), 'language'.
        """
        if not text or not text.strip():
            return _neutral_result(language if language != "auto" else "unknown")

        if language == "auto":
            language = self._detect_language(text)

        if not _TRANSFORMERS_AVAILABLE:
            logger.debug(
                f"transformers indisponivel, retornando score neutro para {language}"
            )
            return _neutral_result(language)

        pipe = self._get_pipeline(language)
        if pipe is None:
            return _neutral_result(language)

        try:
            results = pipe(text[:self._max_length * 4])  # Truncar texto bruto tambem
            if results and isinstance(results, list):
                return self._parse_result(results[0], language)
            return _neutral_result(language)
        except Exception as e:
            logger.error(f"Erro na analise de sentimento ({language}): {e}")
            return _neutral_result(language)

    def analyze_batch(
        self, texts: list[str], language: str = "auto"
    ) -> list[dict[str, Any]]:
        """Analisa sentimento de multiplos textos em batch.

        Args:
            texts: Lista de textos para analise.
            language: Codigo do idioma ou 'auto' para deteccao por texto.

        Returns:
            Lista de dicionarios com resultados de sentimento.
        """
        if not texts:
            return []

        results: list[dict[str, Any]] = []

        if language != "auto":
            # Batch homogeneo: todos no mesmo idioma
            if not _TRANSFORMERS_AVAILABLE:
                return [_neutral_result(language) for _ in texts]

            pipe = self._get_pipeline(language)
            if pipe is None:
                return [_neutral_result(language) for _ in texts]

            try:
                valid_texts = [t if t and t.strip() else " " for t in texts]
                raw_results = pipe(valid_texts)
                for raw in raw_results:
                    results.append(self._parse_result(raw, language))
            except Exception as e:
                logger.error(f"Erro no batch de sentimento ({language}): {e}")
                results = [_neutral_result(language) for _ in texts]
        else:
            # Batch heterogeneo: detectar idioma por texto
            for text in texts:
                results.append(self.analyze(text, language="auto"))

        return results

    def add_multilingual_features(
        self,
        df: pd.DataFrame,
        coin: str,
        texts: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Adiciona features de sentimento multilingual ao DataFrame.

        Args:
            df: DataFrame com dados de mercado.
            coin: Simbolo da moeda (ex: 'BTC', 'ETH').
            texts: Lista de dicionarios com chaves 'text', 'language' (opcional),
                   'source' (opcional). Cada dict representa um texto de uma
                   comunidade/idioma especifico.

        Returns:
            DataFrame com colunas adicionais de sentimento multilingual.
        """
        # Inicializar colunas com valores neutros
        feature_cols = {
            "ml_sentiment_cn": 0.0,
            "ml_sentiment_kr": 0.0,
            "ml_sentiment_jp": 0.0,
            "ml_sentiment_sea": 0.0,
            "ml_sentiment_global": 0.0,
        }

        for col, default_val in feature_cols.items():
            if col not in df.columns:
                df[col] = default_val

        if not texts:
            logger.debug(f"Nenhum texto fornecido para sentimento multilingual ({coin})")
            return df

        # Agrupar textos por idioma
        lang_texts: dict[str, list[str]] = {}
        for entry in texts:
            text = entry.get("text", "")
            lang = entry.get("language", "auto")
            if lang == "auto":
                lang = self._detect_language(text)
            if lang not in lang_texts:
                lang_texts[lang] = []
            lang_texts[lang].append(text)

        # Analisar sentimento por idioma
        lang_sentiments: dict[str, float] = {}
        for lang, lang_text_list in lang_texts.items():
            if not lang_text_list:
                continue
            batch_results = self.analyze_batch(lang_text_list, language=lang)
            sentiments = [r["sentiment"] for r in batch_results if r["confidence"] > 0]
            if sentiments:
                lang_sentiments[lang] = float(np.mean(sentiments))

        # Mapear para colunas de features
        cn_score = lang_sentiments.get("zh", 0.0)
        kr_score = lang_sentiments.get("ko", 0.0)
        jp_score = lang_sentiments.get("ja", 0.0)

        # SEA: media de vi e id
        sea_scores = [
            lang_sentiments[lang]
            for lang in ("vi", "id")
            if lang in lang_sentiments
        ]
        sea_score = float(np.mean(sea_scores)) if sea_scores else 0.0

        # Global: media ponderada de todos os idiomas disponiveis
        weighted_sum = 0.0
        total_weight = 0.0
        for lang, sentiment in lang_sentiments.items():
            weight = REGIONAL_WEIGHTS.get(lang, 0.05)
            weighted_sum += sentiment * weight
            total_weight += weight

        global_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Atribuir valores ao DataFrame (broadcast para todas as linhas)
        df["ml_sentiment_cn"] = cn_score
        df["ml_sentiment_kr"] = kr_score
        df["ml_sentiment_jp"] = jp_score
        df["ml_sentiment_sea"] = sea_score
        df["ml_sentiment_global"] = global_score

        logger.info(
            f"Features multilingual adicionadas para {coin}: "
            f"CN={cn_score:.3f}, KR={kr_score:.3f}, JP={jp_score:.3f}, "
            f"SEA={sea_score:.3f}, Global={global_score:.3f}"
        )

        return df

    def get_supported_languages(self) -> list[str]:
        """Retorna lista de idiomas suportados.

        Returns:
            Lista de codigos de idioma.
        """
        return list(LANGUAGE_MODELS.keys())

    def get_loaded_models(self) -> set[str]:
        """Retorna conjunto de modelos ja carregados.

        Returns:
            Set com nomes dos modelos carregados.
        """
        return self._loaded_models.copy()
