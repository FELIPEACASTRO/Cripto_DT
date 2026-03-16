"""Memoria de contexto de mercado com busca semantica de eventos passados."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    HAS_TFIDF = True
except ImportError:
    HAS_TFIDF = False

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "ctx_similar_outcome_mean",
    "ctx_similar_outcome_std",
    "ctx_pattern_confidence",
    "ctx_regime_duration_similar",
]


class MarketContextMemory:
    """Memoria de longo prazo para eventos e padroes de mercado.

    Armazena eventos de mercado com embeddings semanticos para busca
    por similaridade. Permite ao sistema de trading consultar situacoes
    historicas semelhantes e seus resultados.

    Utiliza sentence-transformers para embeddings; caso nao disponivel,
    recorre a TF-IDF do sklearn como fallback.
    """

    def __init__(
        self,
        memory_path: Path = Path("data/market_memory"),
        embedding_model: str = "all-MiniLM-L6-v2",
        max_events: int = 10000,
    ):
        self.memory_path = Path(memory_path)
        self.embedding_model_name = embedding_model
        self.max_events = max_events
        self.events: list[dict] = []
        self._encoder = None
        self._tfidf: TfidfVectorizer | None = None
        self._use_sentence_transformers = False

        self._init_encoder()

    def _init_encoder(self) -> None:
        """Inicializa o encoder de embeddings."""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self._encoder = SentenceTransformer(self.embedding_model_name)
                self._use_sentence_transformers = True
                logger.info(
                    "Encoder sentence-transformers carregado: %s",
                    self.embedding_model_name,
                )
                return
            except Exception as e:
                logger.warning(
                    "Falha ao carregar sentence-transformers: %s. "
                    "Usando fallback TF-IDF.",
                    e,
                )

        if HAS_TFIDF:
            self._tfidf = TfidfVectorizer(max_features=512)
            self._use_sentence_transformers = False
            logger.info("Usando TF-IDF como fallback para embeddings.")
        else:
            raise ImportError(
                "sentence-transformers ou sklearn e necessario para "
                "MarketContextMemory. Instale com: "
                "pip install sentence-transformers ou pip install scikit-learn"
            )

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Gera embeddings para uma lista de textos.

        Args:
            texts: Lista de descricoes textuais.

        Returns:
            Array (n_texts, embedding_dim) de embeddings.
        """
        if self._use_sentence_transformers and self._encoder is not None:
            return self._encoder.encode(texts, show_progress_bar=False)

        # Fallback TF-IDF
        if self._tfidf is None:
            raise RuntimeError("Nenhum encoder disponivel.")

        # Re-fit TF-IDF com todos os textos existentes + novos
        all_texts = [e["description"] for e in self.events] + texts
        if len(all_texts) == 0:
            return np.array([])

        tfidf_matrix = self._tfidf.fit_transform(all_texts)
        # Retornar apenas os embeddings dos textos solicitados
        return tfidf_matrix[-len(texts):].toarray()

    def add_event(self, description: str, metadata: dict) -> None:
        """Armazena um evento de mercado com seu embedding.

        Args:
            description: Descricao textual do evento.
            metadata: Dicionario com campos como coin, date,
                price_change_pct, direction, regime, volatility,
                outcome_1d, outcome_7d.
        """
        try:
            embedding = self._encode([description])[0]
        except Exception as e:
            logger.error("Falha ao gerar embedding para evento: %s", e)
            return

        event = {
            "description": description,
            "metadata": metadata,
            "embedding": embedding,
            "added_at": datetime.utcnow().isoformat(),
        }
        self.events.append(event)

        # Limitar tamanho da memoria
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
            logger.info(
                "Memoria truncada para %d eventos (max_events=%d).",
                len(self.events),
                self.max_events,
            )

        logger.debug(
            "Evento adicionado: %s (total: %d)", description[:80], len(self.events)
        )

    def query_similar(self, query: str, top_k: int = 5) -> list[dict]:
        """Busca eventos similares por similaridade de cosseno.

        Args:
            query: Descricao textual para busca.
            top_k: Numero maximo de resultados.

        Returns:
            Lista de dicts com description, metadata e similarity.
        """
        if not self.events:
            return []

        try:
            query_emb = self._encode([query])
        except Exception as e:
            logger.error("Falha ao gerar embedding para query: %s", e)
            return []

        # Recalcular embeddings TF-IDF se necessario
        if not self._use_sentence_transformers:
            all_texts = [e["description"] for e in self.events] + [query]
            try:
                tfidf_matrix = self._tfidf.fit_transform(all_texts)
                event_embs = tfidf_matrix[:-1].toarray()
                query_emb = tfidf_matrix[-1:].toarray()
            except Exception as e:
                logger.error("Falha ao calcular TF-IDF para query: %s", e)
                return []
        else:
            event_embs = np.array([e["embedding"] for e in self.events])

        similarities = cosine_similarity(query_emb, event_embs)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "description": self.events[idx]["description"],
                "metadata": self.events[idx]["metadata"],
                "similarity": float(similarities[idx]),
            })

        return results

    def add_market_snapshot(self, df: pd.DataFrame, coin: str) -> None:
        """Gera e armazena eventos automaticamente a partir de dados de mercado.

        Detecta movimentos significativos, mudancas de regime, picos de volume
        e breakouts de volatilidade.

        Args:
            df: DataFrame com colunas close, volume, e opcionalmente
                regime_state, log_return.
            coin: Simbolo da criptomoeda (ex: BTC).
        """
        if df.empty or len(df) < 20:
            logger.warning("DataFrame insuficiente para gerar snapshot de mercado.")
            return

        df = df.copy()

        # Garantir coluna de retorno
        if "log_return" not in df.columns and "close" in df.columns:
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        if "pct_change" not in df.columns and "close" in df.columns:
            df["pct_change"] = df["close"].pct_change() * 100

        events_added = 0

        for i in range(20, len(df)):
            row = df.iloc[i]
            date_str = str(row.get("timestamp", row.name))

            pct = row.get("pct_change", np.nan)
            close = row.get("close", np.nan)
            volume = row.get("volume", np.nan)
            regime = row.get("regime_state", None)

            # Calcular outcomes futuros
            outcome_1d = np.nan
            outcome_7d = np.nan
            if i + 1 < len(df) and "close" in df.columns:
                outcome_1d = (df.iloc[i + 1]["close"] / close - 1) * 100
            if i + 7 < len(df) and "close" in df.columns:
                outcome_7d = (df.iloc[i + 7]["close"] / close - 1) * 100

            # Volatilidade recente
            recent_returns = df["log_return"].iloc[max(0, i - 20):i].dropna()
            volatility = float(recent_returns.std()) if len(recent_returns) > 2 else 0.0

            base_metadata = {
                "coin": coin,
                "date": date_str,
                "price_change_pct": float(pct) if not np.isnan(pct) else 0.0,
                "direction": "up" if pct > 0 else "down",
                "regime": str(regime) if regime is not None else "unknown",
                "volatility": volatility,
                "outcome_1d": float(outcome_1d) if not np.isnan(outcome_1d) else None,
                "outcome_7d": float(outcome_7d) if not np.isnan(outcome_7d) else None,
            }

            # 1. Movimentos significativos (>3% diario)
            if not np.isnan(pct) and abs(pct) > 3.0:
                direction = "alta" if pct > 0 else "queda"
                desc = (
                    f"{coin} teve {direction} significativa de {pct:.1f}% "
                    f"em {date_str}. Volatilidade recente: {volatility:.4f}. "
                    f"Regime: {regime}."
                )
                self.add_event(desc, {**base_metadata, "event_type": "significant_move"})
                events_added += 1

            # 2. Mudanca de regime
            if regime is not None and i > 0:
                prev_regime = df.iloc[i - 1].get("regime_state", None)
                if prev_regime is not None and regime != prev_regime:
                    desc = (
                        f"{coin} mudou de regime {prev_regime} para {regime} "
                        f"em {date_str}. Preco: {close:.2f}. "
                        f"Volatilidade: {volatility:.4f}."
                    )
                    self.add_event(
                        desc, {**base_metadata, "event_type": "regime_change"}
                    )
                    events_added += 1

            # 3. Pico de volume (>2x media de 20 periodos)
            if not np.isnan(volume):
                avg_volume = df["volume"].iloc[max(0, i - 20):i].mean()
                if avg_volume > 0 and volume > 2 * avg_volume:
                    ratio = volume / avg_volume
                    desc = (
                        f"{coin} teve pico de volume ({ratio:.1f}x a media) "
                        f"em {date_str}. Variacao de preco: {pct:.1f}%. "
                        f"Regime: {regime}."
                    )
                    self.add_event(
                        desc, {**base_metadata, "event_type": "volume_spike"}
                    )
                    events_added += 1

            # 4. Breakout de volatilidade
            if len(recent_returns) > 5:
                vol_ma = df["log_return"].iloc[max(0, i - 60):i].dropna().std()
                if vol_ma > 0 and volatility > 2 * vol_ma:
                    desc = (
                        f"{coin} breakout de volatilidade em {date_str}. "
                        f"Volatilidade atual: {volatility:.4f}, "
                        f"media: {vol_ma:.4f}. Preco: {close:.2f}."
                    )
                    self.add_event(
                        desc,
                        {**base_metadata, "event_type": "volatility_breakout"},
                    )
                    events_added += 1

        logger.info(
            "Snapshot de mercado para %s: %d eventos adicionados (total: %d).",
            coin,
            events_added,
            len(self.events),
        )

    def get_context_features(self, df: pd.DataFrame, coin: str) -> pd.DataFrame:
        """Adiciona features de contexto historico ao DataFrame.

        Para cada linha, busca eventos similares na memoria e calcula
        estatisticas dos outcomes passados.

        Features adicionadas:
          - ctx_similar_outcome_mean: media dos outcomes de eventos similares
          - ctx_similar_outcome_std: desvio padrao dos outcomes
          - ctx_pattern_confidence: confianca baseada na quantidade e
            similaridade dos matches
          - ctx_regime_duration_similar: duracao media de regimes similares

        Args:
            df: DataFrame com colunas close e opcionalmente regime_state.
            coin: Simbolo da criptomoeda.

        Returns:
            DataFrame com features de contexto adicionadas.
        """
        df = df.copy()

        # Inicializar colunas com NaN
        for col in FEATURE_NAMES:
            df[col] = np.nan

        if not self.events or df.empty:
            logger.warning(
                "Memoria vazia ou DataFrame vazio. "
                "Retornando features de contexto zeradas."
            )
            return df

        # Garantir coluna de retorno
        if "log_return" not in df.columns and "close" in df.columns:
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        if "pct_change" not in df.columns and "close" in df.columns:
            df["pct_change"] = df["close"].pct_change() * 100

        for i in range(20, len(df)):
            row = df.iloc[i]
            pct = row.get("pct_change", 0.0)
            regime = row.get("regime_state", "unknown")
            recent_returns = df["log_return"].iloc[max(0, i - 20):i].dropna()
            volatility = float(recent_returns.std()) if len(recent_returns) > 2 else 0.0

            query = (
                f"{coin} variacao de {pct:.1f}% com volatilidade {volatility:.4f} "
                f"em regime {regime}."
            )

            try:
                similar = self.query_similar(query, top_k=10)
            except Exception as e:
                logger.debug("Falha ao buscar similares para indice %d: %s", i, e)
                continue

            if not similar:
                continue

            # Filtrar resultados com outcome valido
            outcomes_1d = []
            similarities = []
            regime_durations = []

            for s in similar:
                meta = s["metadata"]
                outcome = meta.get("outcome_1d")
                if outcome is not None:
                    outcomes_1d.append(outcome)
                    similarities.append(s["similarity"])

                # Duracao do regime (se disponivel nos metadados)
                rdur = meta.get("regime_duration")
                if rdur is not None:
                    regime_durations.append(rdur)

            if outcomes_1d:
                outcomes_arr = np.array(outcomes_1d)
                sim_arr = np.array(similarities)

                df.iloc[i, df.columns.get_loc("ctx_similar_outcome_mean")] = float(
                    np.mean(outcomes_arr)
                )
                df.iloc[i, df.columns.get_loc("ctx_similar_outcome_std")] = float(
                    np.std(outcomes_arr)
                )

                # Confianca: combinacao do numero de matches e similaridade media
                n_matches = len(outcomes_arr)
                avg_sim = float(np.mean(sim_arr))
                confidence = min(1.0, (n_matches / 10.0)) * avg_sim
                df.iloc[i, df.columns.get_loc("ctx_pattern_confidence")] = confidence

            if regime_durations:
                df.iloc[
                    i, df.columns.get_loc("ctx_regime_duration_similar")
                ] = float(np.mean(regime_durations))

        logger.info("Features de contexto adicionadas para %s.", coin)
        return df

    def save(self) -> None:
        """Persiste a memoria em disco.

        Salva eventos em formato pickle (com embeddings) e metadados em JSON.
        """
        self.memory_path.mkdir(parents=True, exist_ok=True)

        events_path = self.memory_path / "events.pkl"
        meta_path = self.memory_path / "metadata.json"

        try:
            with open(events_path, "wb") as f:
                pickle.dump(self.events, f)

            meta = {
                "embedding_model": self.embedding_model_name,
                "max_events": self.max_events,
                "n_events": len(self.events),
                "use_sentence_transformers": self._use_sentence_transformers,
                "saved_at": datetime.utcnow().isoformat(),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            logger.info(
                "Memoria salva em %s (%d eventos).", self.memory_path, len(self.events)
            )
        except Exception as e:
            logger.error("Falha ao salvar memoria: %s", e)
            raise

    def load(self) -> None:
        """Carrega a memoria previamente salva do disco."""
        events_path = self.memory_path / "events.pkl"
        meta_path = self.memory_path / "metadata.json"

        if not events_path.exists():
            logger.warning(
                "Arquivo de memoria nao encontrado em %s. "
                "Iniciando com memoria vazia.",
                events_path,
            )
            return

        try:
            with open(events_path, "rb") as f:
                self.events = pickle.load(f)

            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                logger.info(
                    "Memoria carregada de %s: %d eventos (salva em %s).",
                    self.memory_path,
                    len(self.events),
                    meta.get("saved_at", "desconhecido"),
                )
            else:
                logger.info(
                    "Memoria carregada de %s: %d eventos.",
                    self.memory_path,
                    len(self.events),
                )
        except Exception as e:
            logger.error("Falha ao carregar memoria: %s", e)
            raise

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features de contexto geradas."""
        return list(FEATURE_NAMES)
