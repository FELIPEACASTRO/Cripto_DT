"""Testes para os modulos da Fase 4 — descobertas do aiagentstore.ai.

Cobre: SmartMoneyTracker, NarrativeDetector, AdvancedNewsCrawler,
BaseAgent, RiskAgent, AgentOrchestrator.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    """Cria DataFrame OHLCV sintetico para testes."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 40_000 + np.cumsum(rng.normal(0, 200, n))
    high = close + rng.uniform(100, 500, n)
    low = close - rng.uniform(100, 500, n)
    opn = close + rng.normal(0, 100, n)
    volume = rng.uniform(1e9, 5e9, n)
    return pd.DataFrame({
        "timestamp": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _make_articles(n: int = 10) -> list[dict]:
    """Cria lista de artigos sinteticos para testes de NLP."""
    base = datetime.now(timezone.utc)
    titles = [
        "DeFi yield farming surges as liquidity pools grow",
        "AI tokens rally on new GPT model release",
        "Real world assets tokenization gains momentum",
        "Memecoins crash after social media frenzy",
        "L2 scaling solutions see record TVL",
        "SEC announces new crypto regulation framework",
        "BlackRock institutional Bitcoin ETF hits $10B",
        "Gaming NFT marketplace launches on Solana",
        "Bitcoin whale accumulation signals bullish trend",
        "Ethereum DeFi TVL crosses $100 billion",
    ]
    articles = []
    for i in range(min(n, len(titles))):
        articles.append({
            "title": titles[i],
            "text": f"Article body about {titles[i].lower()}. More details here.",
            "source": "test",
            "timestamp": (base - timedelta(days=i)).isoformat(),
        })
    return articles


# ──────────────────────────────────────────────────────────────────────
# SmartMoneyTracker
# ──────────────────────────────────────────────────────────────────────

class TestSmartMoneyTracker:
    def test_init(self):
        from src.data.smart_money import SmartMoneyTracker
        tracker = SmartMoneyTracker()
        assert tracker is not None

    def test_get_feature_names(self):
        from src.data.smart_money import SmartMoneyTracker
        names = SmartMoneyTracker.get_feature_names()
        assert isinstance(names, list)
        assert len(names) >= 4
        assert all(isinstance(n, str) for n in names)

    def test_add_features(self):
        from src.data.smart_money import SmartMoneyTracker
        tracker = SmartMoneyTracker()
        df = _make_ohlcv(120)
        result = tracker.add_smart_money_features(df, "BTC")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        for feat in SmartMoneyTracker.get_feature_names():
            assert feat in result.columns, f"Feature {feat} ausente"

    def test_features_are_numeric(self):
        from src.data.smart_money import SmartMoneyTracker
        tracker = SmartMoneyTracker()
        df = _make_ohlcv(120)
        result = tracker.add_smart_money_features(df, "ETH")
        for feat in SmartMoneyTracker.get_feature_names():
            assert pd.api.types.is_numeric_dtype(result[feat]), f"{feat} nao e numerico"

    def test_no_crash_on_small_data(self):
        from src.data.smart_money import SmartMoneyTracker
        tracker = SmartMoneyTracker()
        df = _make_ohlcv(5)
        result = tracker.add_smart_money_features(df, "SOL")
        assert len(result) == 5


# ──────────────────────────────────────────────────────────────────────
# NarrativeDetector
# ──────────────────────────────────────────────────────────────────────

class TestNarrativeDetector:
    def test_init(self):
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        assert detector is not None
        assert hasattr(detector, "NARRATIVE_KEYWORDS")

    def test_detect_narratives(self):
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        articles = _make_articles()
        scores = detector.detect_narratives(articles)
        assert isinstance(scores, dict)
        assert len(scores) > 0
        # Todas as pontuacoes devem ser >= 0
        for k, v in scores.items():
            assert v >= 0, f"Narrativa {k} com score negativo: {v}"

    def test_detect_narratives_empty(self):
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        scores = detector.detect_narratives([])
        assert isinstance(scores, dict)

    def test_get_dominant_narrative(self):
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        articles = _make_articles()
        name, score = detector.get_dominant_narrative(articles)
        assert isinstance(name, str)
        assert isinstance(score, (int, float))

    def test_get_feature_names(self):
        from src.data.narrative_detector import NarrativeDetector
        names = NarrativeDetector.get_feature_names()
        assert isinstance(names, list)
        assert len(names) == 5
        assert "narrative_dominant_score" in names

    def test_add_narrative_features(self):
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        df = _make_ohlcv(30)
        # NarrativeDetector precisa de coluna _date para merge (tz-naive)
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        result = detector.add_narrative_features(df, "BTC")
        assert isinstance(result, pd.DataFrame)
        for feat in NarrativeDetector.get_feature_names():
            assert feat in result.columns


# ──────────────────────────────────────────────────────────────────────
# AdvancedNewsCrawler
# ──────────────────────────────────────────────────────────────────────

class TestAdvancedNewsCrawler:
    def test_init(self):
        from src.data.advanced_crawler import AdvancedNewsCrawler
        crawler = AdvancedNewsCrawler()
        assert crawler is not None

    def test_get_feature_names(self):
        from src.data.advanced_crawler import AdvancedNewsCrawler
        names = AdvancedNewsCrawler.get_feature_names()
        assert isinstance(names, list)
        assert len(names) >= 5
        assert "crawler_news_count" in names
        assert "crawler_fear_greed" in names

    def test_add_crawler_features_no_network(self):
        """Testa que add_crawler_features nao crasha mesmo sem rede."""
        from src.data.advanced_crawler import AdvancedNewsCrawler
        crawler = AdvancedNewsCrawler()
        df = _make_ohlcv(30)
        df["_date"] = pd.to_datetime(df["timestamp"]).dt.date
        result = crawler.add_crawler_features(df, "BTC")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_dedup_logic(self):
        """Testa que deduplicacao funciona."""
        from src.data.advanced_crawler import AdvancedNewsCrawler
        crawler = AdvancedNewsCrawler()
        articles = [
            {"title": "Test Article One", "text": "Body 1", "source": "a", "timestamp": "2024-01-01"},
            {"title": "Test Article One", "text": "Body 2", "source": "b", "timestamp": "2024-01-01"},
            {"title": "Different Article", "text": "Body 3", "source": "c", "timestamp": "2024-01-02"},
        ]
        deduped = crawler._deduplicate(articles)
        assert len(deduped) == 2


# ──────────────────────────────────────────────────────────────────────
# BaseAgent & AgentResult
# ──────────────────────────────────────────────────────────────────────

class TestBaseAgent:
    def test_agent_result_defaults(self):
        from src.agents.base_agent import AgentResult
        result = AgentResult()
        assert result.success is True
        assert result.data == {}
        assert result.errors == []
        assert result.duration_seconds == 0.0

    def test_concrete_agent_run(self):
        from src.agents.base_agent import BaseAgent, AgentResult

        class DummyAgent(BaseAgent):
            name = "dummy"
            def execute(self, context: dict) -> AgentResult:
                return AgentResult(
                    success=True,
                    data={"resultado": context.get("input", 0) * 2}
                )

        agent = DummyAgent()
        result = agent.run({"input": 21})
        assert result.success is True
        assert result.data["resultado"] == 42
        assert result.duration_seconds > 0
        assert agent.status == "done"

    def test_agent_error_handling(self):
        from src.agents.base_agent import BaseAgent, AgentResult

        class FailAgent(BaseAgent):
            name = "fail"
            def execute(self, context: dict) -> AgentResult:
                raise ValueError("Erro intencional")

        agent = FailAgent()
        result = agent.run({})
        assert result.success is False
        assert len(result.errors) > 0
        assert agent.status == "error"

    def test_health_check(self):
        from src.agents.base_agent import BaseAgent, AgentResult

        class TestAgent(BaseAgent):
            name = "test"
            def execute(self, context: dict) -> AgentResult:
                return AgentResult(success=True)

        agent = TestAgent()
        health = agent.health_check()
        assert health["name"] == "test"
        assert health["status"] == "idle"
        assert health["execution_count"] == 0

        agent.run({})
        health = agent.health_check()
        assert health["status"] == "done"
        assert health["execution_count"] == 1

    def test_safe_import(self):
        from src.agents.base_agent import _safe_import
        # Modulo existente
        cls = _safe_import("json", "JSONEncoder")
        assert cls is not None
        # Modulo inexistente
        cls = _safe_import("modulo_que_nao_existe", "Foo")
        assert cls is None


# ──────────────────────────────────────────────────────────────────────
# RiskAgent
# ──────────────────────────────────────────────────────────────────────

class TestRiskAgent:
    def test_init(self):
        from src.agents.risk_agent import RiskAgent, RiskLimits
        agent = RiskAgent()
        assert "risk" in agent.name
        assert agent.status == "idle"

    def test_risk_limits_defaults(self):
        from src.agents.risk_agent import RiskLimits
        limits = RiskLimits()
        assert limits.max_portfolio_exposure == 0.8
        assert limits.min_confidence == 0.3

    def test_filters_low_confidence(self):
        from src.agents.risk_agent import RiskAgent, RiskLimits
        agent = RiskAgent(risk_limits=RiskLimits(min_confidence=0.5))
        context = {
            "signals": {
                "BTC": {
                    "action": "BUY",
                    "confidence": 0.3,
                    "position_size": 0.1,
                }
            }
        }
        result = agent.run(context)
        assert result.success is True
        # Sinal com confianca baixa deve ser filtrado ou reduzido


# ──────────────────────────────────────────────────────────────────────
# AgentOrchestrator
# ──────────────────────────────────────────────────────────────────────

class TestAgentOrchestrator:
    def test_init(self):
        from src.agents.orchestrator import AgentOrchestrator
        orch = AgentOrchestrator(save_results=False)
        assert orch is not None
        assert orch.save_results is False

    def test_health_check(self):
        from src.agents.orchestrator import AgentOrchestrator
        orch = AgentOrchestrator(save_results=False)
        health = orch.health_check()
        assert isinstance(health, dict)
        assert len(health) > 0
        # Deve conter pelo menos um agente
        assert any("agent" in k for k in health.keys())


# ──────────────────────────────────────────────────────────────────────
# Jina Embeddings Integration (market_context.py atualizado)
# ──────────────────────────────────────────────────────────────────────

class TestJinaEmbeddings:
    def test_backend_auto_detection(self):
        from src.data.market_context import MarketContextMemory
        mem = MarketContextMemory()
        # Deve ter um backend ativo (jina, sentence-transformers ou tfidf)
        assert mem._active_backend in ("jina", "sentence-transformers", "tfidf")

    def test_tfidf_fallback(self):
        from src.data.market_context import MarketContextMemory
        mem = MarketContextMemory(embedding_backend="tfidf")
        assert mem._active_backend == "tfidf"
        mem.add_event("Bitcoin rally driven by ETF approval", {"coin": "BTC"})
        results = mem.query_similar("ETF Bitcoin")
        assert isinstance(results, list)

    def test_narrative_similarity_feature(self):
        from src.data.market_context import MarketContextMemory
        names = MarketContextMemory.get_feature_names()
        assert "ctx_narrative_similarity" in names
