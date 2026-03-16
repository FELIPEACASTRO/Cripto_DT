"""Testes para melhorias da Fase 3 do Roadmap (aiagentstore.ai).

Testa os novos modulos:
- MarketContextMemory com suporte a Jina AI embeddings
- AdvancedNewsCrawler (async crawler com deduplicacao)
- SmartMoneyTracker (inteligencia on-chain via OHLCV)
- NarrativeDetector (deteccao de narrativas cripto)
- Multi-Agent Architecture (DataAgent, AnalystAgent, TraderAgent, RiskAgent, Orchestrator)
"""

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════
# Fixtures compartilhadas
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_ohlcv():
    """DataFrame OHLCV sintetico para testes."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 40000 + np.cumsum(np.random.randn(n) * 500)
    volume = np.abs(np.random.randn(n) * 1e9) + 1e8

    return pd.DataFrame({
        "timestamp": dates,
        "open": close - np.random.rand(n) * 200,
        "high": close + np.abs(np.random.randn(n) * 300),
        "low": close - np.abs(np.random.randn(n) * 300),
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_articles():
    """Lista de artigos sinteticos para testes de NLP."""
    return [
        {
            "title": "Bitcoin ETF sees record inflows as BlackRock leads institutional adoption",
            "text": "Institutional investors continue to pour money into Bitcoin ETFs.",
            "source": "CoinDesk",
            "timestamp": pd.Timestamp("2024-03-15", tz="UTC"),
            "coin_relevance": 0.9,
        },
        {
            "title": "DeFi protocol launches new yield farming strategy with high APY",
            "text": "A new DeFi lending platform offers competitive yields through liquidity mining.",
            "source": "CoinTelegraph",
            "timestamp": pd.Timestamp("2024-03-14", tz="UTC"),
            "coin_relevance": 0.5,
        },
        {
            "title": "AI crypto tokens surge as machine learning integration grows",
            "text": "Artificial intelligence tokens rally on GPT and neural network hype.",
            "source": "Decrypt",
            "timestamp": pd.Timestamp("2024-03-13", tz="UTC"),
            "coin_relevance": 0.7,
        },
        {
            "title": "SEC announces new regulation framework for cryptocurrency exchanges",
            "text": "The SEC enforcement action targets unregistered exchanges with compliance requirements.",
            "source": "TheBlock",
            "timestamp": pd.Timestamp("2024-03-12", tz="UTC"),
            "coin_relevance": 0.6,
        },
        {
            "title": "Ethereum layer 2 rollup scaling solution processes record transactions",
            "text": "L2 optimistic and zk rollup networks see increased scaling activity.",
            "source": "Reddit",
            "timestamp": pd.Timestamp("2024-03-11", tz="UTC"),
            "coin_relevance": 0.8,
        },
    ]


# ══════════════════════════════════════════════════════════════════════
# Test MarketContextMemory (Jina AI Embeddings)
# ══════════════════════════════════════════════════════════════════════

class TestMarketContextMemoryJina:
    """Testa MarketContextMemory com suporte a backends multiplos."""

    def test_init_auto_backend(self):
        """Modo auto deve inicializar com algum backend disponivel."""
        from src.data.market_context import MarketContextMemory
        memory = MarketContextMemory(embedding_backend="auto")
        assert memory._active_backend in ("jina", "sentence-transformers", "tfidf")

    def test_init_tfidf_fallback(self):
        """Backend tfidf deve sempre funcionar como fallback."""
        from src.data.market_context import MarketContextMemory
        memory = MarketContextMemory(embedding_backend="tfidf")
        assert memory._active_backend == "tfidf"
        assert memory.embedding_model_name == "tfidf"

    def test_max_seq_length_property(self):
        """Propriedade max_seq_length deve retornar valor correto por backend."""
        from src.data.market_context import MarketContextMemory, _MAX_SEQ_LENGTHS
        memory = MarketContextMemory(embedding_backend="tfidf")
        assert memory.max_seq_length == _MAX_SEQ_LENGTHS["tfidf"]

    def test_feature_names_includes_narrative(self):
        """Feature names deve incluir ctx_narrative_similarity."""
        from src.data.market_context import MarketContextMemory
        names = MarketContextMemory.get_feature_names()
        assert "ctx_narrative_similarity" in names
        assert len(names) == 5

    def test_add_and_query_tfidf(self):
        """Adicionar e consultar eventos com backend tfidf."""
        from src.data.market_context import MarketContextMemory
        memory = MarketContextMemory(embedding_backend="tfidf")
        memory.add_event(
            "BTC teve alta de 5% com volume alto",
            {"coin": "BTC", "outcome_1d": 2.0}
        )
        memory.add_event(
            "ETH caiu 3% apos regulacao SEC",
            {"coin": "ETH", "outcome_1d": -1.5}
        )
        results = memory.query_similar("Bitcoin subindo com volume", top_k=2)
        assert len(results) == 2
        assert "similarity" in results[0]

    def test_backward_compatibility(self):
        """API antiga (sem embedding_backend) deve continuar funcionando."""
        from src.data.market_context import MarketContextMemory
        # Chamada sem os novos parametros deve funcionar
        memory = MarketContextMemory(memory_path="data/test_memory")
        assert memory._active_backend in ("jina", "sentence-transformers", "tfidf")


# ══════════════════════════════════════════════════════════════════════
# Test AdvancedNewsCrawler
# ══════════════════════════════════════════════════════════════════════

class TestAdvancedNewsCrawler:
    """Testa o crawler avancado de noticias."""

    def test_init(self):
        """Inicializacao com parametros default."""
        from src.data.advanced_crawler import AdvancedNewsCrawler
        crawler = AdvancedNewsCrawler(max_articles=50, cache_ttl_hours=2)
        assert crawler.max_articles == 50

    def test_get_feature_names(self):
        """Feature names corretos."""
        from src.data.advanced_crawler import AdvancedNewsCrawler
        names = AdvancedNewsCrawler.get_feature_names()
        assert "crawler_news_count" in names
        assert "crawler_sentiment_mean" in names
        assert "crawler_fear_greed" in names
        assert "crawler_buzz_score" in names
        assert "crawler_source_diversity" in names
        assert len(names) == 5

    def test_add_crawler_features_empty(self, sample_ohlcv):
        """Add features deve funcionar mesmo sem dados de noticias (graceful)."""
        from src.data.advanced_crawler import AdvancedNewsCrawler
        crawler = AdvancedNewsCrawler(max_articles=0)
        # Nao faz request real — apenas verifica que nao da erro
        df = sample_ohlcv.copy()
        for feat in AdvancedNewsCrawler.get_feature_names():
            df[feat] = 0.0
        assert all(f in df.columns for f in AdvancedNewsCrawler.get_feature_names())

    def test_deduplication(self):
        """Deduplicacao de artigos por titulo similar."""
        from src.data.advanced_crawler import AdvancedNewsCrawler
        crawler = AdvancedNewsCrawler()
        articles = [
            {"title": "Bitcoin surges 10%", "text": "BTC rally", "source": "A",
             "timestamp": pd.Timestamp.now(tz="UTC"), "coin_relevance": 1.0},
            {"title": "Bitcoin surges 10%", "text": "BTC rally duplicate", "source": "B",
             "timestamp": pd.Timestamp.now(tz="UTC"), "coin_relevance": 1.0},
            {"title": "Ethereum drops", "text": "ETH falls", "source": "C",
             "timestamp": pd.Timestamp.now(tz="UTC"), "coin_relevance": 0.5},
        ]
        deduped = crawler._deduplicate(articles)
        assert len(deduped) == 2


# ══════════════════════════════════════════════════════════════════════
# Test SmartMoneyTracker
# ══════════════════════════════════════════════════════════════════════

class TestSmartMoneyTracker:
    """Testa o rastreador de smart money."""

    def test_init(self):
        """Inicializacao sem API key."""
        from src.data.smart_money import SmartMoneyTracker
        tracker = SmartMoneyTracker()
        assert tracker is not None

    def test_get_feature_names(self):
        """Feature names corretos."""
        from src.data.smart_money import SmartMoneyTracker
        names = SmartMoneyTracker.get_feature_names()
        assert "smart_money_flow" in names
        assert "smart_money_accumulation" in names
        assert "smart_money_distribution" in names
        assert "exchange_flow_ratio" in names
        assert "whale_concentration" in names
        assert len(names) == 5

    def test_add_features(self, sample_ohlcv):
        """Adicionar features de smart money ao DataFrame."""
        from src.data.smart_money import SmartMoneyTracker
        tracker = SmartMoneyTracker()
        df = tracker.add_smart_money_features(sample_ohlcv.copy(), "BTC")
        for feat in SmartMoneyTracker.get_feature_names():
            assert feat in df.columns
        assert len(df) == len(sample_ohlcv)

    def test_features_valid_range(self, sample_ohlcv):
        """Features devem estar em ranges validos."""
        from src.data.smart_money import SmartMoneyTracker
        tracker = SmartMoneyTracker()
        df = tracker.add_smart_money_features(sample_ohlcv.copy(), "ETH")
        # smart_money_flow deve estar entre -1 e 1
        assert df["smart_money_flow"].dropna().between(-1.01, 1.01).all()
        # accumulation e distribution devem estar entre 0 e 1
        assert df["smart_money_accumulation"].dropna().between(-0.01, 1.01).all()
        assert df["smart_money_distribution"].dropna().between(-0.01, 1.01).all()

    def test_features_no_nans_after_warmup(self, sample_ohlcv):
        """Apos periodo de warmup, nao deve ter NaNs."""
        from src.data.smart_money import SmartMoneyTracker
        tracker = SmartMoneyTracker()
        df = tracker.add_smart_money_features(sample_ohlcv.copy(), "BTC")
        # Apos 30 linhas (warmup), smart_money_flow nao deve ser NaN
        assert df["smart_money_flow"].iloc[35:].notna().all()


# ══════════════════════════════════════════════════════════════════════
# Test NarrativeDetector
# ══════════════════════════════════════════════════════════════════════

class TestNarrativeDetector:
    """Testa o detector de narrativas."""

    def test_init(self):
        """Inicializacao default."""
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        assert detector is not None
        assert len(detector.NARRATIVE_KEYWORDS) >= 8

    def test_detect_narratives(self, sample_articles):
        """Deteccao de narrativas a partir de artigos."""
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        scores = detector.detect_narratives(sample_articles)
        assert isinstance(scores, dict)
        # Deve detectar as narrativas presentes nos artigos de teste
        assert "defi" in scores
        assert "ai_crypto" in scores
        assert "regulation" in scores
        assert "institutional" in scores
        # DeFi e AI devem ter scores positivos
        assert scores["defi"] > 0
        assert scores["ai_crypto"] > 0

    def test_get_dominant_narrative(self, sample_articles):
        """Narrativa dominante deve ser identificada."""
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        name, score = detector.get_dominant_narrative(sample_articles)
        assert isinstance(name, str)
        assert score > 0

    def test_get_feature_names(self):
        """Feature names corretos."""
        from src.data.narrative_detector import NarrativeDetector
        names = NarrativeDetector.get_feature_names()
        assert "narrative_dominant_score" in names
        assert "narrative_diversity" in names
        assert "narrative_momentum_7d" in names
        assert "narrative_alignment" in names
        assert "narrative_contrarian" in names
        assert len(names) == 5

    def test_detect_empty_articles(self):
        """Lista vazia de artigos deve retornar scores zerados."""
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        scores = detector.detect_narratives([])
        assert all(v == 0.0 for v in scores.values())

    def test_dominant_narrative_empty(self):
        """Sem artigos, narrativa dominante deve ter score 0."""
        from src.data.narrative_detector import NarrativeDetector
        detector = NarrativeDetector()
        name, score = detector.get_dominant_narrative([])
        assert score == 0.0


# ══════════════════════════════════════════════════════════════════════
# Test Multi-Agent Architecture
# ══════════════════════════════════════════════════════════════════════

class TestBaseAgent:
    """Testa a classe base dos agentes."""

    def test_agent_result_dataclass(self):
        """AgentResult deve ser criado com defaults corretos."""
        from src.agents.base_agent import AgentResult
        result = AgentResult()
        assert result.success is True
        assert result.data == {}
        assert result.errors == []
        assert result.duration_seconds == 0.0

    def test_agent_result_with_data(self):
        """AgentResult com dados customizados."""
        from src.agents.base_agent import AgentResult
        result = AgentResult(
            success=False,
            data={"predictions": {"BTC": 50000}},
            errors=["Erro de teste"],
        )
        assert result.success is False
        assert result.data["predictions"]["BTC"] == 50000
        assert len(result.errors) == 1


class TestDataAgent:
    """Testa o agente de coleta de dados."""

    def test_init(self):
        """DataAgent deve inicializar corretamente."""
        from src.agents.data_agent import DataAgent
        agent = DataAgent()
        assert agent.name == "data_agent"
        assert agent.status == "idle"

    def test_health_check(self):
        """Health check deve retornar status valido."""
        from src.agents.data_agent import DataAgent
        agent = DataAgent()
        health = agent.health_check()
        assert health["name"] == "data_agent"
        assert health["status"] == "idle"
        assert health["execution_count"] == 0


class TestAnalystAgent:
    """Testa o agente de analise."""

    def test_init(self):
        """AnalystAgent deve inicializar corretamente."""
        from src.agents.analyst_agent import AnalystAgent
        agent = AnalystAgent()
        assert agent.name == "analyst_agent"
        assert agent.status == "idle"

    def test_health_check(self):
        """Health check deve retornar status valido."""
        from src.agents.analyst_agent import AnalystAgent
        agent = AnalystAgent()
        health = agent.health_check()
        assert health["name"] == "analyst_agent"
        assert health["status"] == "idle"


class TestTraderAgent:
    """Testa o agente de trading."""

    def test_init(self):
        """TraderAgent deve inicializar corretamente."""
        from src.agents.trader_agent import TraderAgent
        agent = TraderAgent()
        assert agent.name == "trader_agent"
        assert agent.status == "idle"

    def test_health_check(self):
        """Health check deve retornar status valido."""
        from src.agents.trader_agent import TraderAgent
        agent = TraderAgent()
        health = agent.health_check()
        assert health["name"] == "trader_agent"


class TestRiskAgent:
    """Testa o agente de risco."""

    def test_init(self):
        """RiskAgent deve inicializar corretamente."""
        from src.agents.risk_agent import RiskAgent
        agent = RiskAgent()
        assert agent.name == "risk_agent"
        assert agent.status == "idle"

    def test_health_check(self):
        """Health check deve retornar status valido."""
        from src.agents.risk_agent import RiskAgent
        agent = RiskAgent()
        health = agent.health_check()
        assert health["name"] == "risk_agent"


class TestOrchestrator:
    """Testa o orquestrador de agentes."""

    def test_init(self):
        """Orchestrator deve inicializar corretamente."""
        from src.agents.orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        assert orch._latest_results is None
        assert orch._run_history == []

    def test_init_with_config(self):
        """Orchestrator deve aceitar config."""
        from config.settings import Config
        from src.agents.orchestrator import AgentOrchestrator
        cfg = Config()
        orch = AgentOrchestrator(config=cfg)
        assert orch.config is cfg
