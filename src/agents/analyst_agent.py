"""AnalystAgent: executa analise tecnica, feature engineering e deteccao de contexto."""

import logging

from src.agents.base_agent import BaseAgent, AgentResult, _safe_import

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Agente responsavel por analise tecnica e feature engineering.

    Coordena multiplos modulos de features:
    - FeaturePipeline para indicadores tecnicos e features de lag
    - MarketContextMemory para contexto de mercado e eventos similares
    - NarrativeDetector para deteccao de narrativas dominantes
    - SmartMoneyTracker para rastreamento de dinheiro inteligente
    - NewsScraper para features de sentimento de noticias

    Recebe dados brutos do DataAgent e retorna DataFrames enriquecidos
    com todas as features necessarias para o TraderAgent.
    """

    name = "analyst_agent"

    def __init__(self, config=None):
        super().__init__()
        self.config = config

        # Componentes inicializados sob demanda
        self._feature_pipeline = None
        self._market_context = None
        self._narrative_detector = None
        self._smart_money_tracker = None
        self._news_scraper = None
        self._preprocessor = None
        self._initialized = False

    def _init_components(self):
        """Inicializa componentes de analise com importacoes seguras."""
        if self._initialized:
            return

        # FeaturePipeline - features tecnicas (componente principal)
        Cls = _safe_import("src.features.pipeline", "FeaturePipeline")
        if Cls:
            try:
                self._feature_pipeline = Cls(self.config) if self.config else Cls()
                logger.info(f"[{self.name}] FeaturePipeline: OK")
            except Exception as e:
                logger.warning(f"[{self.name}] FeaturePipeline: ERRO - {e}")

        # DataPreprocessor - limpeza de dados
        Cls = _safe_import("src.data.preprocessor", "DataPreprocessor")
        if Cls:
            try:
                self._preprocessor = Cls()
                logger.info(f"[{self.name}] DataPreprocessor: OK")
            except Exception as e:
                logger.warning(f"[{self.name}] DataPreprocessor: ERRO - {e}")

        # MarketContextMemory - contexto e busca semantica de eventos
        Cls = _safe_import("src.data.market_context", "MarketContextMemory")
        if Cls:
            try:
                self._market_context = Cls()
                logger.info(f"[{self.name}] MarketContextMemory: OK")
            except Exception as e:
                logger.warning(f"[{self.name}] MarketContextMemory: ERRO - {e}")

        # NarrativeDetector - deteccao de narrativas dominantes (opcional)
        Cls = _safe_import("src.data.news_scraper", "NarrativeDetector")
        if Cls:
            try:
                self._narrative_detector = Cls()
                logger.info(f"[{self.name}] NarrativeDetector: OK")
            except Exception as e:
                logger.debug(f"[{self.name}] NarrativeDetector: {e}")

        # SmartMoneyTracker - rastreamento de smart money (opcional)
        Cls = _safe_import("src.data.whale_monitor", "SmartMoneyTracker")
        if Cls:
            try:
                self._smart_money_tracker = Cls()
                logger.info(f"[{self.name}] SmartMoneyTracker: OK")
            except Exception as e:
                logger.debug(f"[{self.name}] SmartMoneyTracker: {e}")

        # NewsScraper - para features de sentimento
        Cls = _safe_import("src.data.news_scraper", "NewsScraper")
        if Cls:
            try:
                self._news_scraper = Cls()
                logger.info(f"[{self.name}] NewsScraper: OK")
            except Exception as e:
                logger.warning(f"[{self.name}] NewsScraper: ERRO - {e}")

        self._initialized = True

    def execute(self, context: dict) -> AgentResult:
        """Executa analise tecnica e feature engineering.

        Args:
            context: Deve conter (normalmente fornecido pelo DataAgent):
                - coins (list[str]): Lista de moedas
                - ohlcv (dict[str, DataFrame]): Dados OHLCV por moeda
                - news (dict[str, list], opcional): Noticias por moeda

        Returns:
            AgentResult com data contendo:
                - featured_data (dict[str, DataFrame]): DataFrames enriquecidos por moeda
                - market_context (dict): Contexto de mercado agregado
        """
        self._init_components()

        coins = context.get("coins", [])
        ohlcv = context.get("ohlcv", {})

        if not ohlcv:
            return AgentResult(
                success=False,
                errors=["Nenhum dado OHLCV disponivel para analise"],
            )

        result = AgentResult()
        result.data = {
            "featured_data": {},
            "market_context": {},
        }

        # --- Passo 1: Preprocessamento ---
        cleaned_data = {}
        if self._preprocessor is not None:
            for coin, df in ohlcv.items():
                try:
                    cleaned = self._preprocessor.clean(df)
                    if not cleaned.empty:
                        cleaned_data[coin] = cleaned
                except Exception as e:
                    msg = f"Erro preprocessamento {coin}: {e}"
                    logger.warning(f"[{self.name}] {msg}")
                    result.errors.append(msg)
                    cleaned_data[coin] = df  # Usa dado bruto como fallback
            logger.info(
                f"[{self.name}] Preprocessamento: {len(cleaned_data)} moedas"
            )
        else:
            cleaned_data = dict(ohlcv)
            logger.debug(f"[{self.name}] Preprocessor nao disponivel, usando dados brutos")

        # --- Passo 2: Feature Pipeline (indicadores tecnicos + lag + wavelet) ---
        featured_data = {}
        if self._feature_pipeline is not None:
            btc_df = cleaned_data.get("BTC")
            for coin, df in cleaned_data.items():
                try:
                    is_btc = coin == "BTC"
                    ref_btc = None if is_btc else btc_df
                    featured = self._feature_pipeline.transform(
                        df, btc_df=ref_btc, is_btc=is_btc, coin=coin
                    )
                    if not featured.empty:
                        featured_data[coin] = featured
                except Exception as e:
                    msg = f"Erro features {coin}: {e}"
                    logger.warning(f"[{self.name}] {msg}")
                    result.errors.append(msg)
                    featured_data[coin] = df  # Fallback

            logger.info(
                f"[{self.name}] Feature pipeline: {len(featured_data)} moedas processadas"
            )
        else:
            featured_data = dict(cleaned_data)
            result.errors.append("FeaturePipeline nao disponivel")
            logger.warning(f"[{self.name}] FeaturePipeline nao disponivel")

        # --- Passo 3: Features de noticias/sentimento ---
        if self._news_scraper is not None:
            for coin in coins:
                if coin in featured_data:
                    try:
                        featured_data[coin] = self._news_scraper.add_news_features(
                            featured_data[coin], coin
                        )
                    except Exception as e:
                        msg = f"Erro news features {coin}: {e}"
                        logger.warning(f"[{self.name}] {msg}")
                        result.errors.append(msg)
            logger.info(f"[{self.name}] News features adicionadas")
        else:
            logger.debug(f"[{self.name}] NewsScraper nao disponivel para features")

        # --- Passo 4: Contexto de mercado (busca semantica de eventos similares) ---
        if self._market_context is not None:
            for coin in coins:
                if coin in featured_data:
                    try:
                        featured_data[coin] = self._market_context.add_context_features(
                            featured_data[coin], coin
                        )
                        # Armazena contexto de mercado agregado
                        if hasattr(self._market_context, "get_current_context"):
                            result.data["market_context"][coin] = (
                                self._market_context.get_current_context(coin)
                            )
                    except Exception as e:
                        msg = f"Erro market context {coin}: {e}"
                        logger.warning(f"[{self.name}] {msg}")
                        result.errors.append(msg)
            logger.info(f"[{self.name}] Market context features adicionadas")
        else:
            logger.debug(f"[{self.name}] MarketContextMemory nao disponivel")

        # --- Passo 5: Deteccao de narrativas (opcional) ---
        if self._narrative_detector is not None:
            news_data = context.get("news", {})
            for coin in coins:
                if coin in featured_data and coin in news_data:
                    try:
                        if hasattr(self._narrative_detector, "detect"):
                            narratives = self._narrative_detector.detect(news_data[coin])
                            result.data["market_context"].setdefault(coin, {})
                            result.data["market_context"][coin]["narratives"] = narratives
                    except Exception as e:
                        msg = f"Erro narrative detection {coin}: {e}"
                        logger.debug(f"[{self.name}] {msg}")
                        result.errors.append(msg)

        # --- Passo 6: Smart money tracking (opcional) ---
        if self._smart_money_tracker is not None:
            whale_data = context.get("whale_data", {})
            for coin in coins:
                if coin in featured_data:
                    try:
                        if hasattr(self._smart_money_tracker, "add_features"):
                            featured_data[coin] = self._smart_money_tracker.add_features(
                                featured_data[coin],
                                whale_data.get(coin),
                            )
                    except Exception as e:
                        msg = f"Erro smart money {coin}: {e}"
                        logger.debug(f"[{self.name}] {msg}")
                        result.errors.append(msg)

        result.data["featured_data"] = featured_data

        # Resumo
        result.metadata["coins_featured"] = list(featured_data.keys())
        result.metadata["feature_counts"] = {
            coin: len(df.columns) for coin, df in featured_data.items()
        }

        return result
