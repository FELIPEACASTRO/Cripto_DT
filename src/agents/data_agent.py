"""DataAgent: orquestra coleta de dados OHLCV, noticias e on-chain."""

import logging
from datetime import datetime, timezone

from src.agents.base_agent import BaseAgent, AgentResult, _safe_import

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """Agente responsavel pela coleta unificada de dados.

    Coordena multiplas fontes de dados:
    - OHLCV via DataCollector (Binance/CCXT)
    - Noticias via NewsScraper ou AdvancedNewsCrawler
    - Dados on-chain via WhaleMonitor

    Retorna um contexto unificado com todos os dados coletados,
    pronto para consumo pelo AnalystAgent.
    """

    name = "data_agent"

    def __init__(self, config=None):
        super().__init__()
        self.config = config

        # Componentes inicializados sob demanda (lazy-loading)
        self._collector = None
        self._news_scraper = None
        self._news_crawler = None
        self._whale_monitor = None
        self._initialized = False

    def _init_components(self):
        """Inicializa componentes de coleta com importacoes seguras."""
        if self._initialized:
            return

        # DataCollector - coleta OHLCV (componente principal)
        Cls = _safe_import("src.data.collector", "DataCollector")
        if Cls:
            try:
                self._collector = Cls(self.config) if self.config else Cls()
                logger.info(f"[{self.name}] DataCollector: OK")
            except Exception as e:
                logger.warning(f"[{self.name}] DataCollector: ERRO - {e}")

        # NewsScraper - coleta de noticias
        Cls = _safe_import("src.data.news_scraper", "NewsScraper")
        if Cls:
            try:
                self._news_scraper = Cls()
                logger.info(f"[{self.name}] NewsScraper: OK")
            except Exception as e:
                logger.warning(f"[{self.name}] NewsScraper: ERRO - {e}")

        # AdvancedNewsCrawler - crawler avancado de noticias (opcional)
        Cls = _safe_import("src.data.advanced_crawler", "AdvancedNewsCrawler")
        if Cls:
            try:
                self._news_crawler = Cls()
                logger.info(f"[{self.name}] AdvancedNewsCrawler: OK")
            except Exception as e:
                logger.debug(f"[{self.name}] AdvancedNewsCrawler: {e}")

        # WhaleMonitor - dados on-chain
        Cls = _safe_import("src.data.whale_monitor", "WhaleMonitor")
        if Cls:
            try:
                self._whale_monitor = Cls()
                logger.info(f"[{self.name}] WhaleMonitor: OK")
            except Exception as e:
                logger.warning(f"[{self.name}] WhaleMonitor: ERRO - {e}")

        self._initialized = True

    def execute(self, context: dict) -> AgentResult:
        """Executa coleta de dados para todas as moedas solicitadas.

        Args:
            context: Deve conter:
                - coins (list[str]): Lista de moedas (ex: ["BTC", "ETH"])
                - timeframe (str, opcional): Timeframe OHLCV (default: "1d")
                - since_days (int, opcional): Dias de historico (default: 120)

        Returns:
            AgentResult com data contendo:
                - ohlcv (dict[str, DataFrame]): Dados OHLCV por moeda
                - news (dict[str, list]): Noticias por moeda
                - whale_data (dict[str, DataFrame]): Dados on-chain por moeda
        """
        self._init_components()

        coins = context.get("coins", [])
        timeframe = context.get("timeframe", "1d")
        since_days = context.get("since_days", 120)

        if not coins:
            return AgentResult(
                success=False,
                errors=["Nenhuma moeda especificada no contexto"],
            )

        result = AgentResult()
        result.data = {
            "ohlcv": {},
            "news": {},
            "whale_data": {},
        }

        # --- Passo 1: Coleta OHLCV ---
        if self._collector is not None:
            for coin in coins:
                try:
                    df = self._collector.fetch_ohlcv(coin, timeframe, since_days=since_days)
                    if df is not None and not df.empty:
                        result.data["ohlcv"][coin] = df
                        logger.debug(f"[{self.name}] OHLCV {coin}: {len(df)} candles")
                except Exception as e:
                    msg = f"Erro OHLCV {coin}: {e}"
                    logger.warning(f"[{self.name}] {msg}")
                    result.errors.append(msg)

            n_ohlcv = len(result.data["ohlcv"])
            logger.info(f"[{self.name}] OHLCV coletado para {n_ohlcv}/{len(coins)} moedas")
        else:
            result.errors.append("DataCollector nao disponivel")
            logger.error(f"[{self.name}] DataCollector nao disponivel")

        # Sem dados OHLCV, nao vale continuar
        if not result.data["ohlcv"]:
            result.success = False
            result.errors.append("Nenhum dado OHLCV coletado")
            return result

        # --- Passo 2: Coleta de noticias ---
        news_source = self._news_crawler or self._news_scraper
        if news_source is not None:
            for coin in coins:
                try:
                    # Tenta metodo padrao de coleta de noticias
                    if hasattr(news_source, "fetch_news"):
                        articles = news_source.fetch_news(coin)
                        result.data["news"][coin] = articles
                    elif hasattr(news_source, "collect_all_news"):
                        articles = news_source.collect_all_news(coin)
                        result.data["news"][coin] = articles
                except Exception as e:
                    msg = f"Erro noticias {coin}: {e}"
                    logger.warning(f"[{self.name}] {msg}")
                    result.errors.append(msg)

            n_news = len(result.data["news"])
            logger.info(f"[{self.name}] Noticias coletadas para {n_news} moedas")
        else:
            logger.debug(f"[{self.name}] Nenhum scraper de noticias disponivel")

        # --- Passo 3: Dados on-chain (whale monitoring) ---
        if self._whale_monitor is not None:
            for coin in coins:
                try:
                    whale_df = self._whale_monitor.fetch_whale_data(coin)
                    if whale_df is not None and not whale_df.empty:
                        result.data["whale_data"][coin] = whale_df
                except Exception as e:
                    msg = f"Erro whale data {coin}: {e}"
                    logger.warning(f"[{self.name}] {msg}")
                    result.errors.append(msg)

            n_whale = len(result.data["whale_data"])
            logger.info(f"[{self.name}] Whale data coletado para {n_whale} moedas")
        else:
            logger.debug(f"[{self.name}] WhaleMonitor nao disponivel")

        # Resumo
        result.metadata["coins_with_ohlcv"] = list(result.data["ohlcv"].keys())
        result.metadata["coins_with_news"] = list(result.data["news"].keys())
        result.metadata["coins_with_whale"] = list(result.data["whale_data"].keys())

        return result
