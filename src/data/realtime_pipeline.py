"""Pipeline de dados em tempo real: orquestra coleta, features e previsao em loop continuo."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from config.settings import Config, config as default_config

logger = logging.getLogger(__name__)


def _safe_import(module_path: str, class_name: str):
    """Importa uma classe de forma segura, retornando None se falhar."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        logger.debug(f"{class_name} nao disponivel: {e}")
        return None


class RealtimePipeline:
    """Orquestra coleta de dados, feature engineering e previsao em tempo real.

    Executa um pipeline completo em loop continuo com intervalo configuravel,
    coordenando todos os componentes do sistema de previsao.
    """

    def __init__(self, config: Config | None = None, update_interval_minutes: int = 60):
        """Inicializa o pipeline em tempo real.

        Args:
            config: Configuracao do sistema. Se None, usa config global.
            update_interval_minutes: Intervalo entre execucoes em minutos.
        """
        self.config = config if config is not None else default_config
        self.update_interval_minutes = update_interval_minutes

        # Componentes (lazy-loaded)
        self.collector = None
        self.preprocessor = None
        self.feature_pipeline = None
        self.predictor = None
        self.news_scraper = None
        self.market_context = None
        self.signal_generator = None

        self._initialized = False
        self._latest_results: dict | None = None
        self._run_history: list[dict] = []

    def _init_components(self):
        """Inicializa todos os componentes com importacoes seguras."""
        if self._initialized:
            return

        logger.info("Inicializando componentes do pipeline em tempo real...")

        # Componentes obrigatorios
        try:
            from src.data.collector import DataCollector
            self.collector = DataCollector(self.config)
            logger.info("  collector: OK")
        except Exception as e:
            logger.error(f"  collector: ERRO - {e}")

        try:
            from src.data.preprocessor import DataPreprocessor
            self.preprocessor = DataPreprocessor()
            logger.info("  preprocessor: OK")
        except Exception as e:
            logger.error(f"  preprocessor: ERRO - {e}")

        try:
            from src.features.pipeline import FeaturePipeline
            self.feature_pipeline = FeaturePipeline(self.config)
            logger.info("  feature_pipeline: OK")
        except Exception as e:
            logger.error(f"  feature_pipeline: ERRO - {e}")

        try:
            from src.prediction.predictor import Predictor
            self.predictor = Predictor(self.config)
            logger.info("  predictor: OK")
        except Exception as e:
            logger.error(f"  predictor: ERRO - {e}")

        # Componentes opcionais
        Cls = _safe_import("src.data.news_scraper", "NewsScraper")
        if Cls:
            try:
                self.news_scraper = Cls()
                logger.info("  news_scraper: OK")
            except Exception as e:
                logger.warning(f"  news_scraper: ERRO - {e}")
        else:
            logger.debug("  news_scraper: nao disponivel")

        Cls = _safe_import("src.data.market_context", "MarketContext")
        if Cls:
            try:
                self.market_context = Cls()
                logger.info("  market_context: OK")
            except Exception as e:
                logger.warning(f"  market_context: ERRO - {e}")
        else:
            logger.debug("  market_context: nao disponivel")

        Cls = _safe_import("src.trading.signal_generator", "SignalGenerator")
        if Cls:
            try:
                self.signal_generator = Cls()
                logger.info("  signal_generator: OK")
            except Exception as e:
                logger.warning(f"  signal_generator: ERRO - {e}")
        else:
            logger.debug("  signal_generator: nao disponivel")

        self._initialized = True
        logger.info("Componentes inicializados")

    def run_once(self, coins: list[str] | None = None) -> dict:
        """Executa uma iteracao completa do pipeline.

        Args:
            coins: Lista de moedas. Se None, usa config.

        Returns:
            Dicionario com previsoes, sinais e metadados da execucao.
        """
        self._init_components()

        if coins is None:
            coins = self.config.data.coins

        start_time = datetime.now(timezone.utc)
        logger.info(f"Iniciando execucao do pipeline para {len(coins)} moedas...")

        results = {
            "timestamp": start_time.isoformat(),
            "coins": coins,
            "predictions": {},
            "signals": {},
            "metadata": {
                "start_time": start_time.isoformat(),
                "steps_completed": [],
                "errors": [],
            },
        }

        # Step 1: Coletar dados OHLCV
        raw_data = {}
        try:
            if self.collector is not None:
                for coin in coins:
                    try:
                        df = self.collector.fetch_ohlcv(coin, "1d", since_days=120)
                        if not df.empty:
                            raw_data[coin] = df
                    except Exception as e:
                        logger.warning(f"Erro ao coletar {coin}: {e}")
                        results["metadata"]["errors"].append(
                            f"coleta_{coin}: {e}"
                        )
                results["metadata"]["steps_completed"].append("coleta_ohlcv")
                logger.info(f"  Passo 1: Dados coletados para {len(raw_data)} moedas")
            else:
                logger.error("  Passo 1: collector nao disponivel")
                results["metadata"]["errors"].append("collector nao inicializado")
        except Exception as e:
            logger.error(f"  Passo 1: Erro na coleta - {e}")
            results["metadata"]["errors"].append(f"coleta: {e}")

        if not raw_data:
            logger.warning("Nenhum dado coletado, abortando execucao")
            self._finalize_run(results, start_time, success=False)
            return results

        # Step 2: Preprocessar e limpar
        cleaned_data = {}
        try:
            if self.preprocessor is not None:
                for coin, df in raw_data.items():
                    try:
                        cleaned = self.preprocessor.clean(df)
                        if not cleaned.empty:
                            cleaned_data[coin] = cleaned
                    except Exception as e:
                        logger.warning(f"Erro ao preprocessar {coin}: {e}")
                        results["metadata"]["errors"].append(
                            f"preprocess_{coin}: {e}"
                        )
                results["metadata"]["steps_completed"].append("preprocessamento")
                logger.info(f"  Passo 2: Dados preprocessados para {len(cleaned_data)} moedas")
            else:
                cleaned_data = raw_data
                logger.warning("  Passo 2: preprocessor nao disponivel, usando dados brutos")
        except Exception as e:
            logger.error(f"  Passo 2: Erro no preprocessamento - {e}")
            cleaned_data = raw_data
            results["metadata"]["errors"].append(f"preprocessamento: {e}")

        # Step 3: Feature pipeline
        featured_data = {}
        try:
            if self.feature_pipeline is not None:
                btc_df = cleaned_data.get("BTC")
                for coin, df in cleaned_data.items():
                    try:
                        is_btc = coin == "BTC"
                        ref_btc = None if is_btc else btc_df
                        featured = self.feature_pipeline.transform(
                            df, btc_df=ref_btc, is_btc=is_btc, coin=coin
                        )
                        if not featured.empty:
                            featured_data[coin] = featured
                    except Exception as e:
                        logger.warning(f"Erro no feature pipeline para {coin}: {e}")
                        results["metadata"]["errors"].append(
                            f"features_{coin}: {e}"
                        )
                results["metadata"]["steps_completed"].append("feature_engineering")
                logger.info(f"  Passo 3: Features geradas para {len(featured_data)} moedas")
            else:
                logger.warning("  Passo 3: feature_pipeline nao disponivel")
                results["metadata"]["errors"].append("feature_pipeline nao inicializado")
        except Exception as e:
            logger.error(f"  Passo 3: Erro no feature engineering - {e}")
            results["metadata"]["errors"].append(f"feature_engineering: {e}")

        # Step 4: Coletar noticias e adicionar features
        try:
            if self.news_scraper is not None:
                for coin in coins:
                    if coin in featured_data:
                        try:
                            featured_data[coin] = self.news_scraper.add_news_features(
                                featured_data[coin], coin
                            )
                        except Exception as e:
                            logger.warning(f"Erro nas news features para {coin}: {e}")
                            results["metadata"]["errors"].append(
                                f"news_{coin}: {e}"
                            )
                results["metadata"]["steps_completed"].append("news_features")
                logger.info("  Passo 4: News features adicionadas")
            else:
                logger.debug("  Passo 4: news_scraper nao disponivel, pulando")
        except Exception as e:
            logger.warning(f"  Passo 4: Erro nas news features - {e}")
            results["metadata"]["errors"].append(f"news_features: {e}")

        # Step 5: Adicionar contexto de mercado
        try:
            if self.market_context is not None:
                for coin in coins:
                    if coin in featured_data:
                        try:
                            featured_data[coin] = self.market_context.add_context_features(
                                featured_data[coin], coin
                            )
                        except Exception as e:
                            logger.warning(f"Erro no market context para {coin}: {e}")
                            results["metadata"]["errors"].append(
                                f"market_context_{coin}: {e}"
                            )
                results["metadata"]["steps_completed"].append("market_context")
                logger.info("  Passo 5: Market context features adicionadas")
            else:
                logger.debug("  Passo 5: market_context nao disponivel, pulando")
        except Exception as e:
            logger.warning(f"  Passo 5: Erro no market context - {e}")
            results["metadata"]["errors"].append(f"market_context: {e}")

        # Step 6: Gerar previsoes
        predictions = {}
        try:
            if self.predictor is not None:
                for coin in coins:
                    try:
                        df_coin = featured_data.get(coin) or raw_data.get(coin)
                        if df_coin is not None:
                            pred = self.predictor.predict_coin(coin, df=df_coin)
                            predictions[coin] = pred
                    except Exception as e:
                        logger.warning(f"Erro na previsao para {coin}: {e}")
                        results["metadata"]["errors"].append(
                            f"prediction_{coin}: {e}"
                        )
                results["predictions"] = predictions
                results["metadata"]["steps_completed"].append("previsoes")
                logger.info(f"  Passo 6: Previsoes geradas para {len(predictions)} moedas")
            else:
                logger.warning("  Passo 6: predictor nao disponivel")
                results["metadata"]["errors"].append("predictor nao inicializado")
        except Exception as e:
            logger.error(f"  Passo 6: Erro nas previsoes - {e}")
            results["metadata"]["errors"].append(f"previsoes: {e}")

        # Step 7: Gerar sinais de trading
        signals = {}
        try:
            if self.signal_generator is not None and predictions:
                for coin, pred in predictions.items():
                    try:
                        signal = self.signal_generator.generate(pred)
                        signals[coin] = signal
                    except Exception as e:
                        logger.warning(f"Erro no sinal para {coin}: {e}")
                        results["metadata"]["errors"].append(
                            f"signal_{coin}: {e}"
                        )
                results["signals"] = signals
                results["metadata"]["steps_completed"].append("sinais_trading")
                logger.info(f"  Passo 7: Sinais gerados para {len(signals)} moedas")
            else:
                logger.debug("  Passo 7: signal_generator nao disponivel, pulando")
        except Exception as e:
            logger.warning(f"  Passo 7: Erro nos sinais - {e}")
            results["metadata"]["errors"].append(f"sinais_trading: {e}")

        # Step 8: Salvar resultados
        try:
            predictions_dir = self.config.data.predictions_dir
            predictions_dir.mkdir(parents=True, exist_ok=True)

            timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
            output_path = predictions_dir / f"realtime_{timestamp_str}.json"

            # Preparar dados serializaveis
            serializable = {
                "timestamp": results["timestamp"],
                "coins": results["coins"],
                "predictions": results["predictions"],
                "signals": results["signals"],
                "metadata": results["metadata"],
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, default=str)

            results["metadata"]["steps_completed"].append("salvar_resultados")
            logger.info(f"  Passo 8: Resultados salvos em {output_path}")
        except Exception as e:
            logger.error(f"  Passo 8: Erro ao salvar resultados - {e}")
            results["metadata"]["errors"].append(f"salvar: {e}")

        # Step 9: Log resumo
        try:
            n_preds = len(predictions)
            n_signals = len(signals)
            n_errors = len(results["metadata"]["errors"])
            steps = len(results["metadata"]["steps_completed"])

            logger.info(
                f"  Resumo: {n_preds} previsoes, {n_signals} sinais, "
                f"{steps} passos completos, {n_errors} erros"
            )

            for coin, pred in predictions.items():
                direction = pred.get("direction", "?")
                price = pred.get("predicted_price", 0)
                confidence = pred.get("confidence", 0)
                logger.info(
                    f"    {coin}: {direction} | "
                    f"Preco previsto: ${price:,.2f} | "
                    f"Confianca: {confidence:.1%}"
                )
        except Exception as e:
            logger.warning(f"  Passo 9: Erro no log de resumo - {e}")

        self._finalize_run(results, start_time, success=True)
        return results

    def _finalize_run(self, results: dict, start_time: datetime, success: bool):
        """Finaliza uma execucao registrando metadados."""
        end_time = datetime.now(timezone.utc)
        duration_seconds = (end_time - start_time).total_seconds()

        results["metadata"]["end_time"] = end_time.isoformat()
        results["metadata"]["duration_seconds"] = duration_seconds
        results["metadata"]["success"] = success

        self._latest_results = results

        self._run_history.append({
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "success": success,
            "n_predictions": len(results.get("predictions", {})),
            "n_signals": len(results.get("signals", {})),
            "n_errors": len(results["metadata"].get("errors", [])),
        })

        status = "SUCESSO" if success else "FALHA"
        logger.info(
            f"Execucao finalizada: {status} em {duration_seconds:.1f}s"
        )

    def run_continuous(self, coins: list[str] | None = None):
        """Executa o pipeline em loop continuo.

        Args:
            coins: Lista de moedas. Se None, usa config.
        """
        logger.info(
            f"Iniciando pipeline continuo (intervalo: {self.update_interval_minutes} min)"
        )

        iteration = 0
        try:
            while True:
                iteration += 1
                logger.info(f"=== Iteracao {iteration} ===")

                try:
                    self.run_once(coins=coins)
                except Exception as e:
                    logger.error(f"Erro na iteracao {iteration}: {e}")
                    self._run_history.append({
                        "start_time": datetime.now(timezone.utc).isoformat(),
                        "end_time": datetime.now(timezone.utc).isoformat(),
                        "duration_seconds": 0,
                        "success": False,
                        "n_predictions": 0,
                        "n_signals": 0,
                        "n_errors": 1,
                    })

                logger.info(
                    f"Proxima execucao em {self.update_interval_minutes} minutos..."
                )
                time.sleep(self.update_interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info(
                f"Pipeline interrompido pelo usuario apos {iteration} iteracoes"
            )

    def get_latest_results(self) -> dict | None:
        """Retorna os resultados mais recentes.

        Returns:
            Dicionario com previsoes, sinais e metadados, ou None se nenhuma execucao.
        """
        return self._latest_results

    def get_run_history(self) -> list[dict]:
        """Retorna historico de execucoes.

        Returns:
            Lista de dicionarios com timestamp, duracao e status de cada execucao.
        """
        return self._run_history

    def health_check(self) -> dict:
        """Verifica o status de todos os componentes.

        Returns:
            Dicionario com nome do componente -> {"status": str, "message": str}
        """
        self._init_components()

        components = {
            "collector": self.collector,
            "preprocessor": self.preprocessor,
            "feature_pipeline": self.feature_pipeline,
            "predictor": self.predictor,
            "news_scraper": self.news_scraper,
            "market_context": self.market_context,
            "signal_generator": self.signal_generator,
        }

        health = {}
        for name, component in components.items():
            if component is None:
                health[name] = {
                    "status": "missing",
                    "message": f"{name} nao inicializado ou nao disponivel",
                }
            else:
                try:
                    # Verificar se o componente esta funcional
                    repr(component)
                    health[name] = {
                        "status": "ok",
                        "message": f"{name} inicializado e disponivel",
                    }
                except Exception as e:
                    health[name] = {
                        "status": "error",
                        "message": f"{name} com erro: {e}",
                    }

        return health
