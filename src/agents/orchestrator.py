"""AgentOrchestrator: coordena todos os agentes em sequencia."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src.agents.base_agent import AgentResult

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orquestrador que coordena a execucao sequencial dos agentes.

    Pipeline: DataAgent -> AnalystAgent -> TraderAgent -> RiskAgent

    Inspirado na arquitetura MAGIC do SigTech e FinRobot:
    - Cada agente recebe o contexto acumulado dos agentes anteriores
    - Se um agente falha, o pipeline continua com dados parciais
    - Logs detalhados de cada etapa para auditoria
    - Resultados completos com metadados de cada agente

    Usage:
        orchestrator = AgentOrchestrator(config=config)
        results = orchestrator.run(coins=["BTC", "ETH", "SOL"])
    """

    def __init__(self, config=None, save_results: bool = True):
        """Inicializa o orquestrador e seus agentes.

        Args:
            config: Configuracao do sistema. Se None, agentes usam defaults.
            save_results: Se deve salvar resultados em disco.
        """
        self.config = config
        self.save_results = save_results

        # Agentes inicializados sob demanda
        self._agents_initialized = False
        self._data_agent = None
        self._analyst_agent = None
        self._trader_agent = None
        self._risk_agent = None

        # Historico
        self._latest_results: dict | None = None
        self._run_history: list[dict] = []

    def _init_agents(self):
        """Inicializa todos os agentes com importacoes seguras."""
        if self._agents_initialized:
            return

        logger.info("Inicializando agentes do orquestrador...")

        try:
            from src.agents.data_agent import DataAgent
            self._data_agent = DataAgent(config=self.config)
            logger.info("  DataAgent: OK")
        except Exception as e:
            logger.error(f"  DataAgent: ERRO - {e}")

        try:
            from src.agents.analyst_agent import AnalystAgent
            self._analyst_agent = AnalystAgent(config=self.config)
            logger.info("  AnalystAgent: OK")
        except Exception as e:
            logger.error(f"  AnalystAgent: ERRO - {e}")

        try:
            from src.agents.trader_agent import TraderAgent
            self._trader_agent = TraderAgent(config=self.config)
            logger.info("  TraderAgent: OK")
        except Exception as e:
            logger.error(f"  TraderAgent: ERRO - {e}")

        try:
            from src.agents.risk_agent import RiskAgent
            self._risk_agent = RiskAgent()
            logger.info("  RiskAgent: OK")
        except Exception as e:
            logger.error(f"  RiskAgent: ERRO - {e}")

        self._agents_initialized = True
        logger.info("Agentes inicializados")

    def run(
        self,
        coins: list[str] | None = None,
        timeframe: str = "1d",
        since_days: int = 120,
        portfolio_value: float = 0,
    ) -> dict:
        """Executa o pipeline completo de agentes.

        Args:
            coins: Lista de moedas. Se None, usa config.
            timeframe: Timeframe OHLCV (default: "1d").
            since_days: Dias de historico (default: 120).
            portfolio_value: Valor atual do portfolio para gestao de risco.

        Returns:
            Dicionario com resultados completos do pipeline:
                - predictions: Previsoes por moeda
                - signals: Sinais ajustados por risco
                - risk_report: Relatorio de risco
                - agent_results: Resultados individuais de cada agente
                - metadata: Metadados do pipeline
        """
        self._init_agents()

        if coins is None:
            if self.config and hasattr(self.config, "data"):
                coins = self.config.data.coins
            else:
                coins = ["BTC", "ETH"]

        start_time = datetime.now(timezone.utc)
        logger.info(
            f"=== Pipeline Multi-Agent iniciado para {len(coins)} moedas ==="
        )

        # Contexto compartilhado entre agentes - acumula dados a cada etapa
        context = {
            "coins": coins,
            "timeframe": timeframe,
            "since_days": since_days,
            "portfolio_value": portfolio_value,
        }

        pipeline_results = {
            "timestamp": start_time.isoformat(),
            "coins": coins,
            "predictions": {},
            "signals": {},
            "risk_report": {},
            "agent_results": {},
            "metadata": {
                "start_time": start_time.isoformat(),
                "agents_executed": [],
                "errors": [],
            },
        }

        # === Etapa 1: DataAgent - Coleta de dados ===
        data_result = self._run_agent(
            self._data_agent, "data_agent", context, pipeline_results
        )
        if data_result and data_result.success:
            # Propaga dados coletados para o contexto
            context.update(data_result.data)
        elif data_result:
            # Mesmo com erros parciais, tenta continuar com dados disponiveis
            context.update(data_result.data)

        # === Etapa 2: AnalystAgent - Analise e features ===
        analyst_result = self._run_agent(
            self._analyst_agent, "analyst_agent", context, pipeline_results
        )
        if analyst_result:
            context.update(analyst_result.data)

        # === Etapa 3: TraderAgent - Previsoes e sinais ===
        trader_result = self._run_agent(
            self._trader_agent, "trader_agent", context, pipeline_results
        )
        if trader_result:
            context.update(trader_result.data)
            pipeline_results["predictions"] = trader_result.data.get(
                "predictions", {}
            )

        # === Etapa 4: RiskAgent - Gestao de risco ===
        risk_result = self._run_agent(
            self._risk_agent, "risk_agent", context, pipeline_results
        )
        if risk_result:
            pipeline_results["signals"] = risk_result.data.get(
                "risk_adjusted_signals", {}
            )
            pipeline_results["risk_report"] = risk_result.data.get(
                "risk_report", {}
            )

        # Fallback: se RiskAgent nao rodou, usa sinais brutos do TraderAgent
        if not pipeline_results["signals"] and trader_result:
            pipeline_results["signals"] = trader_result.data.get("signals", {})

        # === Finalizacao ===
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        pipeline_results["metadata"]["end_time"] = end_time.isoformat()
        pipeline_results["metadata"]["duration_seconds"] = duration
        pipeline_results["metadata"]["success"] = bool(
            pipeline_results["predictions"] or pipeline_results["signals"]
        )

        self._latest_results = pipeline_results
        self._run_history.append({
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "success": pipeline_results["metadata"]["success"],
            "n_predictions": len(pipeline_results["predictions"]),
            "n_signals": len(pipeline_results["signals"]),
            "n_errors": len(pipeline_results["metadata"]["errors"]),
            "agents_executed": pipeline_results["metadata"]["agents_executed"],
        })

        # Log resumo final
        self._log_summary(pipeline_results, duration)

        # Salvar resultados em disco
        if self.save_results:
            self._save_results(pipeline_results, start_time)

        return pipeline_results

    def _run_agent(
        self,
        agent,
        agent_name: str,
        context: dict,
        pipeline_results: dict,
    ) -> AgentResult | None:
        """Executa um agente individual com tratamento de erros.

        Se o agente falha, registra o erro e retorna None.
        O pipeline continua mesmo com falha parcial.

        Args:
            agent: Instancia do agente.
            agent_name: Nome do agente para logs.
            context: Contexto compartilhado.
            pipeline_results: Resultados acumulados do pipeline.

        Returns:
            AgentResult ou None se o agente nao esta disponivel.
        """
        if agent is None:
            msg = f"{agent_name} nao disponivel, pulando etapa"
            logger.warning(f"[orchestrator] {msg}")
            pipeline_results["metadata"]["errors"].append(msg)
            return None

        try:
            result = agent.run(context)
            pipeline_results["metadata"]["agents_executed"].append(agent_name)
            pipeline_results["agent_results"][agent_name] = {
                "success": result.success,
                "duration_seconds": result.duration_seconds,
                "errors": result.errors,
                "metadata": result.metadata,
            }

            if result.errors:
                for err in result.errors:
                    pipeline_results["metadata"]["errors"].append(
                        f"{agent_name}: {err}"
                    )

            return result

        except Exception as e:
            msg = f"Erro critico no {agent_name}: {e}"
            logger.error(f"[orchestrator] {msg}")
            pipeline_results["metadata"]["errors"].append(msg)
            pipeline_results["agent_results"][agent_name] = {
                "success": False,
                "duration_seconds": 0,
                "errors": [str(e)],
                "metadata": {},
            }
            return None

    def _log_summary(self, results: dict, duration: float):
        """Loga resumo final do pipeline."""
        n_preds = len(results["predictions"])
        n_signals = len(results["signals"])
        n_errors = len(results["metadata"]["errors"])
        agents = results["metadata"]["agents_executed"]

        logger.info(
            f"=== Pipeline Multi-Agent finalizado em {duration:.1f}s ==="
        )
        logger.info(
            f"  Agentes executados: {', '.join(agents) if agents else 'nenhum'}"
        )
        logger.info(
            f"  Previsoes: {n_preds} | Sinais: {n_signals} | Erros: {n_errors}"
        )

        # Log de cada previsao
        for coin, pred in results["predictions"].items():
            direction = pred.get("direction", "?")
            price = pred.get("predicted_price", 0)
            confidence = pred.get("confidence", 0)
            signal = results["signals"].get(coin, {})
            pos_size = signal.get("position_size", 0)
            logger.info(
                f"    {coin}: {direction} | "
                f"Preco: ${price:,.2f} | "
                f"Confianca: {confidence:.1%} | "
                f"Posicao: {pos_size:.2%}"
            )

    def _save_results(self, results: dict, start_time: datetime):
        """Salva resultados do pipeline em disco."""
        try:
            if self.config and hasattr(self.config, "data"):
                output_dir = self.config.data.predictions_dir
            else:
                output_dir = Path("data/predictions")

            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"agents_{timestamp_str}.json"

            # Prepara dados serializaveis (remove DataFrames)
            serializable = {
                "timestamp": results["timestamp"],
                "coins": results["coins"],
                "predictions": results["predictions"],
                "signals": results["signals"],
                "risk_report": results["risk_report"],
                "metadata": results["metadata"],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, default=str)

            logger.info(f"[orchestrator] Resultados salvos em {output_path}")
        except Exception as e:
            logger.warning(f"[orchestrator] Erro ao salvar resultados: {e}")

    def health_check(self) -> dict:
        """Verifica o status de todos os agentes.

        Returns:
            Dicionario com status de cada agente.
        """
        self._init_agents()

        agents = {
            "data_agent": self._data_agent,
            "analyst_agent": self._analyst_agent,
            "trader_agent": self._trader_agent,
            "risk_agent": self._risk_agent,
        }

        health = {}
        for name, agent in agents.items():
            if agent is None:
                health[name] = {
                    "status": "missing",
                    "message": f"{name} nao inicializado",
                }
            else:
                health[name] = agent.health_check()

        return health

    def get_latest_results(self) -> dict | None:
        """Retorna os resultados mais recentes do pipeline."""
        return self._latest_results

    def get_run_history(self) -> list[dict]:
        """Retorna historico de execucoes."""
        return self._run_history
