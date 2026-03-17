"""Classe base para todos os agentes do sistema multi-agent."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _safe_import(module_path: str, class_name: str):
    """Importa uma classe de forma segura, retornando None se falhar."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        logger.debug(f"{class_name} nao disponivel: {e}")
        return None


@dataclass
class AgentResult:
    """Resultado padrao retornado por cada agente.

    Attributes:
        success: Se a execucao foi bem-sucedida.
        data: Dados produzidos pelo agente.
        errors: Lista de erros encontrados durante a execucao.
        duration_seconds: Tempo de execucao em segundos.
        metadata: Informacoes adicionais sobre a execucao.
    """

    success: bool = True
    data: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Classe base para todos os agentes do sistema.

    Cada agente e responsavel por uma etapa especifica do pipeline:
    coleta de dados, analise, trading ou gestao de risco.

    Inspirado na arquitetura MAGIC (SigTech) e FinRobot,
    onde agentes especializados cooperam via passagem de contexto.
    """

    # Subclasses devem definir o nome do agente
    name: str = "base"

    def __init__(self):
        self.status: str = "idle"  # "idle", "running", "error", "done"
        self._last_result: AgentResult | None = None
        self._execution_count: int = 0

    def run(self, context: dict) -> AgentResult:
        """Executa o agente com tratamento de erros e medicao de tempo.

        Args:
            context: Dicionario com dados de entrada e resultados de agentes anteriores.

        Returns:
            AgentResult com dados produzidos, erros e metadados.
        """
        self.status = "running"
        start = time.time()
        logger.info(f"[{self.name}] Iniciando execucao...")

        try:
            result = self.execute(context)
            self.status = "done" if result.success else "error"
        except Exception as e:
            logger.error(f"[{self.name}] Erro critico: {e}")
            result = AgentResult(
                success=False,
                errors=[f"Erro critico no agente {self.name}: {e}"],
            )
            self.status = "error"

        result.duration_seconds = time.time() - start
        result.metadata["agent_name"] = self.name
        result.metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
        result.metadata["execution_number"] = self._execution_count + 1

        self._last_result = result
        self._execution_count += 1

        status_msg = "OK" if result.success else f"ERRO ({len(result.errors)} erros)"
        logger.info(
            f"[{self.name}] Finalizado: {status_msg} em {result.duration_seconds:.2f}s"
        )

        return result

    @abstractmethod
    def execute(self, context: dict) -> AgentResult:
        """Executa a tarefa do agente. Override em subclasses.

        Args:
            context: Dicionario com dados de entrada.

        Returns:
            AgentResult com dados produzidos pelo agente.
        """
        raise NotImplementedError

    def health_check(self) -> dict:
        """Verifica status do agente.

        Returns:
            Dicionario com status, nome e informacoes de execucao.
        """
        return {
            "name": self.name,
            "status": self.status,
            "execution_count": self._execution_count,
            "last_success": (
                self._last_result.success if self._last_result else None
            ),
        }
