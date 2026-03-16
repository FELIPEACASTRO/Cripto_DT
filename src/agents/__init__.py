"""Arquitetura Multi-Agent inspirada em SigTech MAGIC e FinRobot.

Agentes especializados que cooperam para coleta, analise, trading e gestao de risco.
"""

from src.agents.base_agent import BaseAgent
from src.agents.orchestrator import AgentOrchestrator

__all__ = ["BaseAgent", "AgentOrchestrator"]
