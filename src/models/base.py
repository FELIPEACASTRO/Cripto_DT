"""Interface abstrata para todos os modelos."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """Interface base que todos os modelos devem implementar."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Treina o modelo.

        Returns:
            Dicionario com metricas de treino/validacao
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gera previsoes."""

    @abstractmethod
    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera previsoes com estimativa de incerteza.

        Returns:
            (predictions, confidence_scores)
            confidence_scores: valores entre 0 e 1 (1 = alta confianca)
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Salva modelo em disco."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Carrega modelo do disco."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificador do modelo."""
