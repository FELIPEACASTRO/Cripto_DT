"""Validacao walk-forward para series temporais."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """Um split do walk-forward."""
    fold: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


class WalkForwardValidator:
    """Implementa validacao walk-forward com janela expansiva.

    Em cada fold:
    - Training: todos os dados ate o ponto de corte
    - Validation: proxima janela (para tuning / ensemble)
    - Test: janela seguinte (avaliacao final)

    Isso garante que NUNCA usamos dados futuros para treinar.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, n_samples: int) -> list[WalkForwardSplit]:
        """Gera indices de split walk-forward.

        Args:
            n_samples: Numero total de amostras

        Returns:
            Lista de WalkForwardSplit com indices
        """
        # Tamanho minimo de treino (primeiro fold)
        min_train = int(n_samples * self.train_ratio)
        val_size = int(n_samples * self.val_ratio)
        test_size = int(n_samples * self.test_ratio)

        # Espaco disponivel para expandir
        remaining = n_samples - min_train
        if remaining < val_size + test_size:
            logger.warning(
                f"Dados insuficientes para {self.n_splits} splits. "
                f"Usando split unico."
            )
            return [WalkForwardSplit(
                fold=0,
                train_start=0,
                train_end=min_train,
                val_start=min_train,
                val_end=min_train + val_size,
                test_start=min_train + val_size,
                test_end=n_samples,
            )]

        # Step size entre folds
        step_size = max(1, (remaining - val_size - test_size) // max(1, self.n_splits - 1))

        splits = []
        for i in range(self.n_splits):
            train_end = min_train + i * step_size
            val_start = train_end
            val_end = min(val_start + val_size, n_samples)
            test_start = val_end
            test_end = min(test_start + test_size, n_samples)

            if test_end <= test_start or val_end <= val_start:
                break

            splits.append(WalkForwardSplit(
                fold=i,
                train_start=0,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            ))

        logger.info(
            f"Walk-forward: {len(splits)} folds, "
            f"val_size={val_size}, test_size={test_size}"
        )
        for s in splits:
            logger.info(
                f"  Fold {s.fold}: train[0:{s.train_end}] "
                f"val[{s.val_start}:{s.val_end}] "
                f"test[{s.test_start}:{s.test_end}]"
            )

        return splits

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[tuple[
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
    ]]:
        """Gera dados divididos para cada fold.

        Returns:
            Lista de tuplas (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        splits = self.split(len(X))
        result = []
        for s in splits:
            result.append((
                X[s.train_start:s.train_end],
                y[s.train_start:s.train_end],
                X[s.val_start:s.val_end],
                y[s.val_start:s.val_end],
                X[s.test_start:s.test_end],
                y[s.test_start:s.test_end],
            ))
        return result
