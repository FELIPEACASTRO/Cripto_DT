"""Conformal Prediction para intervalos de incerteza calibrados.

Implementa Split Conformal Prediction e Adaptive Conformal Inference (ACI)
para gerar intervalos de previsao com garantia de cobertura finita.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConformalState:
    """Estado serializavel do preditor conformal."""

    alpha: float = 0.1
    gamma: float = 0.005
    scores: list[float] = field(default_factory=list)
    q_hat: float = 0.0
    adaptive: bool = False
    alpha_t: float = 0.1  # alpha adaptativo para ACI


class ConformalPredictor:
    """Intervalos de previsao via Split Conformal Prediction.

    Parameters
    ----------
    alpha : float
        Taxa de nao-cobertura desejada.  ``alpha=0.1`` equivale a
        cobertura nominal de 90 %.
    gamma : float
        Taxa de aprendizado para Adaptive Conformal Inference (ACI).
        Se ``gamma > 0``, o metodo ``update`` ajusta ``alpha`` online
        com base na cobertura observada.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.005) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha deve estar em (0, 1)")
        self.alpha = alpha
        self.gamma = gamma

        self._scores: np.ndarray | None = None
        self._q_hat: float = 0.0

        # Estado adaptativo (ACI)
        self._alpha_t: float = alpha

    # ------------------------------------------------------------------
    # Calibracao
    # ------------------------------------------------------------------

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Calcula nonconformity scores no conjunto de calibracao.

        O score utilizado e o residuo absoluto:  ``|y_true - y_pred|``.

        Parameters
        ----------
        y_true : np.ndarray
            Valores reais do conjunto de calibracao.
        y_pred : np.ndarray
            Previsoes correspondentes.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        if len(y_true) != len(y_pred):
            raise ValueError("y_true e y_pred devem ter o mesmo tamanho")
        if len(y_true) == 0:
            raise ValueError("Conjunto de calibracao vazio")

        self._scores = np.abs(y_true - y_pred)
        self._compute_quantile()
        logger.info(
            "Calibracao conformal: %d amostras, q_hat=%.6f (alpha=%.2f)",
            len(self._scores),
            self._q_hat,
            self.alpha,
        )

    def _compute_quantile(self) -> None:
        """Calcula o quantil conformal a partir dos scores."""
        if self._scores is None or len(self._scores) == 0:
            self._q_hat = 0.0
            return

        n = len(self._scores)
        # Quantil conformal: ceil((n+1)*(1-alpha))/n — garante cobertura finita
        level = np.ceil((n + 1) * (1 - self._alpha_t)) / n
        level = np.clip(level, 0.0, 1.0)
        self._q_hat = float(np.quantile(self._scores, level))

    # ------------------------------------------------------------------
    # Predicao
    # ------------------------------------------------------------------

    def predict_intervals(
        self, y_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera intervalos de previsao a partir de previsoes pontuais.

        Parameters
        ----------
        y_pred : np.ndarray
            Previsoes pontuais do modelo base.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(lower, upper)`` — limites inferior e superior dos intervalos.
        """
        if self._scores is None:
            raise RuntimeError("Preditor nao calibrado — chame calibrate() primeiro.")

        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        lower = y_pred - self._q_hat
        upper = y_pred + self._q_hat
        return lower, upper

    # ------------------------------------------------------------------
    # Atualizacao adaptativa (ACI)
    # ------------------------------------------------------------------

    def update(self, y_true_new: float, lower: float, upper: float) -> None:
        """Atualiza alpha adaptativo (ACI) com base em uma nova observacao.

        Se a observacao cai fora do intervalo, alpha_t diminui (intervalo
        mais largo na proxima vez); caso contrario, alpha_t aumenta
        (intervalo mais estreito).

        Parameters
        ----------
        y_true_new : float
            Valor real observado.
        lower, upper : float
            Limites do intervalo previsto.
        """
        if self.gamma <= 0:
            return

        err_t = 1.0 if (y_true_new < lower or y_true_new > upper) else 0.0
        self._alpha_t = self._alpha_t + self.gamma * (self.alpha - err_t)
        # Manter alpha_t em intervalo razoavel
        self._alpha_t = float(np.clip(self._alpha_t, 0.001, 0.999))
        self._compute_quantile()

    def update_batch(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Atualiza o preditor com um lote de novas observacoes (ACI).

        Chama ``update`` sequencialmente para cada par (y_true, lower, upper).
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        for yt, yp in zip(y_true, y_pred):
            lo, hi = yp - self._q_hat, yp + self._q_hat
            self.update(yt, lo, hi)

    # ------------------------------------------------------------------
    # Avaliacao
    # ------------------------------------------------------------------

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict[str, float]:
        """Avalia a qualidade dos intervalos de previsao.

        Returns
        -------
        dict
            * ``coverage``  — fracao de observacoes dentro dos intervalos.
            * ``avg_width`` — largura media dos intervalos.
            * ``median_width`` — largura mediana.
            * ``q_hat``     — quantil conformal utilizado.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        lower = np.asarray(lower, dtype=np.float64).ravel()
        upper = np.asarray(upper, dtype=np.float64).ravel()

        covered = (y_true >= lower) & (y_true <= upper)
        widths = upper - lower

        metrics = {
            "coverage": float(np.mean(covered)),
            "avg_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
            "q_hat": float(self._q_hat),
            "alpha": float(self.alpha),
            "alpha_t": float(self._alpha_t),
            "n_samples": int(len(y_true)),
        }

        logger.info(
            "Avaliacao conformal: cobertura=%.3f, largura_media=%.6f",
            metrics["coverage"],
            metrics["avg_width"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Salva estado do preditor conformal em JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = ConformalState(
            alpha=self.alpha,
            gamma=self.gamma,
            scores=self._scores.tolist() if self._scores is not None else [],
            q_hat=self._q_hat,
            adaptive=self.gamma > 0,
            alpha_t=self._alpha_t,
        )
        path.write_text(json.dumps(state.__dict__, indent=2), encoding="utf-8")
        logger.info("Preditor conformal salvo: %s", path)

    @classmethod
    def load(cls, path: Path) -> "ConformalPredictor":
        """Carrega preditor conformal de um arquivo JSON."""
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = ConformalState(**raw)

        instance = cls(alpha=state.alpha, gamma=state.gamma)
        instance._scores = (
            np.array(state.scores, dtype=np.float64) if state.scores else None
        )
        instance._q_hat = state.q_hat
        instance._alpha_t = state.alpha_t
        logger.info("Preditor conformal carregado: %s", path)
        return instance
