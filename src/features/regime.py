"""Deteccao de regimes de mercado usando Hidden Markov Model."""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM

    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

# ConvergenceWarning was removed in hmmlearn 0.3.x
try:
    from hmmlearn.base import ConvergenceWarning  # type: ignore[attr-defined]
except ImportError:
    ConvergenceWarning = UserWarning  # type: ignore[misc,assignment]

try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "regime_state",
    "regime_prob_bull",
    "regime_prob_bear",
    "regime_prob_sideways",
    "regime_duration",
    "regime_transition",
]


class RegimeDetector:
    """Detecta regimes de mercado (bull, bear, sideways) via HMM Gaussiano.

    Usa retornos logaritmicos para ajustar um HMM com n_regimes estados.
    Os estados sao rotulados automaticamente pela media dos retornos:
      - Maior media  -> bull
      - Menor media  -> bear
      - Intermediario -> sideways
    """

    def __init__(self, n_regimes: int = 3):
        if not HAS_HMMLEARN:
            raise ImportError(
                "hmmlearn e necessario para RegimeDetector. "
                "Instale com: pip install hmmlearn"
            )
        self.n_regimes = n_regimes
        self.model: GaussianHMM | None = None
        self._label_map: dict[int, str] | None = None
        self._fitted = False

    def fit(self, returns: np.ndarray) -> "RegimeDetector":
        """Ajusta o HMM nos retornos logaritmicos.

        Args:
            returns: Array 1-D de log-retornos.

        Returns:
            self para encadeamento.
        """
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        # Remover NaN/Inf
        mask = np.isfinite(returns).all(axis=1)
        clean = returns[mask]
        if len(clean) < self.n_regimes * 10:
            logger.warning(
                "Poucos dados para ajustar HMM (%d amostras). "
                "Resultados podem ser imprecisos.",
                len(clean),
            )

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            verbose=False,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message=".*did not converge.*")
            try:
                self.model.fit(clean)
                self._fitted = True
                logger.info(
                    "HMM ajustado com %d regimes em %d amostras.",
                    self.n_regimes,
                    len(clean),
                )
            except Exception:
                logger.exception("Falha ao ajustar HMM")
                raise

        # Rotular regimes pela media dos retornos
        self._build_label_map()
        return self

    def _build_label_map(self) -> None:
        """Mapeia indices de estado para labels (bull/bear/sideways)."""
        if self.model is None:
            return

        means = self.model.means_.flatten()
        sorted_indices = np.argsort(means)

        if self.n_regimes == 3:
            self._label_map = {
                int(sorted_indices[0]): "bear",
                int(sorted_indices[1]): "sideways",
                int(sorted_indices[2]): "bull",
            }
        elif self.n_regimes == 2:
            self._label_map = {
                int(sorted_indices[0]): "bear",
                int(sorted_indices[1]): "bull",
            }
        else:
            # Generico: menor = bear, maior = bull, resto = sideways
            self._label_map = {}
            for rank, idx in enumerate(sorted_indices):
                if rank == 0:
                    self._label_map[int(idx)] = "bear"
                elif rank == len(sorted_indices) - 1:
                    self._label_map[int(idx)] = "bull"
                else:
                    self._label_map[int(idx)] = f"sideways_{rank}"

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Prediz o regime para cada timestep.

        Args:
            returns: Array 1-D de log-retornos.

        Returns:
            Array de inteiros representando o estado HMM.
        """
        self._check_fitted()
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        mask = np.isfinite(returns).all(axis=1)
        states = np.full(len(returns), -1, dtype=int)

        if mask.sum() > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                states[mask] = self.model.predict(returns[mask])

        return states

    def _predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Retorna probabilidades posteriores para cada regime.

        Args:
            returns: Array 1-D de log-retornos.

        Returns:
            Array (n_samples, n_regimes) de probabilidades.
        """
        self._check_fitted()
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        mask = np.isfinite(returns).all(axis=1)
        proba = np.full((len(returns), self.n_regimes), np.nan)

        if mask.sum() > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba[mask] = self.model.predict_proba(returns[mask])

        return proba

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de regime ao DataFrame.

        Requer coluna 'log_return' no DataFrame.

        Features adicionadas:
          - regime_state: indice do regime (int)
          - regime_prob_bull: probabilidade do regime bull
          - regime_prob_bear: probabilidade do regime bear
          - regime_prob_sideways: probabilidade do regime sideways
          - regime_duration: quantos periodos consecutivos no regime atual
          - regime_transition: 1 se houve mudanca de regime, 0 caso contrario

        Args:
            df: DataFrame com coluna 'log_return'.

        Returns:
            DataFrame com features de regime adicionadas.
        """
        self._check_fitted()
        df = df.copy()

        if "log_return" not in df.columns:
            logger.warning(
                "Coluna 'log_return' ausente. Tentando calcular a partir de 'close'."
            )
            if "close" in df.columns:
                df["log_return"] = np.log(df["close"] / df["close"].shift(1))
            else:
                raise ValueError(
                    "DataFrame precisa de 'log_return' ou 'close' para deteccao de regime."
                )

        returns = df["log_return"].values

        # Estado do regime
        states = self.predict(returns)
        df["regime_state"] = states

        # Probabilidades por regime
        proba = self._predict_proba(returns)
        label_to_col = {"bull": "regime_prob_bull", "bear": "regime_prob_bear"}
        # Encontrar indice do sideways
        for state_idx, label in (self._label_map or {}).items():
            if label == "bull":
                df["regime_prob_bull"] = proba[:, state_idx]
            elif label == "bear":
                df["regime_prob_bear"] = proba[:, state_idx]
            elif label.startswith("sideways"):
                df["regime_prob_sideways"] = proba[:, state_idx]

        # Garantir que colunas existam mesmo com n_regimes != 3
        for col in ["regime_prob_bull", "regime_prob_bear", "regime_prob_sideways"]:
            if col not in df.columns:
                df[col] = 0.0

        # Duracao do regime atual (periodos consecutivos no mesmo estado)
        duration = np.ones(len(states), dtype=int)
        for i in range(1, len(states)):
            if states[i] == states[i - 1] and states[i] != -1:
                duration[i] = duration[i - 1] + 1
            else:
                duration[i] = 1
        df["regime_duration"] = duration

        # Transicao de regime (1 se mudou, 0 se nao)
        transition = np.zeros(len(states), dtype=int)
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                transition[i] = 1
        df["regime_transition"] = transition

        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features geradas."""
        return list(FEATURE_NAMES)

    def save(self, path: str | Path) -> None:
        """Salva o detector em disco via joblib.

        Args:
            path: Caminho do arquivo de saida.
        """
        if not HAS_JOBLIB:
            raise ImportError("joblib e necessario para salvar. pip install joblib")
        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "n_regimes": self.n_regimes,
            "model": self.model,
            "label_map": self._label_map,
        }
        joblib.dump(state, path)
        logger.info("RegimeDetector salvo em %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeDetector":
        """Carrega detector salvo previamente.

        Args:
            path: Caminho do arquivo salvo.

        Returns:
            Instancia de RegimeDetector ajustada.
        """
        if not HAS_JOBLIB:
            raise ImportError("joblib e necessario para carregar. pip install joblib")

        state = joblib.load(path)
        detector = cls(n_regimes=state["n_regimes"])
        detector.model = state["model"]
        detector._label_map = state["label_map"]
        detector._fitted = True
        logger.info("RegimeDetector carregado de %s", path)
        return detector

    def _check_fitted(self) -> None:
        """Verifica se o modelo foi ajustado."""
        if not self._fitted or self.model is None:
            raise RuntimeError(
                "RegimeDetector nao foi ajustado. Chame fit() primeiro."
            )
