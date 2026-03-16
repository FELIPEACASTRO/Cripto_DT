"""Decomposicao CEEMDAN/EEMD/EMD para extracao de features de series temporais.

Utiliza Empirical Mode Decomposition e variantes para decompor sinais de preco
em Intrinsic Mode Functions (IMFs), extraindo features como razao de energia,
tendencia e media movel de cada componente.
"""

import logging

import numpy as np
import pandas as pd
from PyEMD import CEEMDAN, EEMD, EMD

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"ceemdan", "eemd", "emd"}

_ROLLING_WINDOW = 10


class SignalDecomposer:
    """Decompoe series temporais via EMD e gera features por IMF.

    Parameters
    ----------
    method : str
        Metodo de decomposicao: ``"ceemdan"``, ``"eemd"`` ou ``"emd"``.
    n_imfs : int
        Numero maximo de IMFs a reter. Se a decomposicao produzir menos,
        as IMFs faltantes sao preenchidas com zeros.
    """

    def __init__(self, method: str = "ceemdan", n_imfs: int = 6) -> None:
        method = method.lower()
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Metodo '{method}' nao suportado. Use {_SUPPORTED_METHODS}"
            )
        self.method = method
        self.n_imfs = n_imfs
        self._decomposer = self._build_decomposer()
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Construcao do decompositor
    # ------------------------------------------------------------------

    def _build_decomposer(self) -> CEEMDAN | EEMD | EMD:
        """Instancia o decompositor conforme o metodo escolhido."""
        if self.method == "ceemdan":
            return CEEMDAN(trials=50, epsilon=0.005, ext_EMD=EMD())
        if self.method == "eemd":
            return EEMD(trials=50, noise_width=0.05, ext_EMD=EMD())
        return EMD()

    # ------------------------------------------------------------------
    # Decomposicao
    # ------------------------------------------------------------------

    def decompose(self, series: np.ndarray) -> np.ndarray:
        """Decompoe *series* em IMFs.

        Parameters
        ----------
        series : np.ndarray
            Serie temporal 1-D (float64).

        Returns
        -------
        np.ndarray
            Matriz de forma ``(n_imfs, len(series))``.  Se a decomposicao
            produzir menos IMFs que ``n_imfs``, as restantes sao zero.
        """
        series = np.asarray(series, dtype=np.float64)
        n = len(series)

        if n < 10:
            logger.warning(
                "Serie muito curta (%d pontos) para decomposicao — "
                "retornando zeros.",
                n,
            )
            return np.zeros((self.n_imfs, n), dtype=np.float64)

        try:
            imfs = self._decomposer(series)
        except Exception:
            logger.exception(
                "Falha na decomposicao %s — retornando zeros.", self.method
            )
            return np.zeros((self.n_imfs, n), dtype=np.float64)

        # imfs pode ter mais ou menos linhas que n_imfs
        got = imfs.shape[0]
        result = np.zeros((self.n_imfs, n), dtype=np.float64)
        copy_count = min(got, self.n_imfs)
        result[:copy_count] = imfs[:copy_count, :n]

        return result

    # ------------------------------------------------------------------
    # Geracao de features
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompoe a coluna ``close`` e gera features derivadas das IMFs.

        Features geradas por IMF *i* (0..n_imfs-1):

        * ``imf_energy_ratio_{i}`` — fracao da energia total contida na IMF.
        * ``imf_trend_{i}``        — direcao instantanea (diff) da IMF.
        * ``imf_mean_{i}``         — media movel (janela=10) da IMF.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contendo ao menos a coluna ``close``.

        Returns
        -------
        pd.DataFrame
            Copia do DataFrame original com as novas colunas.
        """
        if "close" not in df.columns:
            raise KeyError("DataFrame deve conter a coluna 'close'.")

        df = df.copy()
        close = df["close"].values.astype(np.float64)
        imfs = self.decompose(close)

        # Energia total (soma dos quadrados de cada IMF)
        energies = np.sum(imfs ** 2, axis=1)  # (n_imfs,)
        total_energy = energies.sum()
        if total_energy == 0.0:
            total_energy = 1.0  # evitar divisao por zero

        feature_names: list[str] = []
        for i in range(self.n_imfs):
            imf_series = pd.Series(imfs[i], index=df.index)

            # Razao de energia
            col_energy = f"imf_energy_ratio_{i}"
            df[col_energy] = energies[i] / total_energy
            feature_names.append(col_energy)

            # Tendencia instantanea (diferenca finita)
            col_trend = f"imf_trend_{i}"
            df[col_trend] = imf_series.diff().fillna(0.0)
            feature_names.append(col_trend)

            # Media movel da IMF
            col_mean = f"imf_mean_{i}"
            df[col_mean] = (
                imf_series
                .rolling(window=_ROLLING_WINDOW, min_periods=1)
                .mean()
            )
            feature_names.append(col_mean)

        self._feature_names = feature_names
        return df

    def get_feature_names(self) -> list[str]:
        """Retorna os nomes das features geradas na ultima chamada a ``transform``."""
        if not self._feature_names:
            # Gerar nomes genericos caso transform ainda nao tenha sido chamado
            names: list[str] = []
            for i in range(self.n_imfs):
                names.extend([
                    f"imf_energy_ratio_{i}",
                    f"imf_trend_{i}",
                    f"imf_mean_{i}",
                ])
            return names
        return list(self._feature_names)
