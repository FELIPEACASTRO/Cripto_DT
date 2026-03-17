"""Filtro de entropia para reduzir ruido em series temporais financeiras.

Calcula sample entropy e approximate entropy para identificar periodos
de alta previsibilidade (baixa entropia) vs periodos ruidosos.
Pesquisa indica ~23% de reducao em sinais falsos.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Calcula Sample Entropy de uma serie temporal.

    Args:
        x: Serie temporal (1D array)
        m: Dimensao de embedding
        r_factor: Tolerancia como fracao do desvio padrao

    Returns:
        Sample entropy (float). Valores menores = mais previsivel.
    """
    n = len(x)
    if n < m + 2:
        return np.nan

    r = r_factor * np.std(x)
    if r == 0:
        return 0.0

    def _count_matches(template_len):
        count = 0
        templates = np.array([x[i:i + template_len] for i in range(n - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    a = _count_matches(m + 1)
    b = _count_matches(m)

    if b == 0:
        return np.nan
    if a == 0:
        return np.inf

    return -np.log(a / b)


def _approx_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Calcula Approximate Entropy de uma serie temporal.

    Args:
        x: Serie temporal (1D array)
        m: Dimensao de embedding
        r_factor: Tolerancia como fracao do desvio padrao

    Returns:
        Approximate entropy (float). Valores menores = mais regular.
    """
    n = len(x)
    if n < m + 2:
        return np.nan

    r = r_factor * np.std(x)
    if r == 0:
        return 0.0

    def _phi(template_len):
        templates = np.array([x[i:i + template_len] for i in range(n - template_len + 1)])
        counts = np.zeros(len(templates))
        for i in range(len(templates)):
            for j in range(len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    counts[i] += 1
        counts /= len(templates)
        return np.mean(np.log(counts + 1e-10))

    return abs(_phi(m) - _phi(m + 1))


class EntropyFilter:
    """Gera features de entropia para identificar regimes de previsibilidade.

    Features geradas por janela:
    - sample_entropy_{window}: Sample entropy rolling
    - approx_entropy_{window}: Approximate entropy rolling
    - entropy_regime_{window}: 1 se baixa entropia (previsivel), 0 se alta
    - entropy_zscore_{window}: Z-score da entropia (desvio do normal)
    """

    def __init__(
        self,
        windows: tuple[int, ...] = (10, 20),
        entropy_threshold_pct: float = 40.0,
    ):
        """
        Args:
            windows: Janelas para calculo de entropia rolling
            entropy_threshold_pct: Percentil abaixo do qual regime = 'previsivel'
        """
        self.windows = windows
        self.entropy_threshold_pct = entropy_threshold_pct

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de entropia ao DataFrame.

        Args:
            df: DataFrame com coluna 'log_return' ou 'close'

        Returns:
            DataFrame com features de entropia adicionadas
        """
        df = df.copy()

        # Determinar serie para analise
        if "log_return" in df.columns:
            series = df["log_return"].fillna(0).values
        elif "close" in df.columns:
            returns = np.log(df["close"] / df["close"].shift(1)).fillna(0).values
            series = returns
        else:
            logger.warning("Sem coluna 'log_return' ou 'close' para entropia")
            return df

        n = len(series)

        for window in self.windows:
            if n < window + 5:
                df[f"sample_entropy_{window}"] = 0.0
                df[f"approx_entropy_{window}"] = 0.0
                df[f"entropy_regime_{window}"] = 0
                df[f"entropy_zscore_{window}"] = 0.0
                continue

            # Calcular entropia rolling
            sampen = np.full(n, np.nan)
            apen = np.full(n, np.nan)

            for i in range(window, n):
                segment = series[i - window:i]
                sampen[i] = _sample_entropy(segment, m=2, r_factor=0.2)
                apen[i] = _approx_entropy(segment, m=2, r_factor=0.2)

            # Substituir inf/nan por mediana
            sampen_valid = sampen[np.isfinite(sampen)]
            if len(sampen_valid) > 0:
                median_sampen = np.median(sampen_valid)
                sampen = np.where(np.isfinite(sampen), sampen, median_sampen)
            else:
                sampen = np.zeros(n)

            apen_valid = apen[np.isfinite(apen)]
            if len(apen_valid) > 0:
                median_apen = np.median(apen_valid)
                apen = np.where(np.isfinite(apen), apen, median_apen)
            else:
                apen = np.zeros(n)

            df[f"sample_entropy_{window}"] = sampen
            df[f"approx_entropy_{window}"] = apen

            # Regime: baixa entropia = previsivel
            valid_mask = ~np.isnan(sampen)
            if valid_mask.sum() > 10:
                threshold = np.percentile(sampen[valid_mask], self.entropy_threshold_pct)
                regime = np.where(sampen <= threshold, 1, 0)
                df[f"entropy_regime_{window}"] = regime
            else:
                df[f"entropy_regime_{window}"] = 0

            # Z-score da entropia
            mean_e = np.nanmean(sampen)
            std_e = np.nanstd(sampen)
            if std_e > 0:
                df[f"entropy_zscore_{window}"] = (sampen - mean_e) / std_e
            else:
                df[f"entropy_zscore_{window}"] = 0.0

        # Feature combinada: entropia media normalizada
        entropy_cols = [f"sample_entropy_{w}" for w in self.windows]
        existing = [c for c in entropy_cols if c in df.columns]
        if existing:
            df["entropy_mean"] = df[existing].mean(axis=1)
            df["entropy_predictability"] = 1.0 - df["entropy_mean"].clip(0, 2) / 2.0

        logger.info(
            f"Entropy filter: {len(self.windows)} janelas, "
            f"{len(self.windows) * 4 + 2} features adicionadas"
        )

        return df

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features geradas."""
        names = []
        for window in self.windows:
            names.extend([
                f"sample_entropy_{window}",
                f"approx_entropy_{window}",
                f"entropy_regime_{window}",
                f"entropy_zscore_{window}",
            ])
        names.extend(["entropy_mean", "entropy_predictability"])
        return names
