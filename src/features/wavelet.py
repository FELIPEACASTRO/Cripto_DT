"""Wavelet Denoising para limpeza de sinais de precos."""

import numpy as np
import pandas as pd
import pywt


class WaveletDenoiser:
    """Aplica Discrete Wavelet Transform para remover ruido dos dados de preco.

    O denoising por wavelet decompoe o sinal em componentes de diferentes
    frequencias e remove os de alta frequencia (ruido), preservando a
    tendencia e padroes ciclicos.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: int | None = None,
        threshold_mode: str = "soft",
    ):
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode

    def _denoise_series(self, series: np.ndarray) -> np.ndarray:
        """Aplica wavelet denoising em uma serie 1D."""
        if len(series) < 8:
            return series

        # Nivel automatico se nao especificado
        level = self.level or pywt.dwt_max_level(len(series), self.wavelet)
        level = min(level, pywt.dwt_max_level(len(series), self.wavelet))

        # Decompor
        coeffs = pywt.wavedec(series, self.wavelet, level=level)

        # Threshold universal (VisuShrink) nos coeficientes de detalhe
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(series)))

        # Aplicar threshold em todos os niveis de detalhe (manter aproximacao)
        denoised_coeffs = [coeffs[0]]
        for detail in coeffs[1:]:
            denoised_coeffs.append(
                pywt.threshold(detail, threshold, mode=self.threshold_mode)
            )

        # Reconstruir
        return pywt.waverec(denoised_coeffs, self.wavelet)[: len(series)]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica wavelet denoising nas colunas OHLCV.

        Cria colunas denoised e preserva as originais.
        """
        df = df.copy()

        for col in ["close", "high", "low", "open"]:
            if col in df.columns:
                values = df[col].values.astype(np.float64)
                denoised = self._denoise_series(values)
                df[f"{col}_denoised"] = denoised

        # Features derivadas do denoising
        if "close_denoised" in df.columns:
            # Diferenca entre original e denoised = proxy de ruido
            df["noise_ratio"] = (
                (df["close"] - df["close_denoised"]).abs()
                / df["close"].replace(0, np.nan)
            )
            # Tendencia suavizada (retorno do sinal limpo)
            df["denoised_return"] = np.log(
                df["close_denoised"] / df["close_denoised"].shift(1)
            )

        return df

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features geradas."""
        return [
            "close_denoised", "high_denoised", "low_denoised", "open_denoised",
            "noise_ratio", "denoised_return",
        ]
