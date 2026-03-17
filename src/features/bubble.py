"""Detector de bolhas GSADF (Generalized Sup ADF).

Baseado em estudo do KAIST sobre deteccao de bolhas em criptomoedas
usando testes ADF recursivos (rolling).
"""

import logging
import os

import numpy as np
import pandas as pd

_HAS_STATSMODELS = False
if not os.environ.get("CRIPTO_DT_NO_TORCH"):
    try:
        from statsmodels.tsa.stattools import adfuller
        _HAS_STATSMODELS = True
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class BubbleDetector:
    """Detecta regimes de bolha usando teste ADF recursivo.

    Implementa uma versao simplificada do teste GSADF (Phillips, Shi & Yu, 2015)
    para identificar periodos de comportamento explosivo em precos de criptomoedas.
    """

    def __init__(
        self,
        min_window: int = 60,
        adf_regression: str = "c",
        adf_maxlag: int | None = None,
        significance_level: float = 0.05,
        sma_trend_period: int = 200,
    ):
        self.min_window = min_window
        self.adf_regression = adf_regression
        self.adf_maxlag = adf_maxlag
        self.significance_level = significance_level
        self.sma_trend_period = sma_trend_period

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de deteccao de bolha ao DataFrame.

        Espera coluna 'close' no DataFrame.

        Args:
            df: DataFrame com dados OHLCV.

        Returns:
            DataFrame com features de bolha adicionadas.
        """
        df = df.copy()
        n = len(df)

        # Inicializar colunas com zeros (robusto a series curtas)
        df["bubble_adf_stat"] = np.nan
        df["bubble_signal"] = 0
        df["bubble_duration"] = 0
        df["bubble_intensity"] = 0.0
        df["log_price_deviation"] = np.nan

        # Verificar se statsmodels esta disponivel
        if not _HAS_STATSMODELS:
            logger.info("statsmodels nao disponivel. Retornando features de bolha com zeros.")
            df["log_price_deviation"] = 0.0
            df["bubble_adf_stat"] = 0.0
            return df

        # Verificar dados minimos
        if n < self.min_window:
            logger.warning(
                "Dados insuficientes para deteccao de bolhas (%d < %d). "
                "Retornando zeros.",
                n,
                self.min_window,
            )
            return df

        # --- Log do preco para o teste ADF ---
        log_price = np.log(df["close"].values)

        # --- Teste ADF recursivo (rolling) ---
        adf_stats, critical_values = self._rolling_adf(log_price)
        df["bubble_adf_stat"] = adf_stats

        # --- Sinal de bolha: estatistica ADF > valor critico ---
        # Em series explosivas, a estatistica ADF fica positiva (ou menos negativa
        # que o valor critico), indicando nao-estacionariedade explosiva.
        # Usamos o valor critico correspondente ao nivel de significancia.
        for i in range(len(adf_stats)):
            if not np.isnan(adf_stats[i]) and not np.isnan(critical_values[i]):
                # Se a estatistica > valor critico -> comportamento explosivo
                if adf_stats[i] > critical_values[i]:
                    df.iloc[i, df.columns.get_loc("bubble_signal")] = 1

        # --- Duracao da bolha: dias consecutivos em regime explosivo ---
        df["bubble_duration"] = self._compute_duration(df["bubble_signal"].values)

        # --- Intensidade da bolha: quao acima do valor critico ---
        # Quanto mais positiva a estatistica em relacao ao critico, mais intensa
        intensity = np.where(
            ~np.isnan(adf_stats) & ~np.isnan(critical_values),
            np.maximum(0, adf_stats - critical_values),
            0.0,
        )
        df["bubble_intensity"] = intensity

        # --- Desvio do log(preco) vs tendencia de longo prazo ---
        log_close = np.log(df["close"])
        sma_log = log_close.rolling(self.sma_trend_period).mean()
        df["log_price_deviation"] = log_close - sma_log

        logger.info(
            "Deteccao de bolhas concluida. Dias em bolha: %d/%d",
            int(df["bubble_signal"].sum()),
            n,
        )

        return df

    def _rolling_adf(
        self, log_price: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Executa teste ADF em janelas moveis crescentes.

        Para cada ponto t, calcula o teste ADF na janela [0, t] (com
        tamanho minimo de self.min_window). Retorna a estatistica ADF
        e o valor critico correspondente.

        Args:
            log_price: Array com log dos precos.

        Returns:
            Tupla (adf_stats, critical_values) com arrays de mesma dimensao.
        """
        n = len(log_price)
        adf_stats = np.full(n, np.nan)
        critical_values = np.full(n, np.nan)

        # Mapear nivel de significancia para chave do dicionario do adfuller
        sig_key = f"{int(self.significance_level * 100)}%"

        for t in range(self.min_window, n):
            window_data = log_price[: t + 1]
            try:
                result = adfuller(
                    window_data,
                    regression=self.adf_regression,
                    maxlag=self.adf_maxlag,
                    autolag="AIC",
                )
                adf_stat = result[0]
                crit_values_dict = result[4]

                adf_stats[t] = adf_stat

                # Obter valor critico para o nivel de significancia
                if sig_key in crit_values_dict:
                    critical_values[t] = crit_values_dict[sig_key]
                else:
                    # Fallback: usar 5% se o nivel solicitado nao existir
                    critical_values[t] = crit_values_dict.get("5%", np.nan)

            except Exception as e:
                # Continuar em caso de erro em janela individual
                logger.debug("Erro no teste ADF na janela [0:%d]: %s", t, e)
                continue

        return adf_stats, critical_values

    @staticmethod
    def _compute_duration(signal: np.ndarray) -> np.ndarray:
        """Calcula dias consecutivos em regime de bolha.

        Args:
            signal: Array binario (0 ou 1) indicando regime de bolha.

        Returns:
            Array com contagem de dias consecutivos em bolha.
        """
        duration = np.zeros_like(signal, dtype=int)
        for i in range(len(signal)):
            if signal[i] == 1:
                duration[i] = duration[i - 1] + 1 if i > 0 else 1
            else:
                duration[i] = 0
        return duration

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features geradas."""
        return [
            "bubble_adf_stat",
            "bubble_signal",
            "bubble_duration",
            "bubble_intensity",
            "log_price_deviation",
        ]
