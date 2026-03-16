"""Features de volatilidade: modelos GARCH e estimadores classicos.

Baseado em estudos da USP e ETH Zurich sobre previsao de volatilidade
em mercados de criptomoedas.
"""

import logging

import numpy as np
import pandas as pd

# Tentar importar arch para modelos GARCH
try:
    from arch import arch_model

    _HAS_ARCH = True
except ImportError:
    _HAS_ARCH = False

logger = logging.getLogger(__name__)


class VolatilityFeatures:
    """Calcula features de volatilidade a partir de dados OHLCV.

    Gera ~8 features incluindo volatilidade GARCH(1,1), volatilidade
    realizada em multiplas janelas, z-score, e estimadores de Parkinson
    e Garman-Klass.
    """

    def __init__(
        self,
        garch_window: int = 252,
        realized_windows: tuple[int, ...] = (5, 10, 20),
        zscore_lookback: int = 60,
    ):
        self.garch_window = garch_window
        self.realized_windows = realized_windows
        self.zscore_lookback = zscore_lookback

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volatilidade ao DataFrame.

        Espera colunas: open, high, low, close (e log_return se disponivel).

        Args:
            df: DataFrame com dados OHLCV.

        Returns:
            DataFrame com features de volatilidade adicionadas.
        """
        df = df.copy()

        # Calcular retornos se nao existirem
        if "log_return" in df.columns:
            returns = df["log_return"]
        else:
            returns = np.log(df["close"] / df["close"].shift(1))
            logger.info("Coluna 'log_return' nao encontrada, calculando a partir de close.")

        # --- Volatilidade realizada em janelas moveis ---
        for window in self.realized_windows:
            col_name = f"realized_vol_{window}"
            df[col_name] = returns.rolling(window).std() * np.sqrt(365)
            logger.debug("Feature '%s' calculada.", col_name)

        # --- Razao de volatilidade (curto prazo / longo prazo) ---
        short_window = min(self.realized_windows)
        long_window = max(self.realized_windows)
        short_col = f"realized_vol_{short_window}"
        long_col = f"realized_vol_{long_window}"

        if short_col in df.columns and long_col in df.columns:
            df["vol_ratio"] = df[short_col] / df[long_col].replace(0, np.nan)
        else:
            df["vol_ratio"] = np.nan

        # --- Z-score da volatilidade de curto prazo ---
        if short_col in df.columns:
            vol_mean = df[short_col].rolling(self.zscore_lookback).mean()
            vol_std = df[short_col].rolling(self.zscore_lookback).std()
            df["vol_zscore"] = (df[short_col] - vol_mean) / vol_std.replace(0, np.nan)
        else:
            df["vol_zscore"] = np.nan

        # --- Estimador de Parkinson (baseado em high/low) ---
        # Estimador mais eficiente que close-to-close para volatilidade intradiaria
        df["parkinson_vol"] = self._parkinson_estimator(df["high"], df["low"])

        # --- Estimador de Garman-Klass (baseado em OHLC) ---
        # Combina informacao de open, high, low, close para estimativa mais precisa
        df["garman_klass_vol"] = self._garman_klass_estimator(
            df["open"], df["high"], df["low"], df["close"]
        )

        # --- GARCH(1,1) ---
        if _HAS_ARCH:
            df["garch_vol"] = self._fit_garch_rolling(returns)
        else:
            logger.warning(
                "Biblioteca 'arch' nao disponivel. "
                "Feature 'garch_vol' sera omitida. "
                "Instale com: pip install arch"
            )

        return df

    def _parkinson_estimator(
        self, high: pd.Series, low: pd.Series, window: int = 20
    ) -> pd.Series:
        """Estimador de volatilidade de Parkinson usando high/low.

        Mais eficiente que o estimador close-to-close pois utiliza
        a amplitude intradiaria.
        """
        # Parkinson: sqrt(1/(4*n*ln2) * sum(ln(H/L)^2))
        log_hl = np.log(high / low)
        log_hl_sq = log_hl ** 2
        factor = 1.0 / (4.0 * np.log(2))
        parkinson_var = factor * log_hl_sq.rolling(window).mean()
        return np.sqrt(parkinson_var) * np.sqrt(365)

    def _garman_klass_estimator(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Estimador de volatilidade de Garman-Klass usando OHLC.

        Combina a amplitude high-low com a relacao close-open para
        uma estimativa mais eficiente da volatilidade.
        """
        # GK = 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        gk_var = gk.rolling(window).mean()
        # Garantir que a variancia nao seja negativa antes de tirar raiz
        gk_var = gk_var.clip(lower=0)
        return np.sqrt(gk_var) * np.sqrt(365)

    def _fit_garch_rolling(self, returns: pd.Series) -> pd.Series:
        """Ajusta modelo GARCH(1,1) em janela movel e retorna volatilidade prevista.

        Usa a biblioteca arch para estimar os parametros do GARCH e
        gerar previsoes de volatilidade condicional.
        """
        garch_vol = pd.Series(np.nan, index=returns.index)

        # Precisamos de dados suficientes para ajustar o GARCH
        if len(returns.dropna()) < self.garch_window:
            logger.warning(
                "Dados insuficientes para GARCH (%d < %d). Retornando NaN.",
                len(returns.dropna()),
                self.garch_window,
            )
            return garch_vol

        # Escalar retornos para porcentagem (arch espera retornos em %)
        scaled_returns = returns.dropna() * 100

        try:
            # Ajustar GARCH(1,1) na serie completa
            model = arch_model(
                scaled_returns,
                vol="Garch",
                p=1,
                q=1,
                mean="Zero",
                rescale=False,
            )
            result = model.fit(disp="off", show_warning=False)

            # Volatilidade condicional (converter de volta de % para decimal)
            conditional_vol = result.conditional_volatility / 100

            # Anualizar a volatilidade
            garch_vol.loc[conditional_vol.index] = conditional_vol * np.sqrt(365)

            logger.info("GARCH(1,1) ajustado com sucesso.")
        except Exception as e:
            logger.error("Erro ao ajustar GARCH(1,1): %s", e)

        return garch_vol

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features geradas."""
        names = []

        # Volatilidade realizada
        for w in self.realized_windows:
            names.append(f"realized_vol_{w}")

        names.extend([
            "vol_ratio",
            "vol_zscore",
            "parkinson_vol",
            "garman_klass_vol",
        ])

        # GARCH so esta presente se a biblioteca estiver disponivel
        if _HAS_ARCH:
            names.append("garch_vol")

        return names
