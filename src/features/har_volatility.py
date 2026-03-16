"""Modelo HAR (Heterogeneous AutoRegressive) para volatilidade realizada."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HARVolatility:
    """Calcula features de volatilidade baseadas no modelo HAR.

    O modelo HAR decompoe a volatilidade realizada em componentes de
    diferentes frequencias (diaria, semanal, mensal) para capturar a
    heterogeneidade dos horizontes de investimento.
    """

    def __init__(self, ols_window: int = 252):
        """Inicializa o modelo HAR.

        Args:
            ols_window: Janela para estimacao OLS rolling (padrao: 252 dias ~ 1 ano)
        """
        self.ols_window = ols_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula features HAR de volatilidade realizada.

        Espera coluna 'close' para calcular retornos.

        Args:
            df: DataFrame com dados OHLCV

        Returns:
            DataFrame com features HAR adicionadas
        """
        df = df.copy()

        if "close" not in df.columns:
            logger.error("Coluna 'close' nao encontrada. Impossivel calcular HAR.")
            return df

        # Retornos logaritmicos
        returns = np.log(df["close"] / df["close"].shift(1))

        # --- Realized Variance em diferentes frequencias ---

        # RV diaria: retorno^2
        df["har_rv_daily"] = returns ** 2

        # RV semanal: media dos ultimos 5 dias
        df["har_rv_weekly"] = df["har_rv_daily"].rolling(5, min_periods=3).mean()

        # RV mensal: media dos ultimos 22 dias
        df["har_rv_monthly"] = df["har_rv_daily"].rolling(22, min_periods=10).mean()

        # --- Previsao HAR via OLS rolling ---
        df["har_predicted"] = self._fit_har_rolling(
            df["har_rv_daily"],
            df["har_rv_weekly"],
            df["har_rv_monthly"],
        )

        # --- Residuo HAR (surpresa de volatilidade) ---
        df["har_residual"] = df["har_rv_daily"] - df["har_predicted"]

        # --- Componente de salto ---
        rv_median_22 = df["har_rv_daily"].rolling(22, min_periods=10).median()
        df["har_jump"] = np.maximum(0, df["har_rv_daily"] - rv_median_22)

        # --- Semivariances (decomposicao assimetrica) ---
        # Semivariance positiva: soma dos retornos^2 quando retorno > 0, rolling 22
        returns_pos_sq = (returns ** 2).where(returns > 0, 0.0)
        df["har_semivar_pos"] = returns_pos_sq.rolling(22, min_periods=10).sum()

        # Semivariance negativa: soma dos retornos^2 quando retorno < 0, rolling 22
        returns_neg_sq = (returns ** 2).where(returns < 0, 0.0)
        df["har_semivar_neg"] = returns_neg_sq.rolling(22, min_periods=10).sum()

        logger.info("Features HAR de volatilidade calculadas com sucesso")
        return df

    def _fit_har_rolling(
        self,
        rv_daily: pd.Series,
        rv_weekly: pd.Series,
        rv_monthly: pd.Series,
    ) -> pd.Series:
        """Estima modelo HAR com OLS rolling.

        HAR: rv_daily(t) = a + b1*rv_daily(t-1) + b2*rv_weekly(t-1) + b3*rv_monthly(t-1)

        Args:
            rv_daily: Realized variance diaria
            rv_weekly: Realized variance semanal
            rv_monthly: Realized variance mensal

        Returns:
            Series com previsoes HAR
        """
        n = len(rv_daily)
        predicted = pd.Series(np.nan, index=rv_daily.index)

        # Construir matriz de regressores (lag 1 para previsao)
        y = rv_daily.values
        x_daily = rv_daily.shift(1).values
        x_weekly = rv_weekly.shift(1).values
        x_monthly = rv_monthly.shift(1).values

        # Minimo de dados necessarios para OLS
        min_obs = max(50, self.ols_window // 2)

        for i in range(min_obs, n):
            # Janela de estimacao
            start = max(0, i - self.ols_window)
            end = i

            y_win = y[start:end]
            x_d = x_daily[start:end]
            x_w = x_weekly[start:end]
            x_m = x_monthly[start:end]

            # Montar matriz X com intercepto
            X = np.column_stack([
                np.ones(end - start),
                x_d,
                x_w,
                x_m,
            ])

            # Remover linhas com NaN
            mask = ~(np.isnan(y_win) | np.isnan(X).any(axis=1))
            if mask.sum() < 20:
                continue

            y_clean = y_win[mask]
            X_clean = X[mask]

            # OLS: beta = (X'X)^(-1) X'y
            try:
                XtX = X_clean.T @ X_clean
                Xty = X_clean.T @ y_clean
                beta = np.linalg.solve(XtX, Xty)

                # Previsao para o dia i
                x_pred = np.array([
                    1.0,
                    x_daily[i] if not np.isnan(x_daily[i]) else 0.0,
                    x_weekly[i] if not np.isnan(x_weekly[i]) else 0.0,
                    x_monthly[i] if not np.isnan(x_monthly[i]) else 0.0,
                ])
                predicted.iloc[i] = x_pred @ beta
            except (np.linalg.LinAlgError, ValueError):
                # Matriz singular ou outro erro numerico
                continue

        return predicted

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features HAR geradas."""
        return [
            "har_rv_daily",
            "har_rv_weekly",
            "har_rv_monthly",
            "har_predicted",
            "har_residual",
            "har_jump",
            "har_semivar_pos",
            "har_semivar_neg",
        ]
