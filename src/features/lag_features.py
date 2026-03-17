"""Features de retornos defasados e estatisticas rolling."""

import numpy as np
import pandas as pd

from config.settings import FeatureConfig, config as default_config


class LagFeatures:
    """Cria features baseadas em retornos passados e janelas moveis."""

    def __init__(self, feature_config: FeatureConfig | None = None):
        self.cfg = feature_config or default_config.features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de lag e rolling ao DataFrame.

        Espera coluna 'log_return' e 'close' ja calculados.
        """
        df = df.copy()

        # Retornos defasados
        for lag in self.cfg.lag_periods:
            df[f"return_lag_{lag}"] = df["log_return"].shift(lag)

        # Retornos acumulados (multi-periodo)
        for lag in [3, 5, 10]:
            df[f"return_cum_{lag}"] = df["log_return"].rolling(lag).sum()

        # Rolling mean e std dos retornos
        for window in self.cfg.rolling_windows:
            df[f"return_mean_{window}"] = df["log_return"].rolling(window).mean()
            df[f"return_std_{window}"] = df["log_return"].rolling(window).std()

        # Rolling min/max ratio (quao perto do maximo/minimo recente)
        for window in self.cfg.rolling_windows:
            rolling_max = df["close"].rolling(window).max()
            rolling_min = df["close"].rolling(window).min()
            range_val = rolling_max - rolling_min
            df[f"price_position_{window}"] = (
                (df["close"] - rolling_min) / range_val.replace(0, float("nan"))
            )

        # Volatilidade realizada (rolling std dos retornos)
        df["realized_vol_10"] = df["log_return"].rolling(10).std() * np.sqrt(10)
        df["realized_vol_20"] = df["log_return"].rolling(20).std() * np.sqrt(20)

        # Momentum (retorno acumulado / volatilidade)
        for window in [10, 20]:
            cum_ret = df["log_return"].rolling(window).sum()
            vol = df["log_return"].rolling(window).std() * np.sqrt(window)
            df[f"sharpe_{window}"] = cum_ret / vol.replace(0, float("nan"))

        # Volume change
        df["volume_change"] = df["volume"].pct_change()
        df["volume_change_5"] = df["volume"].pct_change(5)

        return df

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features geradas."""
        names = []
        for lag in self.cfg.lag_periods:
            names.append(f"return_lag_{lag}")
        for lag in [3, 5, 10]:
            names.append(f"return_cum_{lag}")
        for window in self.cfg.rolling_windows:
            names.extend([
                f"return_mean_{window}",
                f"return_std_{window}",
                f"price_position_{window}",
            ])
        names.extend([
            "realized_vol_10", "realized_vol_20",
            "sharpe_10", "sharpe_20",
            "volume_change", "volume_change_5",
        ])
        return names
