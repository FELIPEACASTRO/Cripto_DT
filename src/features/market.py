"""Features de mercado: correlacao com BTC, dominancia, volume relativo."""

import pandas as pd

from config.settings import FeatureConfig, config as default_config


class MarketFeatures:
    """Cria features cruzadas de mercado (ex: correlacao com BTC)."""

    def __init__(self, feature_config: FeatureConfig | None = None):
        self.cfg = feature_config or default_config.features

    def transform(
        self, df: pd.DataFrame, btc_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Adiciona features de mercado.

        Args:
            df: DataFrame da moeda alvo (com 'log_return' e 'close')
            btc_df: DataFrame do BTC (se None, pula features de correlacao)
        """
        df = df.copy()

        # High-Low spread normalizado (intraday volatility proxy)
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]

        # Close vs Open (corpo do candle normalizado)
        df["candle_body"] = (df["close"] - df["open"]) / df["open"]

        # Upper/lower shadow
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

        if btc_df is not None and "log_return" in btc_df.columns:
            # Alinhar timestamps
            btc_returns = btc_df.set_index("timestamp")["log_return"]
            df_indexed = df.set_index("timestamp")

            # Correlacao rolling com BTC
            window = self.cfg.btc_correlation_window
            aligned = pd.DataFrame({
                "coin": df_indexed["log_return"],
                "btc": btc_returns,
            }).dropna()

            if len(aligned) > window:
                corr = aligned["coin"].rolling(window).corr(aligned["btc"])
                # Mapear de volta ao df original
                df = df.set_index("timestamp")
                df[f"btc_corr_{window}"] = corr
                df = df.reset_index()

                # Beta relativo ao BTC
                btc_var = aligned["btc"].rolling(window).var()
                covar = aligned["coin"].rolling(window).cov(aligned["btc"])
                beta = covar / btc_var.replace(0, float("nan"))
                df_temp = df.set_index("timestamp")
                df_temp[f"btc_beta_{window}"] = beta
                df = df_temp.reset_index()

        return df

    def get_feature_names(self, has_btc: bool = True) -> list[str]:
        """Retorna nomes das features geradas."""
        names = ["hl_spread", "candle_body", "upper_shadow", "lower_shadow"]
        if has_btc:
            w = self.cfg.btc_correlation_window
            names.extend([f"btc_corr_{w}", f"btc_beta_{w}"])
        return names
