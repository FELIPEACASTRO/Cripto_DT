"""Indicadores tecnicos usando a biblioteca 'ta'."""

import pandas as pd
import ta

from config.settings import FeatureConfig, config as default_config


class TechnicalFeatures:
    """Calcula indicadores tecnicos a partir de dados OHLCV."""

    def __init__(self, feature_config: FeatureConfig | None = None):
        self.cfg = feature_config or default_config.features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona todas as features tecnicas ao DataFrame.

        Espera colunas: open, high, low, close, volume
        """
        df = df.copy()

        # --- Tendencia ---
        # MACD
        macd = ta.trend.MACD(
            df["close"],
            window_slow=self.cfg.macd_slow,
            window_fast=self.cfg.macd_fast,
            window_sign=self.cfg.macd_signal,
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # EMAs
        for period in self.cfg.ema_periods:
            df[f"ema_{period}"] = ta.trend.EMAIndicator(
                df["close"], window=period
            ).ema_indicator()

        # SMAs
        for period in self.cfg.sma_periods:
            df[f"sma_{period}"] = ta.trend.SMAIndicator(
                df["close"], window=period
            ).sma_indicator()

        # EMA crossovers (preco relativo as EMAs)
        if "ema_9" in df.columns and "ema_21" in df.columns:
            df["ema_cross_9_21"] = df["ema_9"] / df["ema_21"] - 1
        if "ema_50" in df.columns and "ema_200" in df.columns:
            df["ema_cross_50_200"] = df["ema_50"] / df["ema_200"] - 1

        # --- Momentum ---
        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(
            df["close"], window=self.cfg.rsi_period
        ).rsi()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"], window=self.cfg.stoch_period
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Williams %R
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            df["high"], df["low"], df["close"], lbp=self.cfg.williams_period
        ).williams_r()

        # --- Volatilidade ---
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            df["close"],
            window=self.cfg.bbands_period,
            window_dev=self.cfg.bbands_std,
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pband"] = bb.bollinger_pband()  # %B

        # ATR
        df["atr"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=self.cfg.atr_period
        ).average_true_range()

        # ATR normalizado pelo preco
        df["atr_pct"] = df["atr"] / df["close"]

        # --- Volume ---
        # OBV
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(
            df["close"], df["volume"]
        ).on_balance_volume()

        # Volume ratio (volume atual / media movel do volume)
        vol_sma = df["volume"].rolling(self.cfg.volume_sma_period).mean()
        df["volume_ratio"] = df["volume"] / vol_sma.replace(0, float("nan"))

        # --- Ichimoku Cloud ---
        ichimoku = ta.trend.IchimokuIndicator(
            df["high"], df["low"], window1=9, window2=26, window3=52
        )
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()
        df["ichimoku_base"] = ichimoku.ichimoku_base_line()
        df["ichimoku_conv"] = ichimoku.ichimoku_conversion_line()
        # Preco relativo a nuvem
        cloud_mid = (df["ichimoku_a"] + df["ichimoku_b"]) / 2
        df["ichimoku_cloud_dist"] = (df["close"] - cloud_mid) / df["close"]

        # --- ADX (Average Directional Index) ---
        adx = ta.trend.ADXIndicator(
            df["high"], df["low"], df["close"], window=14
        )
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # --- CCI (Commodity Channel Index) ---
        df["cci"] = ta.trend.CCIIndicator(
            df["high"], df["low"], df["close"], window=20
        ).cci()

        # --- Keltner Channel ---
        keltner = ta.volatility.KeltnerChannel(
            df["high"], df["low"], df["close"], window=20
        )
        df["keltner_upper"] = keltner.keltner_channel_hband()
        df["keltner_lower"] = keltner.keltner_channel_lband()
        df["keltner_width"] = (
            (df["keltner_upper"] - df["keltner_lower"]) / df["close"]
        )
        df["keltner_pband"] = keltner.keltner_channel_pband()

        # --- CMF (Chaikin Money Flow) ---
        df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            df["high"], df["low"], df["close"], df["volume"], window=20
        ).chaikin_money_flow()

        # --- MFI (Money Flow Index) ---
        df["mfi"] = ta.volume.MFIIndicator(
            df["high"], df["low"], df["close"], df["volume"], window=14
        ).money_flow_index()

        return df

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features geradas."""
        names = [
            "macd", "macd_signal", "macd_histogram",
            "rsi", "stoch_k", "stoch_d", "williams_r",
            "bb_upper", "bb_lower", "bb_width", "bb_pband",
            "atr", "atr_pct",
            "obv", "volume_ratio",
            "ema_cross_9_21", "ema_cross_50_200",
            # Ichimoku
            "ichimoku_a", "ichimoku_b", "ichimoku_base",
            "ichimoku_conv", "ichimoku_cloud_dist",
            # ADX
            "adx", "adx_pos", "adx_neg",
            # CCI
            "cci",
            # Keltner
            "keltner_upper", "keltner_lower", "keltner_width", "keltner_pband",
            # CMF & MFI
            "cmf", "mfi",
        ]
        for p in self.cfg.ema_periods:
            names.append(f"ema_{p}")
        for p in self.cfg.sma_periods:
            names.append(f"sma_{p}")
        return names
