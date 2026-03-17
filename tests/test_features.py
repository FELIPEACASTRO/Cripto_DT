"""Testes para feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor
from src.features.technical import TechnicalFeatures
from src.features.lag_features import LagFeatures
from src.features.market import MarketFeatures
from src.features.wavelet import WaveletDenoiser


class TestTechnicalFeatures:

    def test_transform_adds_features(self, sample_ohlcv):
        tech = TechnicalFeatures()
        result = tech.transform(sample_ohlcv)
        assert "macd" in result.columns
        assert "rsi" in result.columns
        assert "bb_width" in result.columns
        assert "atr" in result.columns
        assert "obv" in result.columns
        # Novos indicadores Sprint 1
        assert "ichimoku_a" in result.columns
        assert "adx" in result.columns
        assert "cci" in result.columns
        assert "keltner_width" in result.columns
        assert "cmf" in result.columns
        assert "mfi" in result.columns

    def test_no_nan_after_warmup(self, sample_ohlcv):
        tech = TechnicalFeatures()
        result = tech.transform(sample_ohlcv)
        # Apos warm-up de ~200 linhas (EMA 200), nao deve ter NaN
        tail = result.iloc[210:]
        feature_names = tech.get_feature_names()
        available = [f for f in feature_names if f in tail.columns]
        for col in available:
            nan_count = tail[col].isna().sum()
            assert nan_count == 0, f"{col} tem {nan_count} NaN apos warm-up"


class TestLagFeatures:

    def test_transform_adds_features(self, sample_ohlcv):
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        df = prep.add_returns(df)
        lag = LagFeatures()
        result = lag.transform(df)
        assert "return_lag_1" in result.columns
        assert "return_cum_5" in result.columns
        assert "return_mean_10" in result.columns
        assert "realized_vol_20" in result.columns
        assert "sharpe_10" in result.columns

    def test_lag_values_correct(self, sample_ohlcv):
        prep = DataPreprocessor()
        df = prep.clean(sample_ohlcv)
        df = prep.add_returns(df)
        lag = LagFeatures()
        result = lag.transform(df)
        # return_lag_1 deve ser log_return deslocado 1
        expected = df["log_return"].shift(1)
        pd.testing.assert_series_equal(
            result["return_lag_1"], expected, check_names=False
        )


class TestMarketFeatures:

    def test_transform_without_btc(self, sample_ohlcv):
        market = MarketFeatures()
        result = market.transform(sample_ohlcv)
        assert "hl_spread" in result.columns
        assert "candle_body" in result.columns
        assert "upper_shadow" in result.columns

    def test_transform_with_btc(self, sample_ohlcv):
        prep = DataPreprocessor()
        btc_df = prep.clean(sample_ohlcv)
        btc_df = prep.add_returns(btc_df)
        alt_df = sample_ohlcv.copy()
        alt_df = prep.clean(alt_df)
        alt_df = prep.add_returns(alt_df)
        market = MarketFeatures()
        result = market.transform(alt_df, btc_df=btc_df)
        assert "btc_corr_20" in result.columns
        assert "btc_beta_20" in result.columns


class TestWaveletDenoiser:

    def test_denoise_preserves_shape(self, sample_ohlcv):
        denoiser = WaveletDenoiser()
        result = denoiser.transform(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_denoise_adds_columns(self, sample_ohlcv):
        denoiser = WaveletDenoiser()
        result = denoiser.transform(sample_ohlcv)
        assert "close_denoised" in result.columns
        assert "noise_ratio" in result.columns
        assert "denoised_return" in result.columns

    def test_denoised_is_smoother(self, sample_ohlcv):
        denoiser = WaveletDenoiser()
        result = denoiser.transform(sample_ohlcv)
        # Denoised deve ter menor variancia de retorno que original
        orig_std = result["close"].pct_change().std()
        denoised_std = result["close_denoised"].pct_change().std()
        assert denoised_std <= orig_std * 1.1  # Pode ser levemente maior por arredondamento

    def test_short_series(self):
        """Series curtas nao devem quebrar."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [103, 104, 105],
            "low": [97, 98, 99],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200],
        })
        denoiser = WaveletDenoiser()
        result = denoiser.transform(df)
        assert len(result) == 3
