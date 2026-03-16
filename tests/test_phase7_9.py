"""Testes para Phase 7 (MoE + FinCoT) e Phase 9 (MarketGAN)."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    close = np.maximum(close, 1000)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close + np.random.randn(n) * 50,
        "high": close + np.abs(np.random.randn(n) * 100),
        "low": close - np.abs(np.random.randn(n) * 100),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1e6) + 1e5,
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 1
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 1
    return df


# ============================================================
# Phase 7: MoE Gating Network
# ============================================================

class TestMoEEnsemble:

    def test_init_and_name(self):
        from src.models.moe_ensemble import MoEEnsemble
        model = MoEEnsemble(n_models=5, top_k=3)
        assert model.name == "moe_ensemble"

    def test_fit_predict(self):
        from src.models.moe_ensemble import MoEEnsemble
        np.random.seed(42)
        n_samples, n_models = 100, 5
        val_matrix = np.random.randn(n_samples, n_models).astype(np.float32) * 0.01
        y_val = np.random.randn(n_samples).astype(np.float32) * 0.01
        test_matrix = np.random.randn(30, n_models).astype(np.float32) * 0.01
        y_test = np.random.randn(30).astype(np.float32) * 0.01

        model = MoEEnsemble(n_models=n_models, top_k=3)
        metrics = model.fit(val_matrix, y_val, test_matrix, y_test)
        assert isinstance(metrics, dict)

        preds = model.predict(test_matrix)
        assert preds.shape == (30,)

    def test_predict_with_confidence(self):
        from src.models.moe_ensemble import MoEEnsemble
        np.random.seed(42)
        n_models = 4
        val = np.random.randn(80, n_models).astype(np.float32) * 0.01
        y = np.random.randn(80).astype(np.float32) * 0.01
        test = np.random.randn(20, n_models).astype(np.float32) * 0.01

        model = MoEEnsemble(n_models=n_models, top_k=2)
        model.fit(val, y, test, y[:20])
        preds, conf = model.predict_with_confidence(test)
        assert preds.shape == (20,)
        assert conf.shape == (20,)
        assert np.all(conf >= 0) and np.all(conf <= 1)

    def test_top_k_activation(self):
        from src.models.moe_ensemble import MoEEnsemble
        np.random.seed(42)
        n_models = 8
        val = np.random.randn(100, n_models).astype(np.float32) * 0.01
        y = np.random.randn(100).astype(np.float32) * 0.01
        test = np.random.randn(20, n_models).astype(np.float32) * 0.01

        model = MoEEnsemble(n_models=n_models, top_k=3)
        model.fit(val, y, test, y[:20])
        preds = model.predict(test)
        assert preds.shape == (20,)

    def test_save_load(self, tmp_path):
        from src.models.moe_ensemble import MoEEnsemble
        np.random.seed(42)
        n_models = 4
        val = np.random.randn(60, n_models).astype(np.float32) * 0.01
        y = np.random.randn(60).astype(np.float32) * 0.01
        test = np.random.randn(15, n_models).astype(np.float32) * 0.01

        model = MoEEnsemble(n_models=n_models, top_k=2)
        model.fit(val, y, test, y[:15])

        path = tmp_path / "moe.pt"
        model.save(path)
        loaded = MoEEnsemble.load(path)
        preds1 = model.predict(test)
        preds2 = loaded.predict(test)
        np.testing.assert_allclose(preds1, preds2, atol=1e-5)


# ============================================================
# Phase 7: FinCoT
# ============================================================

class TestFinCoT:

    def test_create_snapshot(self):
        from src.trading.fincot import FinCoTAnalyzer, MarketSnapshot
        analyzer = FinCoTAnalyzer()
        df = _make_ohlcv(50)
        snapshot = analyzer.create_snapshot("BTC", df)
        assert snapshot.coin == "BTC"
        assert snapshot.price > 0

    def test_technical_analysis_template(self):
        from src.trading.fincot import FinCoTTemplates, MarketSnapshot
        snapshot = MarketSnapshot(
            coin="BTC", price=50000, change_24h=2.5, rsi=72,
            macd_signal="bullish", regime="trending_up",
            model_predictions={"lstm": 0.003, "xgb": -0.001},
            model_confidences={"lstm": 0.7, "xgb": 0.6},
        )
        result = FinCoTTemplates.technical_analysis(snapshot)
        assert "BTC" in result
        assert "DATA-CoT" in result
        assert "CONCEPT-CoT" in result
        assert "THESIS-CoT" in result

    def test_sentiment_analysis_template(self):
        from src.trading.fincot import FinCoTTemplates, MarketSnapshot
        snapshot = MarketSnapshot(coin="ETH", fear_greed=20)
        result = FinCoTTemplates.sentiment_analysis(snapshot)
        assert "MEDO EXTREMO" in result
        assert "ETH" in result

    def test_generate_full_analysis(self):
        from src.trading.fincot import FinCoTAnalyzer
        analyzer = FinCoTAnalyzer()
        df = _make_ohlcv(50)
        analyses = analyzer.generate_full_analysis(
            "BTC", df,
            model_predictions={"lstm": 0.002, "xgb": 0.001},
            model_confidences={"lstm": 0.8, "xgb": 0.7},
        )
        assert "technical" in analyses
        assert "sentiment" in analyses
        assert "consensus" in analyses
        assert len(analyses) == 5

    def test_generate_trading_signal(self):
        from src.trading.fincot import FinCoTAnalyzer, MarketSnapshot
        analyzer = FinCoTAnalyzer()
        snapshot = MarketSnapshot(
            coin="BTC", price=50000, rsi=72, fear_greed=80,
            volatility_level="medium",
            model_predictions={"lstm": 0.003, "xgb": 0.002, "rf": 0.001},
            model_confidences={"lstm": 0.8, "xgb": 0.7, "rf": 0.6},
        )
        signal = analyzer.generate_trading_signal(snapshot)
        assert signal["direction"] in ("LONG", "SHORT", "HOLD")
        assert 0 <= signal["confidence"] <= 1
        assert signal["stop_loss"] > 0
        assert signal["take_profit"] > 0

    def test_trading_signal_no_models(self):
        from src.trading.fincot import FinCoTAnalyzer, MarketSnapshot
        analyzer = FinCoTAnalyzer()
        snapshot = MarketSnapshot(coin="BTC")
        signal = analyzer.generate_trading_signal(snapshot)
        assert signal["direction"] == "HOLD"


# ============================================================
# Phase 9: MarketGAN
# ============================================================

class TestMarketGAN:

    def test_init(self):
        from src.data.market_gan import MarketGAN
        gan = MarketGAN()
        assert gan is not None

    def test_prepare_training_data(self):
        from src.data.market_gan import MarketGAN
        df = _make_ohlcv(200)
        gan = MarketGAN(seq_len=20)
        sequences = gan.prepare_training_data(df)
        assert sequences.ndim == 3
        assert sequences.shape[1] == 20  # seq_len
        assert sequences.shape[2] == 5   # OHLCV

    def test_fit_short(self):
        from src.data.market_gan import MarketGAN
        df = _make_ohlcv(100)
        gan = MarketGAN(seq_len=20, batch_size=16)
        gan.fit(df, epochs=2, verbose=False)
        # Deve treinar sem erro

    def test_generate(self):
        from src.data.market_gan import MarketGAN
        df = _make_ohlcv(100)
        gan = MarketGAN(seq_len=20, batch_size=16)
        gan.fit(df, epochs=2, verbose=False)
        synthetic = gan.generate(n=10)
        assert synthetic.shape == (10, 20, 5)

    def test_generate_crash_scenarios(self):
        from src.data.market_gan import MarketGAN
        df = _make_ohlcv(100)
        gan = MarketGAN(seq_len=20, batch_size=16)
        gan.fit(df, epochs=2, verbose=False)
        crashes = gan.generate_crash_scenarios(n=5)
        assert crashes.shape[0] == 5
        mean_return = crashes[:, :, 0].mean()
        assert np.isfinite(mean_return)

    def test_augment_training_data(self):
        from src.data.market_gan import MarketGAN
        df = _make_ohlcv(100)
        gan = MarketGAN(seq_len=20, batch_size=16)
        gan.fit(df, epochs=2, verbose=False)
        augmented = gan.augment_training_data(df, n_synthetic=20)
        assert isinstance(augmented, pd.DataFrame)
        assert len(augmented) > len(df)


# ============================================================
# Config Integration
# ============================================================

class TestPhase79Config:

    def test_moe_config(self):
        from config.settings import Config
        cfg = Config()
        assert hasattr(cfg, "moe_ensemble")
        assert cfg.moe_ensemble.top_k == 6
        assert cfg.training.use_moe_ensemble is True

    def test_market_gan_config(self):
        from config.settings import Config
        cfg = Config()
        assert hasattr(cfg, "market_gan")
        assert cfg.market_gan.latent_dim == 32
        assert cfg.training.use_market_gan is True
