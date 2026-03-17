"""Testes para integracoes: Feature Selection, Timeframe Fusion, EMGNN."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 200, freq: str = "1D") -> pd.DataFrame:
    """Cria DataFrame OHLCV sintetico."""
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
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


class TestFeatureSelection:

    def test_mutual_info_select(self):
        from src.features.feature_selection import FeatureSelector
        np.random.seed(42)
        X = np.random.randn(200, 20).astype(np.float32)
        y = X[:, 0] * 0.5 + X[:, 3] * 0.3 + np.random.randn(200) * 0.1
        names = [f"feat_{i}" for i in range(20)]

        selector = FeatureSelector()
        selected = selector.select_features(X, y.astype(np.float32), names, method="mutual_info")
        assert len(selected) > 0
        assert len(selected) <= 20

    def test_l1_select(self):
        from src.features.feature_selection import FeatureSelector
        np.random.seed(42)
        X = np.random.randn(200, 20).astype(np.float32)
        y = X[:, 0] * 0.5 + X[:, 3] * 0.3 + np.random.randn(200) * 0.1
        names = [f"feat_{i}" for i in range(20)]

        selector = FeatureSelector()
        selected = selector.l1_select(X, y.astype(np.float32), names)
        assert isinstance(selected, list)
        assert len(selected) <= 20

    def test_get_feature_importance(self):
        from src.features.feature_selection import FeatureSelector
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32) * 0.01
        names = [f"feat_{i}" for i in range(10)]

        selector = FeatureSelector()
        importance_df = selector.get_feature_importance(X, y, names)
        assert isinstance(importance_df, pd.DataFrame)
        assert "rf_importance" in importance_df.columns
        assert "mi_importance" in importance_df.columns
        assert len(importance_df) == 10

    def test_select_features_dispatcher(self):
        from src.features.feature_selection import FeatureSelector
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32) * 0.01
        names = [f"feat_{i}" for i in range(10)]

        selector = FeatureSelector()
        # Test dispatcher with invalid method falls back to mutual_info
        selected = selector.select_features(X, y, names, method="invalid_method")
        assert len(selected) > 0


class TestTimeframeFusion:

    def test_aggregate_1h_to_daily(self):
        from src.features.timeframe_fusion import TimeframeFusion
        df_1h = _make_ohlcv(n=480, freq="1h")  # ~20 dias de dados horarios
        tf = TimeframeFusion()
        result = tf.aggregate_timeframe(df_1h)
        assert len(result) > 0
        assert "intraday_volatility" in result.columns
        assert "intraday_range" in result.columns
        assert "n_direction_changes" in result.columns

    def test_merge_timeframes(self):
        from src.features.timeframe_fusion import TimeframeFusion
        df_1d = _make_ohlcv(n=30, freq="1D")
        df_1h = _make_ohlcv(n=720, freq="1h")  # ~30 dias
        tf = TimeframeFusion()
        result = tf.merge_timeframes(df_1d, df_1h=df_1h)
        tf1h_cols = [c for c in result.columns if c.startswith("tf1h_")]
        assert len(tf1h_cols) > 0

    def test_get_feature_names(self):
        from src.features.timeframe_fusion import TimeframeFusion
        names = TimeframeFusion.get_feature_names()
        assert len(names) == 14  # 7 features * 2 timeframes
        assert all(n.startswith("tf1h_") or n.startswith("tf4h_") for n in names)

    def test_empty_input(self):
        from src.features.timeframe_fusion import TimeframeFusion
        df_1h = _make_ohlcv(n=1, freq="1h")  # Apenas 1 registro
        tf = TimeframeFusion()
        result = tf.aggregate_timeframe(df_1h)
        assert isinstance(result, pd.DataFrame)


class TestEMGNN:

    def test_build_correlation_graph(self):
        from src.models.graph_model import DynamicGraphBuilder
        np.random.seed(42)
        returns = np.random.randn(50, 5).astype(np.float32)
        adj = DynamicGraphBuilder.build_correlation_graph(returns)
        assert adj.shape == (5, 5)
        assert np.allclose(np.diag(adj), 0.0)  # Sem auto-loops

    def test_multiscale_graphs(self):
        from src.models.graph_model import DynamicGraphBuilder
        np.random.seed(42)
        returns = np.random.randn(50, 5).astype(np.float32)
        graphs = DynamicGraphBuilder.build_multiscale_graphs(returns, windows=[5, 10])
        assert len(graphs) == 2
        assert graphs[0].shape == (5, 5)

    def test_emgnn_fit_predict(self):
        from src.models.graph_model import EMGNNModel
        np.random.seed(42)
        n_samples, seq_len, n_nodes, n_features = 40, 10, 3, 5
        X = np.random.randn(n_samples, seq_len, n_nodes, n_features).astype(np.float32)
        y = np.random.randn(n_samples, n_nodes).astype(np.float32) * 0.01

        model = EMGNNModel(
            n_nodes=n_nodes,
            hidden_size=16,
            n_scales=2,
            max_epochs=3,
            early_stop_patience=2,
            batch_size=16,
            graph_windows=[5, 10],
        )
        metrics = model.fit(X[:30], y[:30], X[30:], y[30:])
        assert "train_loss" in metrics

        preds = model.predict(X[30:])
        assert preds.shape == (10, n_nodes)

    def test_emgnn_name(self):
        from src.models.graph_model import EMGNNModel
        model = EMGNNModel()
        assert model.name == "emgnn"

    def test_emgnn_save_load(self, tmp_path):
        from src.models.graph_model import EMGNNModel
        np.random.seed(42)
        n_samples, seq_len, n_nodes, n_features = 30, 5, 3, 4
        X = np.random.randn(n_samples, seq_len, n_nodes, n_features).astype(np.float32)
        y = np.random.randn(n_samples, n_nodes).astype(np.float32) * 0.01

        model = EMGNNModel(
            n_nodes=n_nodes, hidden_size=8, n_scales=2,
            max_epochs=2, batch_size=16, graph_windows=[5, 10],
        )
        model.fit(X[:20], y[:20], X[20:], y[20:])

        path = tmp_path / "emgnn.pt"
        model.save(path)
        loaded = EMGNNModel.load(path)
        preds_orig = model.predict(X[20:])
        preds_loaded = loaded.predict(X[20:])
        np.testing.assert_allclose(preds_orig, preds_loaded, atol=1e-5)


class TestTrainerFeatureSelection:
    """Testa integracao da Feature Selection no trainer."""

    def test_apply_feature_selection_none(self):
        from config.settings import Config
        from src.training.trainer import Trainer
        cfg = Config()
        cfg.features.feature_selection_method = "none"
        trainer = Trainer(cfg)

        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        cols = [f"feat_{i}" for i in range(10)]

        X_out, cols_out = trainer._apply_feature_selection(X, y, cols)
        assert X_out.shape == X.shape
        assert cols_out == cols

    def test_apply_feature_selection_mutual_info(self):
        from config.settings import Config
        from src.training.trainer import Trainer
        cfg = Config()
        cfg.features.feature_selection_method = "mutual_info"
        trainer = Trainer(cfg)

        np.random.seed(42)
        X = np.random.randn(200, 30).astype(np.float32)
        y = X[:, 0] * 0.5 + np.random.randn(200).astype(np.float32) * 0.1
        cols = [f"feat_{i}" for i in range(30)]

        X_out, cols_out = trainer._apply_feature_selection(X, y, cols)
        assert X_out.shape[1] <= X.shape[1]
        assert len(cols_out) == X_out.shape[1]
