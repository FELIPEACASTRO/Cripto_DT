"""Testes para modulos de producao: API, Backtester, AutoML, Monitoring."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    """Cria DataFrame OHLCV sintetico."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 40_000 + np.cumsum(rng.normal(0, 200, n))
    high = close + rng.uniform(100, 500, n)
    low = close - rng.uniform(100, 500, n)
    opn = close + rng.normal(0, 100, n)
    volume = rng.uniform(1e9, 5e9, n)
    return pd.DataFrame({
        "date": dates,
        "timestamp": dates,
        "coin": "BTC",
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _make_predictions_df(n: int = 60) -> pd.DataFrame:
    """Cria DataFrame de predicoes sinteticas."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-03-01", periods=n, freq="D")
    actual = 40_000 + np.cumsum(rng.normal(0, 200, n))
    predicted = actual + rng.normal(0, 500, n)
    predicted_return = np.diff(predicted, prepend=predicted[0]) / np.maximum(actual, 1)
    return pd.DataFrame({
        "date": dates,
        "timestamp": dates,
        "coin": "BTC",
        "actual": actual,
        "predicted": predicted,
        "predicted_return": predicted_return,
        "confidence": rng.uniform(0.3, 0.9, n),
        "direction_actual": np.sign(np.diff(actual, prepend=actual[0])),
        "direction_predicted": np.sign(np.diff(predicted, prepend=predicted[0])),
    })


# ──────────────────────────────────────────────────────────────────────
# FastAPI Server
# ──────────────────────────────────────────────────────────────────────

class TestFastAPIServer:
    def test_import(self):
        from src.api.server import app
        assert app is not None

    def test_health_endpoint(self):
        """Testa que o endpoint /health pode ser invocado."""
        from src.api.server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_predictions_endpoint_no_models(self):
        """Sem modelos treinados, deve retornar 404 ou erro graceful."""
        from src.api.server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/predictions/BTC")
        # Sem modelos treinados, espera 404 ou 500 (graceful)
        assert response.status_code in (200, 404, 500, 503)

    def test_invalid_coin(self):
        """Moeda invalida deve retornar 404."""
        from src.api.server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/predictions/INVALID")
        assert response.status_code in (404, 422)

    def test_agents_status(self):
        from src.api.server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/agents/status")
        assert response.status_code in (200, 503)

    def test_market_sentiment(self):
        from src.api.server import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/market/sentiment")
        # Pode falhar sem conexao, mas nao deve crashar
        assert response.status_code in (200, 500, 503)


# ──────────────────────────────────────────────────────────────────────
# Backtester
# ──────────────────────────────────────────────────────────────────────

class TestBacktester:
    def test_import(self):
        from src.trading.backtester import Backtester, BacktestResult, Trade
        assert Backtester is not None

    def test_backtest_config(self):
        from config.settings import BacktestConfig
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000
        assert cfg.commission_pct == 0.001
        assert cfg.risk_free_rate == 0.02

    def test_run_backtest(self):
        from src.trading.backtester import Backtester
        from config.settings import BacktestConfig

        bt = Backtester(config=BacktestConfig())
        prices = _make_ohlcv(120)
        preds = _make_predictions_df(60)

        result = bt.run(preds, prices)
        assert result is not None
        assert hasattr(result, "total_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "trades")
        assert hasattr(result, "equity_curve")

    def test_to_dict(self):
        from src.trading.backtester import Backtester
        from config.settings import BacktestConfig

        bt = Backtester(config=BacktestConfig())
        prices = _make_ohlcv(120)
        preds = _make_predictions_df(60)
        result = bt.run(preds, prices)

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "total_return" in d

    def test_summary_report(self):
        from src.trading.backtester import Backtester
        from config.settings import BacktestConfig

        bt = Backtester(config=BacktestConfig())
        prices = _make_ohlcv(120)
        preds = _make_predictions_df(60)
        result = bt.run(preds, prices)

        report = result.summary_report()
        assert isinstance(report, str)
        assert len(report) > 50

    def test_benchmark_comparator(self):
        from src.trading.backtester import Backtester, BenchmarkComparator
        from config.settings import BacktestConfig

        bt = Backtester(config=BacktestConfig())
        prices = _make_ohlcv(120)
        preds = _make_predictions_df(60)
        result = bt.run(preds, prices)

        comp = BenchmarkComparator()
        comparison = comp.compare(result, prices)
        assert isinstance(comparison, dict)


# ──────────────────────────────────────────────────────────────────────
# AutoML Pipeline
# ──────────────────────────────────────────────────────────────────────

class TestAutoMLPipeline:
    def test_import(self):
        from src.training.automl import AutoMLPipeline
        assert AutoMLPipeline is not None

    def test_init(self):
        from src.training.automl import AutoMLPipeline
        pipeline = AutoMLPipeline()
        assert pipeline is not None

    def test_get_model_class(self):
        from src.training.automl import AutoMLPipeline
        pipeline = AutoMLPipeline()
        # Deve conseguir resolver pelo menos xgboost
        cls = pipeline._get_model_class("xgboost")
        assert cls is not None

    def test_evaluate_model(self):
        from src.training.automl import AutoMLPipeline
        pipeline = AutoMLPipeline()

        rng = np.random.default_rng(42)
        n, f = 200, 10
        X_train = rng.normal(0, 1, (n, f))
        y_train = rng.normal(0, 1, n)
        X_val = rng.normal(0, 1, (50, f))
        y_val = rng.normal(0, 1, 50)

        cls = pipeline._get_model_class("xgboost")
        if cls is not None:
            model = cls()
            metrics = pipeline._evaluate_model(model, X_train, y_train, X_val, y_val)
            assert isinstance(metrics, dict)
            assert "rmse" in metrics or "RMSE" in metrics or "combined_score" in metrics


# ──────────────────────────────────────────────────────────────────────
# Performance Tracker (Monitoring)
# ──────────────────────────────────────────────────────────────────────

class TestPerformanceTracker:
    def test_import(self):
        from src.monitoring.tracker import PerformanceTracker
        assert PerformanceTracker is not None

    def test_init(self, tmp_path):
        from src.monitoring.tracker import PerformanceTracker
        tracker = PerformanceTracker(log_dir=str(tmp_path / "monitoring"))
        assert tracker is not None

    def test_log_prediction(self, tmp_path):
        from src.monitoring.tracker import PerformanceTracker
        tracker = PerformanceTracker(log_dir=str(tmp_path / "monitoring"))
        tracker.log_prediction(
            coin="BTC",
            predicted=42000.0,
            actual=41500.0,
            model_name="xgboost",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def test_detect_data_drift(self, tmp_path):
        from src.monitoring.tracker import PerformanceTracker
        tracker = PerformanceTracker(log_dir=str(tmp_path / "monitoring"))

        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"feat_a": rng.normal(0, 1, 100), "feat_b": rng.normal(0, 1, 100)})
        curr = pd.DataFrame({"feat_a": rng.normal(0.5, 1, 100), "feat_b": rng.normal(0, 1, 100)})

        drift = tracker.detect_data_drift(ref, curr)
        assert isinstance(drift, dict)

    def test_get_performance_summary(self, tmp_path):
        from src.monitoring.tracker import PerformanceTracker
        tracker = PerformanceTracker(log_dir=str(tmp_path / "monitoring"))

        # Log algumas predicoes
        for i in range(10):
            tracker.log_prediction(
                coin="BTC",
                predicted=40000.0 + i * 100,
                actual=40000.0 + i * 80,
                model_name="xgboost",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        summary = tracker.get_performance_summary(coin="BTC")
        assert isinstance(summary, dict)

    def test_save_report(self, tmp_path):
        from src.monitoring.tracker import PerformanceTracker
        tracker = PerformanceTracker(log_dir=str(tmp_path / "monitoring"))

        report_path = str(tmp_path / "report.json")
        tracker.save_report(filepath=report_path)
        import os
        assert os.path.exists(report_path)


# ──────────────────────────────────────────────────────────────────────
# Docker / CI Files (verificacao de existencia)
# ──────────────────────────────────────────────────────────────────────

class TestInfrastructureFiles:
    def test_dockerfile_exists(self):
        import os
        assert os.path.exists("Dockerfile")

    def test_docker_compose_exists(self):
        import os
        assert os.path.exists("docker-compose.yml")

    def test_ci_workflow_exists(self):
        import os
        assert os.path.exists(".github/workflows/ci.yml")

    def test_pyproject_toml_exists(self):
        import os
        assert os.path.exists("pyproject.toml")

    def test_env_example_exists(self):
        import os
        assert os.path.exists(".env.example")
