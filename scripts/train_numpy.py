#!/usr/bin/env python3
"""Treinamento usando APENAS numpy/pandas (sem sklearn/xgboost/lightgbm).

Workaround para Windows DLL lock issue que impede import de sklearn/xgboost.
Implementa: Ridge Regression, Gradient Boosted Stumps, e Ensemble.
"""

import json
import logging
import os
import sys
from pathlib import Path

os.environ["CRIPTO_DT_NO_TORCH"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import joblib


# ============================================================
# Pure-numpy ML models
# ============================================================

class NumpyScaler:
    """StandardScaler using only numpy."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-10] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class ScalerWrapper:
    """Compativel com o predictor existente."""

    def __init__(self, scaler):
        self.scaler = scaler


class RidgeRegression:
    """Ridge Regression via closed-form solution."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.name_ = "ridge_reg"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n, d = X_train.shape
        # w = (X^T X + alpha*I)^-1 X^T y
        XtX = X_train.T @ X_train + self.alpha * np.eye(d)
        Xty = X_train.T @ y_train
        self.weights = np.linalg.solve(XtX, Xty)
        self.bias = np.mean(y_train - X_train @ self.weights)
        return {}

    def predict(self, X):
        return X @ self.weights + self.bias

    def predict_with_confidence(self, X):
        preds = self.predict(X)
        conf = np.ones(len(preds)) * 0.6
        return preds, conf

    @property
    def name(self):
        return self.name_

    def save(self, path):
        joblib.dump({"weights": self.weights, "bias": self.bias, "alpha": self.alpha}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        model = cls(alpha=data["alpha"])
        model.weights = data["weights"]
        model.bias = data["bias"]
        return model

    def get_feature_importance(self):
        if self.weights is None:
            return {}
        importance = np.abs(self.weights)
        return {str(i): float(v) for i, v in enumerate(importance)}


class ElasticNetRegression:
    """ElasticNet via coordinate descent (pure numpy)."""

    def __init__(self, alpha=0.01, l1_ratio=0.5, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.name_ = "enet_reg"

    def _soft_threshold(self, x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t, 0)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n, d = X_train.shape
        self.bias = np.mean(y_train)
        residual = y_train - self.bias
        self.weights = np.zeros(d)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)

        for _ in range(self.max_iter):
            old_weights = self.weights.copy()
            for j in range(d):
                residual += X_train[:, j] * self.weights[j]
                rho = X_train[:, j] @ residual / n
                denom = (X_train[:, j] ** 2).sum() / n + l2
                self.weights[j] = self._soft_threshold(rho, l1) / denom
                residual -= X_train[:, j] * self.weights[j]

            if np.max(np.abs(self.weights - old_weights)) < self.tol:
                break
        return {}

    def predict(self, X):
        return X @ self.weights + self.bias

    def predict_with_confidence(self, X):
        preds = self.predict(X)
        return preds, np.ones(len(preds)) * 0.55

    @property
    def name(self):
        return self.name_

    def save(self, path):
        joblib.dump({
            "weights": self.weights, "bias": self.bias,
            "alpha": self.alpha, "l1_ratio": self.l1_ratio,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        model = cls(alpha=data["alpha"], l1_ratio=data["l1_ratio"])
        model.weights = data["weights"]
        model.bias = data["bias"]
        return model


class GradientBoostedStumps:
    """Gradient Boosted Decision Stumps (pure numpy).

    Each weak learner is a single split on one feature.
    """

    def __init__(self, n_estimators=200, learning_rate=0.05, max_features=0.8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.stumps = []
        self.init_pred = 0.0
        self.name_ = "gbs_reg"

    def _find_best_split(self, X, residuals, feature_subset):
        """Find best split among feature subset."""
        best_gain = -np.inf
        best_feature = 0
        best_threshold = 0.0
        best_left_val = 0.0
        best_right_val = 0.0
        n = len(residuals)

        for j in feature_subset:
            col = X[:, j]
            # Use percentile-based thresholds for speed
            thresholds = np.percentile(col, [20, 40, 50, 60, 80])
            for t in thresholds:
                left_mask = col <= t
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                if n_left < 5 or n_right < 5:
                    continue
                left_mean = residuals[left_mask].mean()
                right_mean = residuals[right_mask].mean()
                # MSE reduction
                gain = (n_left * left_mean ** 2 + n_right * right_mean ** 2) / n
                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = t
                    best_left_val = left_mean
                    best_right_val = right_mean

        return best_feature, best_threshold, best_left_val, best_right_val

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n, d = X_train.shape
        self.init_pred = float(np.mean(y_train))
        predictions = np.full(n, self.init_pred)
        n_features = max(1, int(d * self.max_features))

        for i in range(self.n_estimators):
            residuals = y_train - predictions
            feature_subset = np.random.choice(d, size=n_features, replace=False)
            feat, thresh, left_val, right_val = self._find_best_split(
                X_train, residuals, feature_subset
            )
            self.stumps.append((feat, thresh, left_val, right_val))
            mask = X_train[:, feat] <= thresh
            update = np.where(mask, left_val, right_val)
            predictions += self.learning_rate * update

        return {}

    def predict(self, X):
        predictions = np.full(X.shape[0], self.init_pred)
        for feat, thresh, left_val, right_val in self.stumps:
            mask = X[:, feat] <= thresh
            update = np.where(mask, left_val, right_val)
            predictions += self.learning_rate * update
        return predictions

    def predict_with_confidence(self, X):
        preds = self.predict(X)
        return preds, np.ones(len(preds)) * 0.6

    @property
    def name(self):
        return self.name_

    def save(self, path):
        joblib.dump({
            "stumps": self.stumps, "init_pred": self.init_pred,
            "learning_rate": self.learning_rate, "n_estimators": self.n_estimators,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        model = cls(
            n_estimators=data["n_estimators"],
            learning_rate=data["learning_rate"],
        )
        model.stumps = data["stumps"]
        model.init_pred = data["init_pred"]
        return model

    def get_feature_importance(self):
        counts = {}
        for feat, _, _, _ in self.stumps:
            counts[feat] = counts.get(feat, 0) + 1
        total = len(self.stumps) if self.stumps else 1
        return {str(k): v / total for k, v in sorted(counts.items(), key=lambda x: -x[1])}


class KNNRegressor:
    """K-Nearest Neighbors Regressor (pure numpy)."""

    def __init__(self, k=10):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.name_ = "knn_reg"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        return {}

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            dists = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
            idx = np.argsort(dists)[:self.k]
            # Distance-weighted
            w = 1.0 / (dists[idx] + 1e-10)
            preds[i] = np.average(self.y_train[idx], weights=w)
        return preds

    def predict_with_confidence(self, X):
        preds = self.predict(X)
        return preds, np.ones(len(preds)) * 0.55

    @property
    def name(self):
        return self.name_

    def save(self, path):
        joblib.dump({
            "X_train": self.X_train, "y_train": self.y_train, "k": self.k,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        model = cls(k=data["k"])
        model.X_train = data["X_train"]
        model.y_train = data["y_train"]
        return model


# ============================================================
# Walk-forward validation (pure numpy)
# ============================================================

def walk_forward_split(X, y, n_splits=5, train_ratio=0.7, val_ratio=0.15):
    """Expanding window walk-forward splits."""
    n = len(X)
    min_train = max(50, int(n * 0.3))
    test_size = int(n * (1 - train_ratio - val_ratio) / n_splits)
    test_size = max(test_size, 10)

    splits = []
    for i in range(n_splits):
        test_end = n - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        val_size = int(test_size * val_ratio / (1 - train_ratio - val_ratio + 0.001))
        val_size = max(val_size, 5)
        val_start = test_start - val_size
        train_end = val_start

        if train_end < min_train:
            continue

        splits.append((
            X[:train_end], y[:train_end],
            X[val_start:test_start], y[val_start:test_start],
            X[test_start:test_end], y[test_start:test_end],
        ))

    return splits


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred, prefix=""):
    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}dir_accuracy": dir_acc,
    }


# ============================================================
# Feature engineering (minimal, numpy-only compatible)
# ============================================================

def generate_features(df, coin, btc_df=None):
    """Generate features using only pandas/numpy (no sklearn dependency)."""
    from src.data.preprocessor import DataPreprocessor
    from src.features.technical import TechnicalFeatures
    from src.features.lag_features import LagFeatures
    from src.features.market import MarketFeatures

    NON_FEATURE_COLS = {
        "timestamp", "open", "high", "low", "close", "volume",
        "log_return", "pct_return", "direction", "target",
        "close_denoised", "high_denoised", "low_denoised", "open_denoised",
    }

    preprocessor = DataPreprocessor()
    df = preprocessor.prepare(df, horizon=1, target_type="log_return")

    try:
        from src.features.wavelet import WaveletDenoiser
        df = WaveletDenoiser().transform(df)
    except Exception:
        pass

    from config.settings import config
    df = TechnicalFeatures(config.features).transform(df)
    df = LagFeatures(config.features).transform(df)
    btc_ref = None if coin == "BTC" else btc_df
    df = MarketFeatures(config.features).transform(df, btc_df=btc_ref)

    # Optional feature modules
    for mod_path, cls_name in [
        ("src.features.volatility", "VolatilityFeatures"),
        ("src.features.har_volatility", "HARVolatility"),
        ("src.features.bubble", "BubbleDetector"),
        ("src.features.anomaly", "AnomalyDetector"),
        ("src.features.chart_patterns", "ChartPatternDetector"),
        ("src.features.regional_intelligence", "RegionalIntelligence"),
        ("src.features.finbert_sentiment", "NLPSentimentFeatures"),
    ]:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            df = cls().transform(df)
        except Exception:
            pass

    # Cleanup
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    nan_ratio = df[feature_cols].isna().mean()
    cols_to_drop = nan_ratio[nan_ratio > 0.5].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)

    if "target" in df.columns:
        df = df.dropna(subset=["target"]).reset_index(drop=True)

    return df


# ============================================================
# Main training
# ============================================================

def train_coin(coin, df_features, feature_columns):
    """Train all numpy models for one coin."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Treinando {coin}")
    logger.info(f"{'='*60}")

    X = df_features[feature_columns].values.astype(np.float64)
    y = df_features["target"].values.astype(np.float64)
    logger.info(f"Dados: {X.shape[0]} amostras, {X.shape[1]} features")

    splits = walk_forward_split(X, y, n_splits=5)
    if not splits:
        logger.warning(f"  {coin}: dados insuficientes para walk-forward")
        return None

    all_fold_metrics = []
    best_models = {}
    best_scaler = None

    for fold_idx, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        scaler = NumpyScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        fold_preds = {}
        fold_val_preds = {}
        fold_metrics = {"fold": fold_idx}

        # --- Ridge Regression (multiple alphas) ---
        for alpha in [0.1, 1.0, 10.0]:
            name = f"ridge_a{alpha}"
            logger.info(f"  Treinando {name}...")
            model = RidgeRegression(alpha=alpha)
            model.fit(X_train_s, y_train)
            fold_val_preds[name] = model.predict(X_val_s)
            fold_preds[name] = model.predict(X_test_s)

        # --- ElasticNet ---
        logger.info("  Treinando ElasticNet...")
        enet = ElasticNetRegression(alpha=0.01, l1_ratio=0.5)
        enet.fit(X_train_s, y_train)
        fold_val_preds["enet_reg"] = enet.predict(X_val_s)
        fold_preds["enet_reg"] = enet.predict(X_test_s)

        # --- Gradient Boosted Stumps ---
        for n_est, lr in [(100, 0.05), (200, 0.03), (300, 0.02)]:
            name = f"gbs_{n_est}"
            logger.info(f"  Treinando GBS (n={n_est}, lr={lr})...")
            gbs = GradientBoostedStumps(n_estimators=n_est, learning_rate=lr)
            gbs.fit(X_train_s, y_train)
            fold_val_preds[name] = gbs.predict(X_val_s)
            fold_preds[name] = gbs.predict(X_test_s)

        # --- KNN ---
        logger.info("  Treinando KNN...")
        knn = KNNRegressor(k=10)
        knn.fit(X_train_s, y_train)
        fold_val_preds["knn_reg"] = knn.predict(X_val_s)
        fold_preds["knn_reg"] = knn.predict(X_test_s)

        # --- Ensemble ---
        logger.info("  Construindo Ensemble...")
        model_names = list(fold_val_preds.keys())
        min_val = min(len(v) for v in fold_val_preds.values())
        min_test = min(len(v) for v in fold_preds.values())

        val_matrix = np.column_stack([v[-min_val:] for v in fold_val_preds.values()])
        test_matrix = np.column_stack([v[-min_test:] for v in fold_preds.values()])
        y_val_a = y_val[-min_val:]
        y_test_a = y_test[-min_test:]

        # Weights from validation directional accuracy
        val_dir_accs = np.array([
            float(np.mean(np.sign(fold_val_preds[n][-min_val:]) == np.sign(y_val_a)))
            for n in model_names
        ])
        weights = val_dir_accs - 0.45
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()

        ensemble_preds = test_matrix @ weights

        fold_metrics.update(compute_metrics(y_test_a, ensemble_preds, "ensemble_"))
        for name, preds in fold_preds.items():
            fold_metrics.update(compute_metrics(y_test_a, preds[-min_test:], f"{name}_"))

        all_fold_metrics.append(fold_metrics)
        logger.info(
            f"  Fold {fold_idx + 1}: ensemble RMSE={fold_metrics.get('ensemble_rmse', 0):.6f}, "
            f"Dir.Acc={fold_metrics.get('ensemble_dir_accuracy', 0):.4f}"
        )
        for n in model_names:
            da = fold_metrics.get(f"{n}_dir_accuracy", 0)
            logger.info(f"    {n}: dir_acc={da:.4f}, weight={weights[model_names.index(n)]:.4f}")

    # Best models from last fold
    best_models = {
        "ridge_a0.1": RidgeRegression(alpha=0.1),
        "ridge_a1.0": RidgeRegression(alpha=1.0),
        "ridge_a10.0": RidgeRegression(alpha=10.0),
        "enet_reg": ElasticNetRegression(alpha=0.01, l1_ratio=0.5),
        "gbs_100": GradientBoostedStumps(n_estimators=100, learning_rate=0.05),
        "gbs_200": GradientBoostedStumps(n_estimators=200, learning_rate=0.03),
        "gbs_300": GradientBoostedStumps(n_estimators=300, learning_rate=0.02),
        "knn_reg": KNNRegressor(k=10),
    }

    # Retrain all on full data
    logger.info("\nRetreinando no dataset completo...")
    full_scaler = NumpyScaler()
    X_full = df_features[feature_columns].values.astype(np.float64)
    y_full = df_features["target"].values.astype(np.float64)
    X_full_s = full_scaler.fit_transform(X_full)

    for name, model in best_models.items():
        model.fit(X_full_s, y_full)

    # Ensemble data
    ensemble_data = {"model_names": model_names, "weights": weights.tolist()}

    # Average metrics
    avg_metrics = {}
    metric_keys = [k for k in all_fold_metrics[0] if k != "fold"]
    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics if key in m]
        avg_metrics[f"avg_{key}"] = float(np.mean(values))
        avg_metrics[f"std_{key}"] = float(np.std(values))

    logger.info(f"\n{coin} - Metricas medias ({len(splits)} folds):")
    for k, v in avg_metrics.items():
        if k.startswith("avg_ensemble"):
            logger.info(f"  {k}: {v:.6f}")

    return {
        "models": best_models,
        "scaler": ScalerWrapper(full_scaler),
        "fold_metrics": all_fold_metrics,
        "avg_metrics": avg_metrics,
        "feature_columns": feature_columns,
        "ensemble_data": ensemble_data,
    }


def main():
    from config.settings import config
    from src.data.storage import DataStorage

    storage = DataStorage(config)

    # Load data
    coins = config.data.coins
    data = {}
    for coin in coins:
        df = storage.load(coin, "1d", stage="raw")
        if not df.empty:
            data[coin] = df
            logger.info(f"  {coin}: {len(df)} candles")

    if not data:
        logger.error("Nenhum dado. Execute collect_data.py primeiro.")
        sys.exit(1)

    logger.info(f"\nProcessando features para {len(data)} moedas...")

    # BTC for correlation
    from src.data.preprocessor import DataPreprocessor
    btc_df = None
    if "BTC" in data:
        btc_df = DataPreprocessor().prepare(data["BTC"], horizon=1, target_type="log_return")

    NON_FEATURE_COLS = {
        "timestamp", "open", "high", "low", "close", "volume",
        "log_return", "pct_return", "direction", "target",
        "close_denoised", "high_denoised", "low_denoised", "open_denoised",
    }

    features_data = {}
    for coin, df in data.items():
        logger.info(f"Features para {coin}...")
        btc_ref = None if coin == "BTC" else btc_df
        df_feat = generate_features(df, coin, btc_ref)
        features_data[coin] = df_feat
        fc = [c for c in df_feat.columns if c not in NON_FEATURE_COLS]
        logger.info(f"  {coin}: {len(df_feat)} amostras, {len(fc)} features")

    # Align columns across all coins
    all_feature_cols = set()
    for df in features_data.values():
        all_feature_cols.update(c for c in df.columns if c not in NON_FEATURE_COLS)
    for coin in features_data:
        missing = all_feature_cols - set(features_data[coin].columns)
        for col in missing:
            features_data[coin][col] = 0.0
    feature_columns = sorted(all_feature_cols)
    logger.info(f"Features finais: {len(feature_columns)} (alinhadas)")

    # Train
    results = {}
    for coin, df in features_data.items():
        if df.empty or "target" not in df.columns:
            continue
        result = train_coin(coin, df, feature_columns)
        if result:
            results[coin] = result

    # Save
    models_dir = config.training.models_dir
    for coin, result in results.items():
        coin_dir = models_dir / coin
        coin_dir.mkdir(parents=True, exist_ok=True)

        for name, model in result["models"].items():
            model.save(coin_dir / f"{name}.joblib")

        joblib.dump(result["scaler"], coin_dir / "scaler.joblib")

        with open(coin_dir / "feature_columns.json", "w") as f:
            json.dump(result["feature_columns"], f)

        with open(coin_dir / "metrics.json", "w") as f:
            json.dump(result["avg_metrics"], f, indent=2)

        with open(coin_dir / "fold_metrics.json", "w") as f:
            json.dump(result["fold_metrics"], f, indent=2)

        with open(coin_dir / "ensemble_data.json", "w") as f:
            json.dump(result["ensemble_data"], f, indent=2)

        # Feature importance from GBS
        gbs = result["models"].get("gbs_200")
        if gbs:
            importance = gbs.get_feature_importance()
            if importance:
                # Map indices to feature names
                named_imp = {}
                for idx_str, val in importance.items():
                    idx = int(idx_str)
                    if idx < len(feature_columns):
                        named_imp[feature_columns[idx]] = val
                with open(coin_dir / "feature_importance.json", "w") as f:
                    json.dump(named_imp, f, indent=2)

        logger.info(f"Salvos em {coin_dir}")

    # Summary
    print("\n" + "=" * 70)
    print("  RESUMO DO TREINO (numpy-only)")
    print("=" * 70)
    for coin, result in results.items():
        avg = result["avg_metrics"]
        dir_acc = avg.get("avg_ensemble_dir_accuracy", 0)
        rmse = avg.get("avg_ensemble_rmse", 0)
        n_models = len(result["models"])
        n_features = len(result.get("feature_columns", []))
        print(f"\n  {coin}:")
        print(f"    Acuracia Direcional: {dir_acc:.2%}")
        print(f"    RMSE:               {rmse:.6f}")
        print(f"    Modelos:            {n_models}")
        print(f"    Features:           {n_features}")
        print(f"    Folds:              {len(result['fold_metrics'])}")
    print("\n" + "=" * 70)
    print(f"  Modelos em: {config.training.models_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
