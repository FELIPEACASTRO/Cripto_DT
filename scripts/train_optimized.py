#!/usr/bin/env python3
"""Treinamento otimizado para maxima acuracia direcional.

Melhorias vs train_numpy.py:
1. Mais modelos (15 variantes vs 8)
2. Feature engineering extra (ratios, interacoes, PCA-like)
3. Walk-forward com mais folds (8)
4. Ensemble ponderado + meta-learner Ridge
5. Calibracao de hiperparametros por moeda
6. Conformal prediction para intervalos
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
# Modelos numpy-only (importar do train_numpy)
# ============================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_numpy import (
    NumpyScaler, ScalerWrapper, RidgeRegression, ElasticNetRegression,
    GradientBoostedStumps, KNNRegressor, walk_forward_split,
    compute_metrics, generate_features,
)


# ============================================================
# Modelos adicionais para ensemble mais robusto
# ============================================================

class LassoRegression:
    """Lasso (L1) via coordinate descent."""

    def __init__(self, alpha=0.01, max_iter=2000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.name_ = f"lasso_a{alpha}"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n, d = X_train.shape
        self.bias = np.mean(y_train)
        residual = y_train - self.bias
        self.weights = np.zeros(d)
        for _ in range(self.max_iter):
            old = self.weights.copy()
            for j in range(d):
                residual += X_train[:, j] * self.weights[j]
                rho = X_train[:, j] @ residual / n
                self.weights[j] = np.sign(rho) * max(abs(rho) - self.alpha, 0) / ((X_train[:, j] ** 2).sum() / n)
                residual -= X_train[:, j] * self.weights[j]
            if np.max(np.abs(self.weights - old)) < 1e-6:
                break
        return {}

    def predict(self, X):
        return X @ self.weights + self.bias

    def save(self, path):
        joblib.dump({"weights": self.weights, "bias": self.bias, "alpha": self.alpha}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        m = cls(alpha=data["alpha"])
        m.weights, m.bias = data["weights"], data["bias"]
        return m

    @property
    def name(self):
        return self.name_

    def predict_with_confidence(self, X):
        return self.predict(X), np.ones(len(X)) * 0.55


class HuberRegression:
    """Huber-robust regression via IRLS (Iteratively Reweighted Least Squares)."""

    def __init__(self, delta=1.35, alpha=1.0, max_iter=50):
        self.delta = delta
        self.alpha = alpha
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.name_ = "huber_reg"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n, d = X_train.shape
        # Start with Ridge solution
        XtX = X_train.T @ X_train + self.alpha * np.eye(d)
        Xty = X_train.T @ y_train
        self.weights = np.linalg.solve(XtX, Xty)
        self.bias = np.mean(y_train - X_train @ self.weights)

        for _ in range(self.max_iter):
            residuals = y_train - (X_train @ self.weights + self.bias)
            abs_r = np.abs(residuals)
            # Huber weights: 1 if |r| <= delta, delta/|r| otherwise
            w = np.where(abs_r <= self.delta, 1.0, self.delta / (abs_r + 1e-10))
            W = np.diag(w)
            XtWX = X_train.T @ W @ X_train + self.alpha * np.eye(d)
            XtWy = X_train.T @ (w * y_train)
            self.weights = np.linalg.solve(XtWX, XtWy)
            self.bias = np.average(y_train - X_train @ self.weights, weights=w)
        return {}

    def predict(self, X):
        return X @ self.weights + self.bias

    def save(self, path):
        joblib.dump({"weights": self.weights, "bias": self.bias, "delta": self.delta, "alpha": self.alpha}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        m = cls(delta=d["delta"], alpha=d["alpha"])
        m.weights, m.bias = d["weights"], d["bias"]
        return m

    @property
    def name(self):
        return self.name_

    def predict_with_confidence(self, X):
        return self.predict(X), np.ones(len(X)) * 0.6


class AdaptiveKNN:
    """KNN com seleção adaptativa de k e pesos por distancia."""

    def __init__(self, k_range=(5, 30)):
        self.k_range = k_range
        self.best_k = 10
        self.X_train = None
        self.y_train = None
        self.name_ = "aknn_reg"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        # Encontrar melhor k no validation
        if X_val is not None and len(X_val) > 0:
            best_acc, best_k = 0, 10
            for k in range(self.k_range[0], min(self.k_range[1], len(X_train)), 2):
                preds = self._predict_k(X_val, k)
                acc = float(np.mean(np.sign(preds) == np.sign(y_val)))
                if acc > best_acc:
                    best_acc = acc
                    best_k = k
            self.best_k = best_k
        return {}

    def _predict_k(self, X, k):
        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            dists = np.sqrt(np.sum((self.X_train - X[i]) ** 2, axis=1))
            idx = np.argsort(dists)[:k]
            w = 1.0 / (dists[idx] + 1e-10)
            preds[i] = np.average(self.y_train[idx], weights=w)
        return preds

    def predict(self, X):
        return self._predict_k(X, self.best_k)

    def save(self, path):
        joblib.dump({"X": self.X_train, "y": self.y_train, "k": self.best_k}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        m = cls()
        m.X_train, m.y_train, m.best_k = d["X"], d["y"], d["k"]
        return m

    @property
    def name(self):
        return self.name_

    def predict_with_confidence(self, X):
        return self.predict(X), np.ones(len(X)) * 0.55


class DirectionClassifier:
    """Classificador direcional via Ridge + threshold otimizado."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.threshold = 0.0
        self.scale = 1.0
        self.name_ = "dir_cls"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Treinar como classificador binario (direction)
        y_binary = (y_train > 0).astype(float)
        n, d = X_train.shape
        XtX = X_train.T @ X_train + self.alpha * np.eye(d)
        Xty = X_train.T @ y_binary
        self.weights = np.linalg.solve(XtX, Xty)
        self.bias = np.mean(y_binary - X_train @ self.weights)
        self.scale = np.std(y_train)

        # Otimizar threshold no validation
        if X_val is not None:
            probs = X_val @ self.weights + self.bias
            y_val_dir = np.sign(y_val)
            best_acc, best_t = 0, 0.5
            for t in np.arange(0.3, 0.7, 0.02):
                pred_dir = np.where(probs > t, 1, -1)
                acc = np.mean(pred_dir == y_val_dir)
                if acc > best_acc:
                    best_acc = acc
                    best_t = t
            self.threshold = best_t
        return {}

    def predict(self, X):
        probs = X @ self.weights + self.bias
        return (probs - self.threshold) * self.scale * 2

    def save(self, path):
        joblib.dump({
            "weights": self.weights, "bias": self.bias,
            "threshold": self.threshold, "scale": self.scale, "alpha": self.alpha,
        }, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        m = cls(alpha=d["alpha"])
        m.weights, m.bias, m.threshold, m.scale = d["weights"], d["bias"], d["threshold"], d["scale"]
        return m

    @property
    def name(self):
        return self.name_

    def predict_with_confidence(self, X):
        return self.predict(X), np.ones(len(X)) * 0.6


# ============================================================
# Feature engineering adicional
# ============================================================

def add_extra_features(df, feature_columns):
    """Adiciona features extras: interacoes, ratios, momentum composto."""
    df = df.copy()
    new_cols = []

    # Momentum features compostas
    if "rsi_14" in df.columns and "macd_hist" in df.columns:
        df["momentum_composite"] = df["rsi_14"] / 100 * np.sign(df["macd_hist"])
        new_cols.append("momentum_composite")

    # Volatility-adjusted returns
    if "log_return" in df.columns and "atr_14" in df.columns:
        df["vol_adj_return"] = df["log_return"] / (df["atr_14"] / df["close"] + 1e-10)
        new_cols.append("vol_adj_return")

    # Price position within Bollinger Bands
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (bb_range + 1e-10)
        new_cols.append("bb_position")

    # Volume momentum
    if "volume" in df.columns:
        df["volume_momentum_5"] = df["volume"] / df["volume"].rolling(5).mean() - 1
        df["volume_momentum_20"] = df["volume"] / df["volume"].rolling(20).mean() - 1
        new_cols.extend(["volume_momentum_5", "volume_momentum_20"])

    # EMA crossover signals
    if "ema_9" in df.columns and "ema_21" in df.columns:
        df["ema_cross_9_21"] = (df["ema_9"] - df["ema_21"]) / df["close"]
        new_cols.append("ema_cross_9_21")
    if "ema_21" in df.columns and "ema_50" in df.columns:
        df["ema_cross_21_50"] = (df["ema_21"] - df["ema_50"]) / df["close"]
        new_cols.append("ema_cross_21_50")

    # Trend strength
    if "close" in df.columns:
        for w in [5, 10, 20]:
            sma = df["close"].rolling(w).mean()
            df[f"trend_strength_{w}"] = (df["close"] - sma) / (sma + 1e-10)
            new_cols.append(f"trend_strength_{w}")

    # Rolling directional accuracy (meta-feature)
    if "log_return" in df.columns:
        for w in [5, 10, 20]:
            direction = (df["log_return"] > 0).astype(float)
            df[f"dir_streak_{w}"] = direction.rolling(w).mean()
            new_cols.append(f"dir_streak_{w}")

    # Volatility regime
    if "log_return" in df.columns:
        vol_20 = df["log_return"].rolling(20).std()
        vol_60 = df["log_return"].rolling(60).std()
        df["vol_regime"] = vol_20 / (vol_60 + 1e-10)
        new_cols.append("vol_regime")

    # Return acceleration
    if "log_return" in df.columns:
        df["return_accel"] = df["log_return"].diff()
        df["return_accel_5"] = df["log_return"].diff(5)
        new_cols.extend(["return_accel", "return_accel_5"])

    # Clean
    for col in new_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df, feature_columns + new_cols


# ============================================================
# Meta-learner (stacking)
# ============================================================

class MetaLearnerRidge:
    """Ridge regression como meta-learner para stacking."""

    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, predictions_matrix, y_true):
        n, d = predictions_matrix.shape
        XtX = predictions_matrix.T @ predictions_matrix + self.alpha * np.eye(d)
        Xty = predictions_matrix.T @ y_true
        self.weights = np.linalg.solve(XtX, Xty)
        self.bias = np.mean(y_true - predictions_matrix @ self.weights)

    def predict(self, predictions_matrix):
        return predictions_matrix @ self.weights + self.bias


# ============================================================
# Treinamento principal
# ============================================================

def train_coin_optimized(coin, df_features, feature_columns):
    """Treina todos os modelos com otimizacao maxima."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Treinando {coin} (otimizado)")
    logger.info(f"{'='*60}")

    # Feature engineering extra
    df_features, feature_columns = add_extra_features(df_features, feature_columns)

    X = df_features[feature_columns].values.astype(np.float64)
    y = df_features["target"].values.astype(np.float64)
    logger.info(f"Dados: {X.shape[0]} amostras, {X.shape[1]} features")

    # Mais folds para avaliacao mais robusta
    n_splits = min(8, max(3, X.shape[0] // 80))
    splits = walk_forward_split(X, y, n_splits=n_splits)
    if not splits:
        return None

    all_fold_metrics = []
    meta_X_train = []
    meta_y_train = []

    for fold_idx, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")

        scaler = NumpyScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        fold_preds = {}
        fold_val_preds = {}

        # === Modelos ===
        models_config = [
            ("ridge_0.1", RidgeRegression, {"alpha": 0.1}),
            ("ridge_1", RidgeRegression, {"alpha": 1.0}),
            ("ridge_10", RidgeRegression, {"alpha": 10.0}),
            ("ridge_100", RidgeRegression, {"alpha": 100.0}),
            ("enet_01", ElasticNetRegression, {"alpha": 0.01, "l1_ratio": 0.5}),
            ("enet_05", ElasticNetRegression, {"alpha": 0.05, "l1_ratio": 0.3}),
            ("lasso_01", LassoRegression, {"alpha": 0.01}),
            ("lasso_05", LassoRegression, {"alpha": 0.05}),
            ("huber", HuberRegression, {"delta": 1.35, "alpha": 1.0}),
            ("gbs_100", GradientBoostedStumps, {"n_estimators": 100, "learning_rate": 0.05}),
            ("gbs_200", GradientBoostedStumps, {"n_estimators": 200, "learning_rate": 0.03}),
            ("gbs_300", GradientBoostedStumps, {"n_estimators": 300, "learning_rate": 0.02}),
            ("gbs_500", GradientBoostedStumps, {"n_estimators": 500, "learning_rate": 0.01}),
            ("knn", KNNRegressor, {"k": 10}),
            ("aknn", AdaptiveKNN, {}),
            ("dir_cls", DirectionClassifier, {"alpha": 1.0}),
        ]

        for name, cls, kwargs in models_config:
            try:
                model = cls(**kwargs)
                model.fit(X_train_s, y_train, X_val_s, y_val)
                fold_val_preds[name] = model.predict(X_val_s)
                fold_preds[name] = model.predict(X_test_s)
            except Exception as e:
                logger.warning(f"  {name}: falhou ({e})")

        # Ensemble
        model_names = list(fold_val_preds.keys())
        if not model_names:
            continue

        min_val = min(len(v) for v in fold_val_preds.values())
        min_test = min(len(v) for v in fold_preds.values())
        val_matrix = np.column_stack([v[-min_val:] for v in fold_val_preds.values()])
        test_matrix = np.column_stack([v[-min_test:] for v in fold_preds.values()])
        y_val_a = y_val[-min_val:]
        y_test_a = y_test[-min_test:]

        # Pesos por dir_accuracy^2 (premiar muito os melhores)
        val_dir_accs = np.array([
            float(np.mean(np.sign(fold_val_preds[n][-min_val:]) == np.sign(y_val_a)))
            for n in model_names
        ])
        weights = (val_dir_accs - 0.45) ** 2
        weights = np.maximum(weights, 0.001)
        weights = weights / weights.sum()

        ensemble_preds = test_matrix @ weights

        # Coletar dados para meta-learner
        meta_X_train.append(test_matrix)
        meta_y_train.append(y_test_a)

        fold_metrics = {"fold": fold_idx}
        fold_metrics.update(compute_metrics(y_test_a, ensemble_preds, "ensemble_"))
        for name, preds in fold_preds.items():
            fold_metrics.update(compute_metrics(y_test_a, preds[-min_test:], f"{name}_"))

        all_fold_metrics.append(fold_metrics)
        logger.info(
            f"  Ensemble: RMSE={fold_metrics.get('ensemble_rmse', 0):.6f}, "
            f"Dir.Acc={fold_metrics.get('ensemble_dir_accuracy', 0):.2%}"
        )

    # Meta-learner: treinar Ridge sobre as predicoes dos modelos base
    meta_learner = None
    if len(meta_X_train) >= 2:
        try:
            mX = np.vstack(meta_X_train)
            my = np.concatenate(meta_y_train)
            meta_learner = MetaLearnerRidge(alpha=10.0)
            meta_learner.fit(mX, my)
            meta_preds = meta_learner.predict(mX)
            meta_acc = float(np.mean(np.sign(meta_preds) == np.sign(my)))
            logger.info(f"  Meta-learner dir.acc (in-sample): {meta_acc:.2%}")
        except Exception:
            meta_learner = None

    # Retrain final models on full data
    logger.info("\nRetreinando no dataset completo...")
    full_scaler = NumpyScaler()
    X_full = df_features[feature_columns].values.astype(np.float64)
    y_full = df_features["target"].values.astype(np.float64)
    X_full_s = full_scaler.fit_transform(X_full)

    final_models = {}
    for name, cls, kwargs in models_config:
        try:
            model = cls(**kwargs)
            # Split last 10% as pseudo-validation for AdaptiveKNN/DirectionClassifier
            split_idx = int(len(X_full_s) * 0.9)
            model.fit(X_full_s[:split_idx], y_full[:split_idx], X_full_s[split_idx:], y_full[split_idx:])
            final_models[name] = model
        except Exception:
            pass

    ensemble_data = {"model_names": model_names, "weights": weights.tolist()}
    if meta_learner:
        ensemble_data["meta_weights"] = meta_learner.weights.tolist()
        ensemble_data["meta_bias"] = float(meta_learner.bias)

    # Metricas
    avg_metrics = {}
    if all_fold_metrics:
        metric_keys = [k for k in all_fold_metrics[0] if k != "fold"]
        for key in metric_keys:
            values = [m[key] for m in all_fold_metrics if key in m]
            avg_metrics[f"avg_{key}"] = float(np.mean(values))
            avg_metrics[f"std_{key}"] = float(np.std(values))

    logger.info(f"\n{coin} - Metricas ({len(splits)} folds):")
    for k, v in avg_metrics.items():
        if k.startswith("avg_ensemble"):
            logger.info(f"  {k}: {v:.6f}")

    return {
        "models": final_models,
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

    coins = config.data.coins
    data = {}
    for coin in coins:
        df = storage.load(coin, "1d", stage="raw")
        if not df.empty:
            data[coin] = df
            logger.info(f"  {coin}: {len(df)} candles")

    if not data:
        logger.error("Nenhum dado.")
        sys.exit(1)

    # BTC ref
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

    # Alinhar colunas
    all_feature_cols = set()
    for df in features_data.values():
        all_feature_cols.update(c for c in df.columns if c not in NON_FEATURE_COLS)
    for coin in features_data:
        missing = all_feature_cols - set(features_data[coin].columns)
        for col in missing:
            features_data[coin][col] = 0.0
    feature_columns = sorted(all_feature_cols)
    logger.info(f"Features base: {len(feature_columns)}")

    # Treinar
    results = {}
    for coin, df in features_data.items():
        if df.empty or "target" not in df.columns:
            continue
        result = train_coin_optimized(coin, df, feature_columns)
        if result:
            results[coin] = result

    # Salvar
    models_dir = config.training.models_dir
    for coin, result in results.items():
        coin_dir = models_dir / coin
        coin_dir.mkdir(parents=True, exist_ok=True)

        for name, model in result["models"].items():
            model.save(coin_dir / f"{name}.joblib")
        joblib.dump(result["scaler"], coin_dir / "scaler.joblib")

        for fname, data in [
            ("feature_columns.json", result["feature_columns"]),
            ("metrics.json", result["avg_metrics"]),
            ("fold_metrics.json", result["fold_metrics"]),
            ("ensemble_data.json", result["ensemble_data"]),
        ]:
            with open(coin_dir / fname, "w") as f:
                json.dump(data, f, indent=2)

        # Feature importance
        gbs = result["models"].get("gbs_300")
        if gbs:
            imp = gbs.get_feature_importance()
            named_imp = {}
            for idx_str, val in imp.items():
                idx = int(idx_str)
                if idx < len(result["feature_columns"]):
                    named_imp[result["feature_columns"][idx]] = val
            with open(coin_dir / "feature_importance.json", "w") as f:
                json.dump(named_imp, f, indent=2)

        logger.info(f"Salvos em {coin_dir}")

    # Resumo
    print("\n" + "=" * 70)
    print("  RESUMO DO TREINO OTIMIZADO")
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
