#!/usr/bin/env python3
"""Treinamento local otimizado - apenas modelos que NAO dependem de PyTorch.

Treina XGBoost, LightGBM, Random Forest, SVM + Ensemble para todas as moedas.
Para modelos deep learning (LSTM, GRU, CNN-LSTM, etc.), use o notebook Kaggle.
"""

import json
import logging
import sys
import os
from pathlib import Path

# Impedir import de torch (evita hang no Windows)
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
from sklearn.preprocessing import StandardScaler


def safe_import(module_path, class_name):
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        logger.debug(f"{class_name} nao disponivel: {e}")
        return None


class ScalerWrapper:
    """Wrapper para StandardScaler compativel com pickle e predictor."""
    def __init__(self, scaler):
        self.scaler = scaler


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


def train_coin_local(coin, df_features, feature_columns, config):
    """Treina modelos sklearn para uma moeda."""
    from src.training.walk_forward import WalkForwardValidator

    logger.info(f"\n{'='*60}")
    logger.info(f"Treinando modelos para {coin}")
    logger.info(f"{'='*60}")

    X = df_features[feature_columns].values.astype(np.float32)
    y = df_features["target"].values.astype(np.float32)
    logger.info(f"Dados: {X.shape[0]} amostras, {X.shape[1]} features")

    wf = WalkForwardValidator(
        n_splits=config.training.walk_forward_splits,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio,
    )
    splits = wf.split_data(X, y)

    all_fold_metrics = []
    best_models = {}
    best_scaler = None
    conformal_residuals = []

    for fold_idx, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Scaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        fold_preds = {}
        fold_val_preds = {}
        fold_metrics = {"fold": fold_idx}

        # --- XGBoost Regressor ---
        from src.models.xgboost_model import XGBoostRegressor, XGBoostClassifier
        logger.info("  Treinando XGBoost Regressor...")
        xgb_reg = XGBoostRegressor(config.xgboost)
        xgb_reg.fit(X_train_s, y_train, X_val_s, y_val)
        fold_val_preds["xgb_reg"] = xgb_reg.predict(X_val_s)
        fold_preds["xgb_reg"] = xgb_reg.predict(X_test_s)

        # --- XGBoost Classifier ---
        xgb_cls = None
        if config.xgboost.use_direction:
            logger.info("  Treinando XGBoost Classifier...")
            xgb_cls = XGBoostClassifier(config.xgboost)
            xgb_cls.fit(X_train_s, y_train, X_val_s, y_val)
            scale = np.std(y_train) if np.std(y_train) > 0 else 1.0
            fold_val_preds["xgb_cls"] = (xgb_cls.predict(X_val_s) - 0.5) * 2 * scale
            fold_preds["xgb_cls"] = (xgb_cls.predict(X_test_s) - 0.5) * 2 * scale

        # --- LightGBM ---
        LGBMReg = safe_import("src.models.lightgbm_model", "LightGBMRegressor")
        lgbm_reg = None
        if LGBMReg:
            try:
                import dataclasses
                lgbm_params = {
                    k: v for k, v in dataclasses.asdict(config.lightgbm).items()
                    if k != "use_direction"
                }
                logger.info("  Treinando LightGBM Reg...")
                lgbm_reg = LGBMReg(**lgbm_params)
                lgbm_reg.fit(X_train_s, y_train, X_val_s, y_val)
                fold_val_preds["lgbm_reg"] = lgbm_reg.predict(X_val_s)
                fold_preds["lgbm_reg"] = lgbm_reg.predict(X_test_s)
            except Exception as e:
                logger.warning(f"  LightGBM falhou: {e}")

        # --- Random Forest ---
        RFReg = safe_import("src.models.random_forest", "RandomForestRegressorModel")
        rf_reg = None
        if RFReg:
            try:
                logger.info("  Treinando Random Forest...")
                rf_reg = RFReg()
                rf_reg.fit(X_train_s, y_train, X_val_s, y_val)
                fold_val_preds["rf_reg"] = rf_reg.predict(X_val_s)
                fold_preds["rf_reg"] = rf_reg.predict(X_test_s)
            except Exception as e:
                logger.warning(f"  Random Forest falhou: {e}")

        # --- SVM ---
        SVMReg = safe_import("src.models.svm_model", "SVMRegressor")
        svm_reg = None
        if SVMReg:
            try:
                logger.info("  Treinando SVM Reg...")
                svm_reg = SVMReg()
                svm_reg.fit(X_train_s, y_train, X_val_s, y_val)
                fold_val_preds["svm_reg"] = svm_reg.predict(X_val_s)
                fold_preds["svm_reg"] = svm_reg.predict(X_test_s)
            except Exception as e:
                logger.warning(f"  SVM falhou: {e}")

        # --- Ensemble (media ponderada por performance) ---
        logger.info("  Construindo Ensemble...")
        model_names = list(fold_val_preds.keys())
        min_val_len = min(len(v) for v in fold_val_preds.values())
        min_test_len = min(len(v) for v in fold_preds.values())

        val_matrix = np.column_stack([v[-min_val_len:] for v in fold_val_preds.values()])
        test_matrix = np.column_stack([v[-min_test_len:] for v in fold_preds.values()])
        y_val_aligned = y_val[-min_val_len:]
        y_test_aligned = y_test[-min_test_len:]

        # Pesos baseados em dir_accuracy no validation
        val_dir_accs = np.array([
            float(np.mean(np.sign(fold_val_preds[name][-min_val_len:]) == np.sign(y_val_aligned)))
            for name in model_names
        ])
        # Normalizar pesos
        weights = val_dir_accs - 0.5  # Bonus por acima de random
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()

        ensemble_preds = test_matrix @ weights
        conformal_residuals.extend((y_test_aligned - ensemble_preds).tolist())

        # Metricas
        fold_metrics.update(compute_metrics(y_test_aligned, ensemble_preds, "ensemble_"))
        for name, preds in fold_preds.items():
            fold_metrics.update(compute_metrics(y_test_aligned, preds[-min_test_len:], f"{name}_"))

        all_fold_metrics.append(fold_metrics)
        logger.info(
            f"  Fold {fold_idx + 1} ensemble: "
            f"RMSE={fold_metrics.get('ensemble_rmse', 0):.6f}, "
            f"Dir.Acc={fold_metrics.get('ensemble_dir_accuracy', 0):.4f}"
        )

    # Ultimo fold = modelos finais
    best_models = {"xgb_reg": xgb_reg}
    if xgb_cls:
        best_models["xgb_cls"] = xgb_cls
    if lgbm_reg:
        best_models["lgbm_reg"] = lgbm_reg
    if rf_reg:
        best_models["rf_reg"] = rf_reg
    if svm_reg:
        best_models["svm_reg"] = svm_reg

    # Ensemble simples (salvar pesos e nomes)
    ensemble_data = {"model_names": model_names, "weights": weights.tolist()}

    # Metricas medias
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
        "scaler": ScalerWrapper(scaler),
        "fold_metrics": all_fold_metrics,
        "avg_metrics": avg_metrics,
        "feature_columns": feature_columns,
        "ensemble_data": ensemble_data,
    }


def main():
    from config.settings import config

    # Desabilitar modelos torch-dependentes
    config.training.use_mamba = False
    config.training.use_helformer = False
    config.training.use_cnn_lstm = False
    config.training.use_tcn = False
    config.training.use_mdn = False
    config.training.use_bnn = False
    config.training.use_evidential = False
    config.training.use_dual_prediction = False
    config.training.use_emgnn = False
    config.training.use_chronos = False
    config.training.use_ttm = False
    config.training.use_moirai = False
    config.training.use_moe_ensemble = False
    config.training.use_evolutionary_ensemble = False
    config.training.use_market_gan = False
    config.features.feature_selection_method = "mutual_info"

    from src.data.storage import DataStorage

    storage = DataStorage(config)

    # Carregar dados
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

    # Feature engineering (sem torch)
    from src.data.preprocessor import DataPreprocessor
    from src.features.technical import TechnicalFeatures
    from src.features.lag_features import LagFeatures
    from src.features.market import MarketFeatures

    preprocessor = DataPreprocessor()
    technical = TechnicalFeatures(config.features)
    lag = LagFeatures(config.features)
    market = MarketFeatures(config.features)

    NON_FEATURE_COLS = {
        "timestamp", "open", "high", "low", "close", "volume",
        "log_return", "pct_return", "direction", "target",
        "close_denoised", "high_denoised", "low_denoised", "open_denoised",
    }

    # Preparar BTC para correlacao
    btc_df = None
    if "BTC" in data:
        btc_df = preprocessor.prepare(data["BTC"], horizon=1, target_type="log_return")

    features_data = {}
    for coin, df in data.items():
        logger.info(f"Features para {coin}...")
        df = preprocessor.prepare(df, horizon=1, target_type="log_return")

        # Wavelet
        try:
            from src.features.wavelet import WaveletDenoiser
            wavelet = WaveletDenoiser()
            df = wavelet.transform(df)
        except Exception:
            pass

        df = technical.transform(df)
        df = lag.transform(df)
        btc_ref = None if coin == "BTC" else btc_df
        df = market.transform(df, btc_df=btc_ref)

        # Volatility
        try:
            from src.features.volatility import VolatilityFeatures
            df = VolatilityFeatures().transform(df)
        except Exception:
            pass

        # HAR
        try:
            from src.features.har_volatility import HARVolatility
            df = HARVolatility().transform(df)
        except Exception:
            pass

        # Bubble
        try:
            from src.features.bubble import BubbleDetector
            df = BubbleDetector().transform(df)
        except Exception:
            pass

        # Anomaly
        try:
            from src.features.anomaly import AnomalyDetector
            df = AnomalyDetector().transform(df)
        except Exception:
            pass

        # Chart Patterns
        try:
            from src.features.chart_patterns import ChartPatternDetector
            df = ChartPatternDetector().transform(df)
        except Exception:
            pass

        # Regional Intelligence
        try:
            from src.features.regional_intelligence import RegionalIntelligence
            df = RegionalIntelligence().transform(df)
        except Exception:
            pass

        # FinBERT proxy (sem torch)
        try:
            from src.features.finbert_sentiment import NLPSentimentFeatures
            df = NLPSentimentFeatures().transform(df)
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

        features_data[coin] = df
        logger.info(f"  {coin}: {len(df)} amostras, {len(feature_cols)} features")

    # Alinhar colunas
    all_feature_cols = set()
    for df in features_data.values():
        all_feature_cols.update(c for c in df.columns if c not in NON_FEATURE_COLS)
    for coin in features_data:
        missing = all_feature_cols - set(features_data[coin].columns)
        for col in missing:
            features_data[coin][col] = 0.0
    feature_columns = sorted(all_feature_cols)
    logger.info(f"Features finais: {len(feature_columns)} (alinhadas)")

    # Treinar
    results = {}
    for coin, df in features_data.items():
        if df.empty or "target" not in df.columns:
            continue
        results[coin] = train_coin_local(coin, df, feature_columns, config)

    # Salvar
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

        # Feature importance
        xgb_model = result["models"].get("xgb_reg")
        if xgb_model and hasattr(xgb_model, "get_feature_importance"):
            importance = xgb_model.get_feature_importance()
            if importance:
                with open(coin_dir / "feature_importance.json", "w") as f:
                    json.dump(importance, f, indent=2)

        logger.info(f"Salvos em {coin_dir}")

    # Resumo
    print("\n" + "=" * 70)
    print("  RESUMO DO TREINO")
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
