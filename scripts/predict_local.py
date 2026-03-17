#!/usr/bin/env python3
"""Gera previsoes usando modelos treinados localmente (sem PyTorch)."""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CRIPTO_DT_NO_TORCH"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import joblib
import numpy as np
import pandas as pd

from config.settings import config
from src.data.storage import DataStorage


def load_models(coin):
    """Carrega modelos treinados para uma moeda."""
    coin_dir = config.training.models_dir / coin
    if not coin_dir.exists():
        return None

    models = {}

    # XGBoost
    from src.models.xgboost_model import XGBoostRegressor, XGBoostClassifier
    for name, cls in [("xgb_reg", XGBoostRegressor), ("xgb_cls", XGBoostClassifier)]:
        path = coin_dir / f"{name}.joblib"
        if path.exists():
            try:
                models[name] = cls.load(path)
            except Exception:
                pass

    # LightGBM, RF, SVM
    optional = [
        ("lgbm_reg", "src.models.lightgbm_model", "LightGBMRegressor"),
        ("rf_reg", "src.models.random_forest", "RandomForestRegressorModel"),
        ("svm_reg", "src.models.svm_model", "SVMRegressor"),
    ]
    for name, mod_path, cls_name in optional:
        path = coin_dir / f"{name}.joblib"
        if path.exists():
            try:
                mod = __import__(mod_path, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                models[name] = cls.load(path)
            except Exception:
                pass

    # Scaler
    scaler_path = coin_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    # Feature columns
    fc_path = coin_dir / "feature_columns.json"
    feature_columns = json.load(open(fc_path)) if fc_path.exists() else []

    # Ensemble data
    ens_path = coin_dir / "ensemble_data.json"
    ensemble_data = json.load(open(ens_path)) if ens_path.exists() else None

    return {
        "models": models,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "ensemble_data": ensemble_data,
    }


def generate_features(df, coin, btc_df=None):
    """Gera features para previsao (sem torch)."""
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

    df = TechnicalFeatures(config.features).transform(df)
    df = LagFeatures(config.features).transform(df)
    btc_ref = None if coin == "BTC" else btc_df
    df = MarketFeatures(config.features).transform(df, btc_df=btc_ref)

    for module_path, cls_name in [
        ("src.features.volatility", "VolatilityFeatures"),
        ("src.features.har_volatility", "HARVolatility"),
        ("src.features.bubble", "BubbleDetector"),
        ("src.features.anomaly", "AnomalyDetector"),
        ("src.features.chart_patterns", "ChartPatternDetector"),
        ("src.features.regional_intelligence", "RegionalIntelligence"),
        ("src.features.finbert_sentiment", "NLPSentimentFeatures"),
    ]:
        try:
            mod = __import__(module_path, fromlist=[cls_name])
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

    return df


def predict_coin(coin, storage, btc_df=None):
    """Gera previsao para uma moeda."""
    loaded = load_models(coin)
    if loaded is None or not loaded["models"]:
        logger.warning(f"Modelos nao encontrados para {coin}")
        return None

    models = loaded["models"]
    scaler = loaded["scaler"]
    feature_columns = loaded["feature_columns"]
    ensemble_data = loaded["ensemble_data"]

    # Dados
    df = storage.load(coin, "1d", stage="raw")
    if df.empty:
        return None

    btc_ref = None
    if coin != "BTC" and btc_df is not None:
        from src.data.preprocessor import DataPreprocessor
        btc_ref = DataPreprocessor().prepare(btc_df, horizon=1, target_type="log_return")

    df_features = generate_features(df, coin, btc_ref)
    if df_features.empty:
        return None

    # Alinhar features
    missing = [c for c in feature_columns if c not in df_features.columns]
    for col in missing:
        df_features[col] = 0.0

    X = df_features[feature_columns].fillna(0).values.astype(np.float32)
    X_scaled = scaler.scaler.transform(X)

    # Previsoes individuais
    base_preds = {}
    for name, model in models.items():
        try:
            pred = model.predict(X_scaled)
            base_preds[name] = float(pred[-1])
        except Exception as e:
            logger.warning(f"  {name} falhou: {e}")

    # Ensemble
    if ensemble_data and base_preds:
        model_names = ensemble_data["model_names"]
        weights = np.array(ensemble_data["weights"])
        vals = [base_preds.get(name, 0.0) for name in model_names]
        final_pred = float(np.dot(vals, weights))
    else:
        final_pred = float(np.mean(list(base_preds.values()))) if base_preds else 0.0

    current_price = float(df_features["close"].iloc[-1])
    predicted_price = current_price * np.exp(final_pred)
    direction = "ALTA" if final_pred > 0 else "BAIXA"

    # Confianca baseada em concordancia dos modelos
    if base_preds:
        signs = [np.sign(v) for v in base_preds.values()]
        agreement = abs(sum(signs)) / len(signs)
        confidence = 0.5 + agreement * 0.3  # 50-80%
    else:
        confidence = 0.5

    logger.info(
        f"{coin}: {direction} | Retorno: {final_pred:+.4f} | "
        f"Preco: ${predicted_price:,.2f} | Confianca: {confidence:.1%}"
    )

    return {
        "coin": coin,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_price": current_price,
        "predicted_return": final_pred,
        "predicted_price": predicted_price,
        "direction": direction,
        "confidence": confidence,
        "model_predictions": base_preds,
        "n_models": len(base_preds),
    }


def main():
    storage = DataStorage(config)

    # BTC para correlacao
    btc_df = storage.load("BTC", "1d", stage="raw")

    predictions = []
    for coin in config.data.coins:
        try:
            pred = predict_coin(coin, storage, btc_df)
            if pred:
                predictions.append(pred)
        except Exception as e:
            logger.error(f"Erro ao prever {coin}: {e}")

    if not predictions:
        logger.error("Nenhuma previsao gerada.")
        sys.exit(1)

    df = pd.DataFrame(predictions)

    # Salvar
    pred_dir = config.data.predictions_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    filename = f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(pred_dir / filename, index=False)

    # Exibir
    print("\n" + "=" * 80)
    print("  PREVISOES DE CRIPTOMOEDAS")
    print("=" * 80)
    for _, row in df.iterrows():
        arrow = "^" if row["direction"] == "ALTA" else "v"
        print(
            f"  {row['coin']:>6s} {arrow} {row['direction']:>5s} | "
            f"Atual: ${row['current_price']:>10,.2f} | "
            f"Previsto: ${row['predicted_price']:>10,.2f} | "
            f"Retorno: {row['predicted_return']:>+.4f} | "
            f"Confianca: {row['confidence']:.1%} | "
            f"Modelos: {row['n_models']}"
        )
    print("=" * 80)
    print(f"  Previsoes salvas em: {pred_dir / filename}")
    print("=" * 80)


if __name__ == "__main__":
    main()
