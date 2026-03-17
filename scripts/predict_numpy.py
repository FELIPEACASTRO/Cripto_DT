#!/usr/bin/env python3
"""Gera previsoes usando modelos numpy-only treinados localmente."""

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

import numpy as np
import pandas as pd
import joblib

from config.settings import config
from src.data.storage import DataStorage


# Import numpy-only models
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_numpy import (
    RidgeRegression, ElasticNetRegression, GradientBoostedStumps,
    KNNRegressor, NumpyScaler, ScalerWrapper, generate_features,
)

# Registrar classes no __main__ para que joblib.load funcione
import __main__
__main__.ScalerWrapper = ScalerWrapper
__main__.NumpyScaler = NumpyScaler
__main__.RidgeRegression = RidgeRegression
__main__.ElasticNetRegression = ElasticNetRegression
__main__.GradientBoostedStumps = GradientBoostedStumps
__main__.KNNRegressor = KNNRegressor


MODEL_CLASSES = {
    "ridge_a0.1": RidgeRegression,
    "ridge_a1.0": RidgeRegression,
    "ridge_a10.0": RidgeRegression,
    "enet_reg": ElasticNetRegression,
    "gbs_100": GradientBoostedStumps,
    "gbs_200": GradientBoostedStumps,
    "gbs_300": GradientBoostedStumps,
    "knn_reg": KNNRegressor,
}


def load_models(coin):
    """Load trained models for a coin."""
    coin_dir = config.training.models_dir / coin
    if not coin_dir.exists():
        return None

    models = {}
    for name, cls in MODEL_CLASSES.items():
        path = coin_dir / f"{name}.joblib"
        if path.exists():
            try:
                models[name] = cls.load(path)
            except Exception:
                pass

    scaler_path = coin_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    fc_path = coin_dir / "feature_columns.json"
    feature_columns = json.load(open(fc_path)) if fc_path.exists() else []

    ens_path = coin_dir / "ensemble_data.json"
    ensemble_data = json.load(open(ens_path)) if ens_path.exists() else None

    return {
        "models": models,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "ensemble_data": ensemble_data,
    }


def predict_coin(coin, storage, btc_df=None):
    """Generate prediction for one coin."""
    loaded = load_models(coin)
    if loaded is None or not loaded["models"]:
        logger.warning(f"Modelos nao encontrados para {coin}")
        return None

    models = loaded["models"]
    scaler = loaded["scaler"]
    feature_columns = loaded["feature_columns"]
    ensemble_data = loaded["ensemble_data"]

    df = storage.load(coin, "1d", stage="raw")
    if df.empty:
        return None

    from src.data.preprocessor import DataPreprocessor
    btc_ref = None
    if coin != "BTC" and btc_df is not None:
        btc_ref = DataPreprocessor().prepare(btc_df, horizon=1, target_type="log_return")

    df_features = generate_features(df, coin, btc_ref)
    if df_features.empty:
        return None

    # Align features
    missing = [c for c in feature_columns if c not in df_features.columns]
    for col in missing:
        df_features[col] = 0.0

    X = df_features[feature_columns].fillna(0).values.astype(np.float64)
    X_scaled = scaler.scaler.transform(X)

    # Individual predictions
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

    # Confidence from model agreement
    if base_preds:
        signs = [np.sign(v) for v in base_preds.values()]
        agreement = abs(sum(signs)) / len(signs)
        confidence = 0.5 + agreement * 0.3
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

    # Save
    pred_dir = config.data.predictions_dir
    pred_dir.mkdir(parents=True, exist_ok=True)
    filename = f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(pred_dir / filename, index=False)

    # Display
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
