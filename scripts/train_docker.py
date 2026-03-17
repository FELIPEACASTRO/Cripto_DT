#!/usr/bin/env python3
"""Training script designed to run INSIDE Docker or Kaggle where sklearn,
xgboost, and lightgbm are available.

Models:
  - XGBClassifier  (direction)
  - XGBRegressor   (magnitude)
  - LGBMClassifier (direction diversity)
  - LGBMRegressor  (magnitude diversity)
  - Stacking meta-learner (XGBoost on out-of-fold predictions)

Training: walk-forward validation with 8 folds, expanding window.
Features: All pipeline features + visual patterns + microstructure + entropy
Feature Selection: Correlation filter + importance-based selection

Usage:
  docker-compose run trainer
  OR
  python scripts/train_docker.py  (if sklearn/xgboost/lightgbm are available)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["CRIPTO_DT_NO_TORCH"] = "1"

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
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from config.settings import config
from src.data.storage import DataStorage
from src.data.preprocessor import DataPreprocessor


# ============================================================
# Constants
# ============================================================

NON_FEATURE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume",
    "log_return", "pct_return", "direction", "target",
    "close_denoised", "high_denoised", "low_denoised", "open_denoised",
}

N_FOLDS = 8
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test ratio = 0.15 (implicit)


# ============================================================
# Feature Engineering
# ============================================================

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gera features usando o pipeline completo do projeto."""
    try:
        from src.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(config)
        return pipeline.transform(df)
    except Exception as e:
        logger.warning(f"Pipeline completo falhou: {e}. Usando features basicas.")
        return _basic_features(df)


def generate_features_all(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Gera features para todas as moedas (com cross-crypto)."""
    try:
        from src.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(config)
        return pipeline.transform_all(data)
    except Exception as e:
        logger.warning(f"Pipeline transform_all falhou: {e}. Processando individualmente.")
        results = {}
        for coin, df in data.items():
            try:
                results[coin] = generate_features(df)
            except Exception as ex:
                logger.error(f"Falha ao gerar features para {coin}: {ex}")
        return results


def _basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features basicas se o pipeline completo falhar."""
    preprocessor = DataPreprocessor()
    df = preprocessor.prepare(df, horizon=1, target_type="log_return")

    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df["log_return"].shift(lag)
    for w in [3, 5, 10]:
        df[f"return_cum_{w}"] = df["log_return"].rolling(w).sum()
    for w in [5, 10, 20]:
        df[f"return_mean_{w}"] = df["log_return"].rolling(w).mean()
        df[f"return_std_{w}"] = df["log_return"].rolling(w).std()
        df[f"volume_ratio_{w}"] = volume / volume.rolling(w).mean()
        df[f"momentum_{w}"] = close / close.shift(w) - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper_dist"] = (close - (bb_ma + 2 * bb_std)) / close
    df["bb_lower_dist"] = (close - (bb_ma - 2 * bb_std)) / close

    df = df.ffill().bfill().fillna(0)
    if "target" in df.columns:
        df = df.dropna(subset=["target"]).reset_index(drop=True)
    return df


# ============================================================
# Feature Selection
# ============================================================

def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    max_features: int = 60,
    corr_threshold: float = 0.95,
) -> tuple[list[int], list[str]]:
    """Seleciona features removendo redundância e baixa variância."""
    n_features = X_train.shape[1]

    # 1. Variance filter
    variances = np.var(X_train, axis=0)
    var_mask = variances > 0.001
    logger.info(f"  Variance filter: {var_mask.sum()}/{n_features} features mantidas")

    # 2. Correlation filter
    valid_indices = np.where(var_mask)[0]
    X_valid = X_train[:, valid_indices]

    if len(valid_indices) > max_features:
        corr_matrix = np.corrcoef(X_valid.T)
        to_remove = set()
        for i in range(len(valid_indices)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(valid_indices)):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > corr_threshold:
                    if variances[valid_indices[i]] >= variances[valid_indices[j]]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

        corr_mask = [i for i in range(len(valid_indices)) if i not in to_remove]
        valid_indices = valid_indices[corr_mask]
        logger.info(f"  Correlation filter: {len(valid_indices)} features mantidas")

    # 3. Importance-based selection via quick XGBoost
    if len(valid_indices) > max_features:
        direction = (y_train > 0).astype(int)
        quick_xgb = XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42,
        )
        quick_xgb.fit(X_train[:, valid_indices], direction)
        importances = quick_xgb.feature_importances_
        top_idx = np.argsort(importances)[::-1][:max_features]
        valid_indices = valid_indices[top_idx]
        logger.info(f"  Importance filter: top {max_features} features selecionadas")

    selected_names = [feature_names[i] for i in valid_indices]
    return valid_indices.tolist(), selected_names


# ============================================================
# Walk-Forward Training
# ============================================================

def walk_forward_train(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    coin: str,
    models_dir: Path,
):
    """Treina modelos com walk-forward validation."""
    n = len(y)
    direction = (y > 0).astype(int)

    fold_metrics = []
    oof_predictions = np.full((n, 4), np.nan)
    oof_probas = np.full((n, 2), np.nan)

    best_val_acc = 0
    best_models = {}

    # Feature selection
    train_end = int(n * 0.85)
    selected_idx, selected_names = select_features(
        X[:train_end], y[:train_end], feature_names, max_features=60
    )
    X_sel = X[:, selected_idx]

    logger.info(f"  Features selecionadas: {len(selected_names)} de {len(feature_names)}")

    for fold in range(N_FOLDS):
        fold_size = n // (N_FOLDS + 2)
        train_end = fold_size * (fold + 2)
        val_end = train_end + int(fold_size * VAL_RATIO / TRAIN_RATIO)
        test_end = min(val_end + int(fold_size * 0.15 / TRAIN_RATIO), n)

        if test_end > n or train_end >= n:
            continue

        X_train = X_sel[:train_end]
        y_train = y[:train_end]
        d_train = direction[:train_end]

        X_val = X_sel[train_end:val_end]
        y_val = y[train_end:val_end]
        d_val = direction[train_end:val_end]

        X_test = X_sel[val_end:test_end]
        y_test = y[val_end:test_end]
        d_test = direction[val_end:test_end]

        if len(X_val) < 5 or len(X_test) < 5:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        models = {}

        # 1. XGBClassifier
        try:
            xgb_cls = XGBClassifier(
                n_estimators=500, max_depth=7, learning_rate=0.01,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="logloss", verbosity=0, random_state=42,
                early_stopping_rounds=50,
            )
            xgb_cls.fit(X_train_s, d_train, eval_set=[(X_val_s, d_val)], verbose=False)
            models["xgb_cls"] = xgb_cls
        except Exception as e:
            logger.warning(f"  XGBClassifier fold {fold}: {e}")

        # 2. XGBRegressor
        try:
            xgb_reg = XGBRegressor(
                n_estimators=500, max_depth=7, learning_rate=0.01,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="rmse", verbosity=0, random_state=42,
                early_stopping_rounds=50,
            )
            xgb_reg.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            models["xgb_reg"] = xgb_reg
        except Exception as e:
            logger.warning(f"  XGBRegressor fold {fold}: {e}")

        # 3. LGBMClassifier
        try:
            import lightgbm
            lgbm_cls = LGBMClassifier(
                n_estimators=500, max_depth=7, learning_rate=0.01,
                num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                verbose=-1, random_state=42,
            )
            lgbm_cls.fit(
                X_train_s, d_train,
                eval_set=[(X_val_s, d_val)],
                callbacks=[lightgbm.early_stopping(50, verbose=False), lightgbm.log_evaluation(0)],
            )
            models["lgbm_cls"] = lgbm_cls
        except Exception as e:
            logger.warning(f"  LGBMClassifier fold {fold}: {e}")

        # 4. LGBMRegressor
        try:
            import lightgbm
            lgbm_reg = LGBMRegressor(
                n_estimators=500, max_depth=7, learning_rate=0.01,
                num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                verbose=-1, random_state=42,
            )
            lgbm_reg.fit(
                X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                callbacks=[lightgbm.early_stopping(50, verbose=False), lightgbm.log_evaluation(0)],
            )
            models["lgbm_reg"] = lgbm_reg
        except Exception as e:
            logger.warning(f"  LGBMRegressor fold {fold}: {e}")

        # Evaluate
        test_preds = {}
        test_probas = {}

        for name, model in models.items():
            if "cls" in name:
                pred_dir = model.predict(X_test_s)
                pred_proba = model.predict_proba(X_test_s)[:, 1]
                test_preds[name] = pred_dir
                test_probas[name] = pred_proba
            else:
                pred_val = model.predict(X_test_s)
                test_preds[name] = (pred_val > 0).astype(int)

        # OOF storage
        for j, name in enumerate(["xgb_cls", "xgb_reg", "lgbm_cls", "lgbm_reg"]):
            if name in test_preds:
                end_idx = min(val_end + len(test_preds[name]), n)
                oof_predictions[val_end:end_idx, j] = test_preds[name][:end_idx - val_end]
        for k, name in enumerate(["xgb_cls", "lgbm_cls"]):
            if name in test_probas:
                end_idx = min(val_end + len(test_probas[name]), n)
                oof_probas[val_end:end_idx, k] = test_probas[name][:end_idx - val_end]

        # Ensemble
        if test_preds:
            ensemble_dir = np.round(np.mean(list(test_preds.values()), axis=0)).astype(int)
        else:
            ensemble_dir = np.zeros(len(d_test))

        # Metrics
        fold_result = {"fold": fold, "n_train": len(X_train), "n_test": len(X_test)}
        for name, pred in test_preds.items():
            fold_result[f"{name}_dir_acc"] = round(float(accuracy_score(d_test[:len(pred)], pred)), 4)

        ens_acc = float(accuracy_score(d_test[:len(ensemble_dir)], ensemble_dir))
        fold_result["ensemble_dir_acc"] = round(ens_acc, 4)

        # High-confidence
        if "xgb_cls" in test_probas:
            proba = test_probas["xgb_cls"]
            high_conf = (proba > 0.6) | (proba < 0.4)
            if high_conf.sum() > 0:
                fold_result["high_conf_acc"] = round(float(accuracy_score(
                    d_test[:len(proba)][high_conf], test_preds["xgb_cls"][high_conf]
                )), 4)
                fold_result["high_conf_coverage"] = round(float(high_conf.mean()), 4)

        if "xgb_reg" in models:
            pred_reg = models["xgb_reg"].predict(X_test_s)
            fold_result["xgb_reg_mae"] = round(float(mean_absolute_error(y_test[:len(pred_reg)], pred_reg)), 6)
            fold_result["xgb_reg_rmse"] = round(float(np.sqrt(mean_squared_error(y_test[:len(pred_reg)], pred_reg))), 6)

        fold_metrics.append(fold_result)

        if ens_acc > best_val_acc:
            best_val_acc = ens_acc
            best_models = models.copy()

        logger.info(
            f"  Fold {fold}: ens={ens_acc:.4f} | "
            f"xgb={fold_result.get('xgb_cls_dir_acc', 'N/A')} | "
            f"lgbm={fold_result.get('lgbm_cls_dir_acc', 'N/A')} | "
            f"HC={fold_result.get('high_conf_acc', 'N/A')} "
            f"({fold_result.get('high_conf_coverage', 'N/A')})"
        )

    # Meta-learner
    meta_learner = None
    valid_oof = ~np.isnan(oof_predictions[:, 0])
    if valid_oof.sum() > 50:
        meta_X = np.nan_to_num(oof_predictions[valid_oof], nan=0.5)
        valid_proba_mask = ~np.isnan(oof_probas[valid_oof, 0])
        if valid_proba_mask.sum() > 0:
            meta_X = np.column_stack([meta_X, np.nan_to_num(oof_probas[valid_oof], nan=0.5)])
        meta_y = direction[valid_oof]
        try:
            meta_learner = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                verbosity=0, random_state=42,
            )
            meta_learner.fit(meta_X, meta_y)
            meta_acc = accuracy_score(meta_y, meta_learner.predict(meta_X))
            logger.info(f"  Meta-learner accuracy (in-sample): {meta_acc:.4f}")
        except Exception as e:
            logger.warning(f"  Meta-learner falhou: {e}")

    # Retrain on full data
    logger.info(f"  Retraining on full data ({n} samples)...")
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_sel)
    final_models = {}

    for name, ModelCls, target_data, extra in [
        ("xgb_cls", XGBClassifier, direction, {"eval_metric": "logloss"}),
        ("xgb_reg", XGBRegressor, y, {"eval_metric": "rmse"}),
        ("lgbm_cls", LGBMClassifier, direction, {"verbose": -1}),
        ("lgbm_reg", LGBMRegressor, y, {"verbose": -1}),
    ]:
        try:
            base_params = dict(
                n_estimators=500, max_depth=7, learning_rate=0.01,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            )
            if "lgbm" in name:
                base_params["num_leaves"] = 31
            base_params.update(extra)
            model = ModelCls(**base_params)
            model.fit(X_full_s, target_data)
            final_models[name] = model
        except Exception as e:
            logger.warning(f"  {name} retrain falhou: {e}")

    # Save
    coin_dir = models_dir / coin
    coin_dir.mkdir(parents=True, exist_ok=True)

    for name, model in final_models.items():
        joblib.dump(model, coin_dir / f"{name}.joblib")
    if meta_learner:
        joblib.dump(meta_learner, coin_dir / "meta_learner.joblib")
    joblib.dump(scaler_full, coin_dir / "scaler.joblib")

    with open(coin_dir / "feature_columns.json", "w") as f:
        json.dump(selected_names, f, indent=2)
    with open(coin_dir / "selected_feature_indices.json", "w") as f:
        json.dump(selected_idx, f, indent=2)

    avg_metrics = {}
    for key in fold_metrics[0].keys():
        if key in ("fold", "n_train", "n_test"):
            continue
        values = [fm[key] for fm in fold_metrics if key in fm]
        if values:
            avg_metrics[f"avg_{key}"] = round(float(np.mean(values)), 4)
            avg_metrics[f"std_{key}"] = round(float(np.std(values)), 4)

    with open(coin_dir / "metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)
    with open(coin_dir / "fold_metrics.json", "w") as f:
        json.dump(fold_metrics, f, indent=2)

    if "xgb_cls" in final_models:
        imp = final_models["xgb_cls"].feature_importances_
        importance_dict = dict(zip(selected_names, [round(float(v), 6) for v in imp]))
        importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        with open(coin_dir / "feature_importance.json", "w") as f:
            json.dump(importance_sorted, f, indent=2)

    ensemble_data = {
        "models": list(final_models.keys()),
        "n_features": len(selected_names),
        "n_folds": N_FOLDS,
        "has_meta_learner": meta_learner is not None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(coin_dir / "ensemble_data.json", "w") as f:
        json.dump(ensemble_data, f, indent=2)

    return avg_metrics


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("CRIPTO DT — Treinamento XGBoost/LightGBM + Stacking")
    logger.info("=" * 70)

    storage = DataStorage(config)
    models_dir = config.training.models_dir

    logger.info("Carregando dados...")
    all_data = {}
    for coin in config.data.coins:
        try:
            df = storage.load_ohlcv(coin, "1d")
            if df is not None and len(df) > 100:
                all_data[coin] = df
                logger.info(f"  {coin}: {len(df)} candles")
        except Exception as e:
            logger.warning(f"  {coin}: {e}")

    if not all_data:
        logger.error("Nenhum dado carregado! Abortando.")
        return

    logger.info("\nGerando features (pipeline completo)...")
    processed_data = generate_features_all(all_data)

    results_summary = {}
    for coin in config.data.coins:
        if coin not in processed_data:
            continue

        df = processed_data[coin]
        if "target" not in df.columns:
            continue

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Treinando {coin} ({len(df)} samples)")
        logger.info(f"{'=' * 50}")

        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        X = np.nan_to_num(df[feature_cols].values.astype(np.float32), nan=0, posinf=0, neginf=0)
        y = np.nan_to_num(df["target"].values.astype(np.float32), nan=0, posinf=0, neginf=0)

        logger.info(f"  Features: {len(feature_cols)}, Samples: {len(y)}")
        logger.info(f"  Balance: {(y > 0).mean():.1%} up / {(y <= 0).mean():.1%} down")

        try:
            metrics = walk_forward_train(X, y, feature_cols, coin, models_dir)
            results_summary[coin] = metrics
        except Exception as e:
            logger.error(f"  ERRO {coin}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"RELATÓRIO FINAL — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    logger.info(f"{'=' * 70}")

    if results_summary:
        all_accs = []
        all_hc = []
        for coin, m in results_summary.items():
            ea = m.get("avg_ensemble_dir_acc", 0)
            hc = m.get("avg_high_conf_acc", "N/A")
            all_accs.append(ea)
            if isinstance(hc, float):
                all_hc.append(hc)
            logger.info(f"  {coin:6s}: ens={ea:.4f} | HC={hc}")

        logger.info(f"\n  MEDIA GERAL: {np.mean(all_accs):.4f}")
        if all_hc:
            logger.info(f"  MEDIA HIGH-CONF: {np.mean(all_hc):.4f}")

        report = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "avg_ensemble_dir_acc": round(float(np.mean(all_accs)), 4),
            "avg_high_conf_acc": round(float(np.mean(all_hc)), 4) if all_hc else "N/A",
            "per_coin": results_summary,
        }
        Path("reports").mkdir(exist_ok=True)
        with open("reports/training_report.json", "w") as f:
            json.dump(report, f, indent=2)

    logger.info("\nConcluido!")


if __name__ == "__main__":
    main()
