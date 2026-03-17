#!/usr/bin/env python3
"""Script para preparar dados e executar treinamento.

Pode ser executado:
1. Localmente (se sklearn/xgboost/lightgbm funcionam)
2. No Docker: docker-compose run trainer
3. No Kaggle: copiar para notebook, ajustar paths

Este script gera features usando apenas numpy (sem dependências pesadas)
e treina modelos XGBoost/LightGBM.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

os.environ["CRIPTO_DT_NO_TORCH"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

import numpy as np
import pandas as pd

# Tentar importar modelos reais, senão usar numpy fallback
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    import lightgbm
    HAS_ML = True
    logger.info("sklearn/xgboost/lightgbm disponíveis!")
except ImportError:
    HAS_ML = False
    logger.warning("sklearn/xgboost/lightgbm NÃO disponíveis. Usando numpy fallback.")

# Import project modules
from config.settings import config
from src.data.storage import DataStorage


# ============================================================
# Feature Generation (standalone, numpy-only)
# ============================================================

NON_FEATURE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume",
    "log_return", "pct_return", "direction", "target",
    "close_denoised", "high_denoised", "low_denoised", "open_denoised",
}


def generate_all_features(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Gera features para todas as moedas usando o pipeline completo."""
    try:
        from src.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(config)
        results = pipeline.transform_all(data)
        logger.info(f"Pipeline completo: {len(pipeline.feature_columns)} features")
        return results
    except Exception as e:
        logger.warning(f"Pipeline falhou: {e}. Gerando features inline.")
        return {coin: _inline_features(df) for coin, df in data.items() if len(df) > 50}


def _inline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gera features essenciais inline (fallback se pipeline falhar)."""
    from src.data.preprocessor import DataPreprocessor
    prep = DataPreprocessor()
    df = prep.prepare(df, horizon=1, target_type="log_return")

    c = df["close"]
    v = df.get("volume", pd.Series(1, index=df.index))
    h = df.get("high", c)
    l = df.get("low", c)
    o = df.get("open", c)
    lr = df.get("log_return", c.pct_change())

    # Returns & momentum
    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_lag{lag}"] = lr.shift(lag)
    for w in [3, 5, 10, 20]:
        df[f"ret_cum{w}"] = lr.rolling(w).sum()
        df[f"ret_mean{w}"] = lr.rolling(w).mean()
        df[f"ret_std{w}"] = lr.rolling(w).std()
        df[f"vol_ratio{w}"] = v / v.rolling(w).mean()
        df[f"mom{w}"] = c / c.shift(w) - 1

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # Bollinger
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_pband"] = (c - ma20) / (2 * std20 + 1e-10)
    df["bb_width"] = 4 * std20 / (ma20 + 1e-10)

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["macd_hist"] = macd - signal
    df["macd_signal_cross"] = np.sign(macd - signal)

    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / (c + 1e-10)

    # Candlestick patterns
    body = c - o
    body_abs = body.abs()
    total_range = (h - l).clip(lower=1e-10)
    df["candle_body_pct"] = body_abs / total_range
    df["candle_upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / total_range
    df["candle_lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / total_range
    df["candle_direction"] = np.sign(body)

    # Doji, hammer, shooting star
    df["is_doji"] = (df["candle_body_pct"] < 0.1).astype(float)
    df["is_hammer"] = ((df["candle_lower_shadow"] > 0.6) & (df["candle_upper_shadow"] < 0.1)).astype(float)
    df["is_shooting_star"] = ((df["candle_upper_shadow"] > 0.6) & (df["candle_lower_shadow"] < 0.1)).astype(float)

    # Volume features
    df["vol_change"] = v.pct_change()
    df["vol_ma_ratio"] = v / v.rolling(20).mean()
    df["price_vol_corr"] = lr.rolling(20).corr(v.pct_change())

    # Hurst exponent proxy (autocorrelation)
    df["ret_autocorr1"] = lr.rolling(20).apply(lambda x: x.autocorr(1) if len(x) > 2 else 0, raw=False)

    # Market structure
    df["higher_high"] = (h > h.shift(1)).astype(float).rolling(5).mean()
    df["lower_low"] = (l < l.shift(1)).astype(float).rolling(5).mean()

    df = df.ffill().bfill().fillna(0)
    if "target" in df.columns:
        df = df.dropna(subset=["target"]).reset_index(drop=True)

    return df


# ============================================================
# Feature Selection
# ============================================================

def select_features(X, y, names, max_feat=60):
    """Seleciona features via variance + correlation + importance."""
    # Variance filter
    var = np.var(X, axis=0)
    mask = var > 0.001
    idx = np.where(mask)[0]
    logger.info(f"  Variance: {len(idx)}/{X.shape[1]}")

    # Correlation filter
    if len(idx) > max_feat * 2:
        Xv = X[:, idx]
        corr = np.corrcoef(Xv.T)
        remove = set()
        for i in range(len(idx)):
            if i in remove:
                continue
            for j in range(i + 1, len(idx)):
                if j in remove:
                    continue
                if abs(corr[i, j]) > 0.95:
                    remove.add(j if var[idx[i]] >= var[idx[j]] else i)
        idx = idx[[i for i in range(len(idx)) if i not in remove]]
        logger.info(f"  Correlation: {len(idx)}")

    # XGBoost importance (se disponível)
    if HAS_ML and len(idx) > max_feat:
        d = (y > 0).astype(int)
        m = XGBClassifier(n_estimators=50, max_depth=4, verbosity=0, random_state=42)
        m.fit(X[:, idx], d)
        top = np.argsort(m.feature_importances_)[::-1][:max_feat]
        idx = idx[top]
        logger.info(f"  Importance: {len(idx)}")

    return idx.tolist(), [names[i] for i in idx]


# ============================================================
# Training
# ============================================================

def train_coin(X, y, feat_names, coin, models_dir):
    """Treina modelos para uma moeda."""
    n = len(y)
    direction = (y > 0).astype(int)
    N_FOLDS = 8

    # Feature selection
    sel_idx, sel_names = select_features(X[:int(n * 0.85)], y[:int(n * 0.85)], feat_names)
    Xs = X[:, sel_idx]
    logger.info(f"  {len(sel_names)} features selecionadas")

    fold_metrics = []

    for fold in range(N_FOLDS):
        fs = n // (N_FOLDS + 2)
        tr_end = fs * (fold + 2)
        va_end = tr_end + int(fs * 0.15 / 0.70)
        te_end = min(va_end + int(fs * 0.15 / 0.70), n)

        if te_end > n or tr_end >= n:
            continue

        Xtr, ytr, dtr = Xs[:tr_end], y[:tr_end], direction[:tr_end]
        Xva, yva, dva = Xs[tr_end:va_end], y[tr_end:va_end], direction[tr_end:va_end]
        Xte, yte, dte = Xs[va_end:te_end], y[va_end:te_end], direction[va_end:te_end]

        if len(Xva) < 5 or len(Xte) < 5:
            continue

        if HAS_ML:
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xva_s = scaler.transform(Xva)
            Xte_s = scaler.transform(Xte)
        else:
            mu, sigma = Xtr.mean(0), Xtr.std(0) + 1e-10
            Xtr_s = (Xtr - mu) / sigma
            Xva_s = (Xva - mu) / sigma
            Xte_s = (Xte - mu) / sigma

        preds = {}
        probas = {}

        if HAS_ML:
            # XGBClassifier
            try:
                m = XGBClassifier(
                    n_estimators=500, max_depth=7, learning_rate=0.01,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.1, reg_lambda=1.0,
                    eval_metric="logloss", verbosity=0, random_state=42,
                    early_stopping_rounds=50,
                )
                m.fit(Xtr_s, dtr, eval_set=[(Xva_s, dva)], verbose=False)
                preds["xgb_cls"] = m.predict(Xte_s)
                probas["xgb_cls"] = m.predict_proba(Xte_s)[:, 1]
            except Exception as e:
                logger.warning(f"    XGB fold {fold}: {e}")

            # LGBMClassifier
            try:
                m2 = LGBMClassifier(
                    n_estimators=500, max_depth=7, learning_rate=0.01,
                    num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=42,
                )
                m2.fit(Xtr_s, dtr, eval_set=[(Xva_s, dva)],
                       callbacks=[lightgbm.early_stopping(50, verbose=False), lightgbm.log_evaluation(0)])
                preds["lgbm_cls"] = m2.predict(Xte_s)
                probas["lgbm_cls"] = m2.predict_proba(Xte_s)[:, 1]
            except Exception as e:
                logger.warning(f"    LGBM fold {fold}: {e}")

            # XGBRegressor
            try:
                m3 = XGBRegressor(
                    n_estimators=500, max_depth=7, learning_rate=0.01,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.1, reg_lambda=1.0,
                    eval_metric="rmse", verbosity=0, random_state=42,
                    early_stopping_rounds=50,
                )
                m3.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], verbose=False)
                preds["xgb_reg"] = (m3.predict(Xte_s) > 0).astype(int)
            except Exception as e:
                logger.warning(f"    XGB_reg fold {fold}: {e}")
        else:
            # Numpy fallback: Ridge classifier
            lam = 1.0
            XtX = Xtr_s.T @ Xtr_s + lam * np.eye(Xtr_s.shape[1])
            Xty = Xtr_s.T @ ytr
            w = np.linalg.solve(XtX, Xty)
            pred_val = Xte_s @ w
            preds["ridge"] = (pred_val > 0).astype(int)

        # Ensemble
        if preds:
            ens = np.round(np.mean(list(preds.values()), axis=0)).astype(int)
        else:
            ens = np.zeros(len(dte))

        fm = {"fold": fold, "n_test": len(dte)}
        for name, pred in preds.items():
            fm[f"{name}_acc"] = round(float((pred == dte[:len(pred)]).mean()), 4)

        fm["ensemble_acc"] = round(float((ens == dte[:len(ens)]).mean()), 4)

        # High-confidence
        for name, prob in probas.items():
            hc = (prob > 0.6) | (prob < 0.4)
            if hc.sum() > 0:
                fm[f"{name}_hc_acc"] = round(float((preds[name][hc] == dte[:len(prob)][hc]).mean()), 4)
                fm[f"{name}_hc_cov"] = round(float(hc.mean()), 4)

        fold_metrics.append(fm)
        logger.info(
            f"    Fold {fold}: ens={fm['ensemble_acc']:.3f} "
            f"xgb={fm.get('xgb_cls_acc', 'N/A')} "
            f"HC={fm.get('xgb_cls_hc_acc', 'N/A')} ({fm.get('xgb_cls_hc_cov', 'N/A')})"
        )

    # Average metrics
    avg = {}
    if fold_metrics:
        for k in fold_metrics[0]:
            if k in ("fold", "n_test"):
                continue
            vals = [fm[k] for fm in fold_metrics if k in fm]
            if vals:
                avg[f"avg_{k}"] = round(float(np.mean(vals)), 4)
                avg[f"std_{k}"] = round(float(np.std(vals)), 4)

    # Retrain on full data and save
    if HAS_ML:
        scaler_full = StandardScaler()
        Xf = scaler_full.fit_transform(Xs)

        coin_dir = models_dir / coin
        coin_dir.mkdir(parents=True, exist_ok=True)

        import joblib

        # Train final models
        for name, Cls, target, extra in [
            ("xgb_cls", XGBClassifier, direction, {"eval_metric": "logloss"}),
            ("xgb_reg", XGBRegressor, y, {"eval_metric": "rmse"}),
            ("lgbm_cls", LGBMClassifier, direction, {"verbose": -1}),
            ("lgbm_reg", LGBMRegressor, y, {"verbose": -1}),
        ]:
            try:
                params = dict(n_estimators=500, max_depth=7, learning_rate=0.01,
                              subsample=0.8, colsample_bytree=0.7,
                              reg_alpha=0.1, reg_lambda=1.0, random_state=42)
                if "lgbm" in name:
                    params["num_leaves"] = 31
                params.update(extra)
                model = Cls(**params)
                model.fit(Xf, target)
                joblib.dump(model, coin_dir / f"{name}.joblib")
            except Exception as e:
                logger.warning(f"  {name} retrain: {e}")

        joblib.dump(scaler_full, coin_dir / "scaler.joblib")

        with open(coin_dir / "feature_columns.json", "w") as f:
            json.dump(sel_names, f, indent=2)
        with open(coin_dir / "metrics.json", "w") as f:
            json.dump(avg, f, indent=2)
        with open(coin_dir / "fold_metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

        # Feature importance
        try:
            import joblib
            xgb_model = joblib.load(coin_dir / "xgb_cls.joblib")
            imp = dict(zip(sel_names, [round(float(v), 6) for v in xgb_model.feature_importances_]))
            imp = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
            with open(coin_dir / "feature_importance.json", "w") as f:
                json.dump(imp, f, indent=2)
        except Exception:
            pass

        with open(coin_dir / "ensemble_data.json", "w") as f:
            json.dump({"models": ["xgb_cls", "xgb_reg", "lgbm_cls", "lgbm_reg"],
                        "n_features": len(sel_names), "timestamp": datetime.now().isoformat()}, f, indent=2)

    return avg


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("CRIPTO DT — Treinamento Otimizado")
    logger.info(f"ML libs: {'SIM' if HAS_ML else 'NAO (numpy fallback)'}")
    logger.info("=" * 60)

    storage = DataStorage(config)
    models_dir = config.training.models_dir

    # Load data
    logger.info("\nCarregando dados...")
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
        logger.error("Sem dados! Abortando.")
        return

    # Generate features
    logger.info("\nGerando features...")
    processed = generate_all_features(all_data)

    # Train
    results = {}
    for coin in config.data.coins:
        if coin not in processed:
            continue
        df = processed[coin]
        if "target" not in df.columns:
            continue

        logger.info(f"\n{'=' * 40}")
        logger.info(f"  {coin} ({len(df)} samples)")
        logger.info(f"{'=' * 40}")

        feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        X = np.nan_to_num(df[feat_cols].values.astype(np.float32))
        y = np.nan_to_num(df["target"].values.astype(np.float32))

        logger.info(f"  {len(feat_cols)} features, {(y > 0).mean():.1%} up")

        try:
            results[coin] = train_coin(X, y, feat_cols, coin, models_dir)
        except Exception as e:
            logger.error(f"  ERRO {coin}: {e}")
            import traceback
            traceback.print_exc()

    # Report
    elapsed = time.time() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"RESULTADO — {elapsed:.0f}s")
    logger.info(f"{'=' * 60}")

    accs, hcs = [], []
    for coin, m in results.items():
        ea = m.get("avg_ensemble_acc", 0)
        ha = m.get("avg_xgb_cls_hc_acc", "N/A")
        accs.append(ea)
        if isinstance(ha, float):
            hcs.append(ha)
        logger.info(f"  {coin:6s}: ens={ea:.4f} HC={ha}")

    if accs:
        logger.info(f"\n  MEDIA: {np.mean(accs):.4f}")
        if hcs:
            logger.info(f"  MEDIA HC: {np.mean(hcs):.4f}")

    # Save report
    Path("reports").mkdir(exist_ok=True)
    with open("reports/training_report.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "elapsed": round(elapsed, 1),
            "avg_acc": round(float(np.mean(accs)), 4) if accs else 0,
            "avg_hc": round(float(np.mean(hcs)), 4) if hcs else "N/A",
            "per_coin": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
