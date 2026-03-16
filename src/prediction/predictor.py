"""Gerador de previsoes: carrega modelos treinados e gera previsoes."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import Config, config as default_config
from src.data.collector import DataCollector
from src.data.storage import DataStorage
from src.features.pipeline import FeaturePipeline
from src.models.lstm_gru import SequenceModel
from src.models.xgboost_model import XGBoostRegressor, XGBoostClassifier
from src.models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)


def _safe_load_model(path: Path, model_class):
    """Carrega modelo se o arquivo existir."""
    if path.exists():
        try:
            return model_class.load(path)
        except Exception as e:
            logger.warning(f"Erro ao carregar {path}: {e}")
    return None


def _safe_import_and_load(path: Path, module_path: str, class_name: str):
    """Importa classe e carrega modelo se ambos disponiveis."""
    if not path.exists():
        return None
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
        return cls.load(path)
    except (ImportError, AttributeError, Exception) as e:
        logger.debug(f"Nao carregado {class_name} de {path}: {e}")
        return None


class Predictor:
    """Gera previsoes usando modelos treinados."""

    def __init__(self, config: Config = default_config):
        self.config = config
        self.collector = DataCollector(config)
        self.storage = DataStorage(config)
        self.feature_pipeline = FeaturePipeline(config)

    def load_models(self, coin: str) -> dict:
        """Carrega modelos treinados para uma moeda."""
        coin_dir = self.config.training.models_dir / coin

        if not coin_dir.exists():
            raise FileNotFoundError(f"Modelos nao encontrados para {coin}: {coin_dir}")

        models = {}

        # Modelos core (sempre disponiveis)
        xgb_reg = _safe_load_model(coin_dir / "xgb_reg.joblib", XGBoostRegressor)
        if xgb_reg:
            models["xgb_reg"] = xgb_reg

        xgb_cls = _safe_load_model(coin_dir / "xgb_cls.joblib", XGBoostClassifier)
        if xgb_cls:
            models["xgb_cls"] = xgb_cls

        lstm = _safe_load_model(coin_dir / "lstm.pt", SequenceModel)
        if lstm:
            models["lstm"] = lstm

        gru = _safe_load_model(coin_dir / "gru.pt", SequenceModel)
        if gru:
            models["gru"] = gru

        # Modelos opcionais — importacao segura
        optional_models = {
            "lgbm_reg": ("lgbm_reg.joblib", "src.models.lightgbm_model", "LightGBMRegressor"),
            "rf_reg": ("rf_reg.joblib", "src.models.random_forest", "RandomForestRegressorModel"),
            "svm_reg": ("svm_reg.joblib", "src.models.svm_model", "SVMRegressor"),
            "cnn_lstm": ("cnn_lstm.pt", "src.models.cnn_lstm", "CNNLSTMModel"),
            "tcn": ("tcn.pt", "src.models.tcn", "TCNModel"),
            "helformer": ("helformer.pt", "src.models.helformer", "HelformerModel"),
            "mdn": ("mdn.pt", "src.models.mdn", "MDNModel"),
            "bnn": ("bnn.pt", "src.models.bnn", "BNNModel"),
        }
        for name, (filename, mod_path, cls_name) in optional_models.items():
            m = _safe_import_and_load(coin_dir / filename, mod_path, cls_name)
            if m:
                models[name] = m

        # Ensemble
        ens = _safe_load_model(coin_dir / "ensemble.joblib", EnsembleModel)
        if ens:
            models["ensemble"] = ens

        # Conformal
        conformal = None
        conformal_path = coin_dir / "conformal.joblib"
        if conformal_path.exists():
            try:
                from src.models.conformal import ConformalPredictor
                conformal = ConformalPredictor.load(conformal_path)
            except (ImportError, Exception) as e:
                logger.warning(f"Conformal nao carregado: {e}")

        # Scaler
        scaler = joblib.load(coin_dir / "scaler.joblib")

        # Feature columns
        with open(coin_dir / "feature_columns.json") as f:
            feature_columns = json.load(f)

        logger.info(f"Modelos carregados para {coin}: {list(models.keys())}")
        return {
            "models": models,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "conformal": conformal,
        }

    def predict_coin(self, coin: str, df: pd.DataFrame | None = None) -> dict:
        """Gera previsao para uma moeda."""
        loaded = self.load_models(coin)
        models = loaded["models"]
        scaler = loaded["scaler"]
        feature_columns = loaded["feature_columns"]
        conformal = loaded.get("conformal")

        # Obter dados
        if df is None:
            df = self.collector.fetch_ohlcv(coin, "1d", since_days=120)

        if df.empty:
            raise ValueError(f"Dados vazios para {coin}")

        # BTC para correlacao
        btc_df = None
        if coin != "BTC":
            btc_df = self.collector.fetch_ohlcv("BTC", "1d", since_days=120)

        # Feature engineering
        is_btc = coin == "BTC"
        df_features = self.feature_pipeline.transform(df, btc_df=btc_df, is_btc=is_btc)

        if df_features.empty:
            raise ValueError(f"Features vazias para {coin}")

        # Align features: fill any columns the model expects but the data is missing
        # (e.g. ema_200 needs 200+ days of data, but prediction only fetches 120 days)
        missing_cols = [c for c in feature_columns if c not in df_features.columns]
        if missing_cols:
            logger.warning(
                f"{coin}: {len(missing_cols)} features ausentes no prediction "
                f"(preenchidas com 0): {missing_cols}"
            )
            for col in missing_cols:
                df_features[col] = 0.0

        # Extrair e escalar features
        X = df_features[feature_columns].fillna(0).values.astype(np.float32)
        X_scaled = scaler.scaler.transform(X)

        # Previsoes de cada modelo
        base_preds = {}
        base_confidence = {}

        for name, model in models.items():
            if name == "ensemble":
                continue
            try:
                pred, conf = model.predict_with_confidence(X_scaled)
                base_preds[name] = float(pred[-1])
                base_confidence[name] = float(conf[-1])
            except Exception as e:
                logger.warning(f"Erro ao prever com {name}: {e}")

        # Ensemble
        if "ensemble" in models and len(base_preds) > 0:
            ens_model = models["ensemble"]
            # Use the model_names stored during training to ensure correct
            # column ordering.  Fall back to dict insertion order if the
            # ensemble was trained before model_names were persisted.
            if hasattr(ens_model, "model_names") and ens_model.model_names:
                ordered_names = ens_model.model_names
            else:
                ordered_names = list(base_preds.keys())

            # Build prediction array in the same column order as training
            ordered_vals = []
            for name in ordered_names:
                if name in base_preds:
                    ordered_vals.append(base_preds[name])
                else:
                    # Model missing at prediction time — use 0 (neutral)
                    logger.warning(f"  Modelo {name} ausente na predicao, usando 0")
                    ordered_vals.append(0.0)

            pred_array = np.array([ordered_vals]).reshape(1, -1)
            ensemble_pred, ensemble_conf = ens_model.predict_with_confidence(
                pred_array
            )
            final_pred = float(ensemble_pred[0])
            final_conf = float(ensemble_conf[0])
        else:
            final_pred = float(np.mean(list(base_preds.values())))
            confs = list(base_confidence.values())
            final_conf = float(np.mean(confs)) if confs else 0.5

        # Preco atual e previsto
        current_price = float(df_features["close"].iloc[-1])
        predicted_price = current_price * np.exp(final_pred)
        direction = "ALTA" if final_pred > 0 else "BAIXA"

        # Conformal prediction intervals
        pred_lower = None
        pred_upper = None
        if conformal is not None:
            try:
                lower, upper = conformal.predict_intervals(np.array([final_pred]))
                pred_lower = float(current_price * np.exp(lower[0]))
                pred_upper = float(current_price * np.exp(upper[0]))
            except Exception as e:
                logger.warning(f"Conformal interval falhou: {e}")

        # MDN scenarios (se disponivel)
        scenarios = None
        if "mdn" in models:
            try:
                mdn_model = models["mdn"]
                if hasattr(mdn_model, "predict_quantiles"):
                    quantiles = mdn_model.predict_quantiles(
                        X_scaled, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
                    )
                    if quantiles is not None:
                        scenarios = {
                            f"q{int(q*100)}": float(current_price * np.exp(v[-1]))
                            for q, v in zip([0.05, 0.25, 0.5, 0.75, 0.95], quantiles)
                        }
            except Exception as e:
                logger.warning(f"MDN scenarios falhou: {e}")

        result = {
            "coin": coin,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_price": current_price,
            "predicted_return": final_pred,
            "predicted_price": predicted_price,
            "direction": direction,
            "confidence": final_conf,
            "model_predictions": base_preds,
            "model_confidence": base_confidence,
            "n_models": len(base_preds),
        }

        if pred_lower is not None:
            result["price_lower_90"] = pred_lower
            result["price_upper_90"] = pred_upper

        if scenarios is not None:
            result["scenarios"] = scenarios

        logger.info(
            f"{coin}: {direction} | "
            f"Retorno previsto: {final_pred:+.4f} | "
            f"Preco previsto: ${predicted_price:,.2f} | "
            f"Confianca: {final_conf:.2%} | "
            f"Modelos: {len(base_preds)}"
        )

        return result

    def predict_all(self) -> pd.DataFrame:
        """Gera previsoes para todas as moedas configuradas."""
        predictions = []
        for coin in self.config.data.coins:
            try:
                pred = self.predict_coin(coin)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Erro ao prever {coin}: {e}")

        if not predictions:
            return pd.DataFrame()

        df = pd.DataFrame(predictions)

        # Salvar
        filename = f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
        self.storage.save_predictions(df, filename)

        return df
