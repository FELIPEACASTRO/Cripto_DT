"""Orquestrador de treino: coordena modelos, walk-forward e ensemble."""

import dataclasses
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import Config, config as default_config
from src.data.preprocessor import FeatureScaler
from src.features.pipeline import FeaturePipeline
from src.models.lstm_gru import SequenceModel
from src.models.xgboost_model import XGBoostRegressor, XGBoostClassifier
from src.models.ensemble import EnsembleModel
from src.training.walk_forward import WalkForwardValidator
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _safe_import(module_path: str, class_name: str):
    """Importa uma classe de forma segura."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        logger.debug(f"{class_name} nao disponivel")
        return None


def _align_sequence_preds(preds: np.ndarray, target_len: int) -> np.ndarray:
    """Alinha previsoes de modelos de sequencia ao tamanho do target."""
    if len(preds) < target_len:
        return preds
    return preds[:target_len]


class Trainer:
    """Orquestra treino completo: features -> walk-forward -> modelos -> ensemble.

    Modelos suportados (20 total):
    - Core: XGBoost Reg/Cls, LSTM, GRU
    - Tree-based: LightGBM Reg/Cls, Random Forest Reg
    - Deep Learning: CNN-LSTM, TCN, Helformer, Mamba SSM
    - Probabilistico: MDN, BNN, Evidential Regression
    - Dual: CryptoPulse Dual-Prediction
    - Classico: SVM Reg
    - Meta: Ensemble (Performance Weighted), Evolutionary Ensemble (Sakana GA)
    - Incerteza: Conformal Prediction
    """

    def __init__(self, config: Config = default_config):
        self.config = config
        self.feature_pipeline = FeaturePipeline(config)
        self.walk_forward = WalkForwardValidator(
            n_splits=config.training.walk_forward_splits,
            train_ratio=config.training.train_ratio,
            val_ratio=config.training.val_ratio,
            test_ratio=config.training.test_ratio,
        )

    def _prepare_data(
        self, df: pd.DataFrame, feature_columns: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrai arrays X e y do DataFrame."""
        X = df[feature_columns].values.astype(np.float32)
        y = df["target"].values.astype(np.float32)
        return X, y

    def _train_sequence_model(
        self, name: str, cell_type: str,
        X_train_s, y_train, X_val_s, y_val, X_test_s,
    ) -> tuple:
        """Treina um modelo de sequencia (LSTM/GRU) e retorna previsoes."""
        logger.info(f"  Treinando {name}...")
        model = SequenceModel(cell_type, lstm_config=self.config.lstm)
        model.fit(X_train_s, y_train, X_val_s, y_val)
        val_pred = model.predict(X_val_s)
        test_pred = model.predict(X_test_s)
        return model, val_pred, test_pred

    def _train_optional_model(
        self, name: str, module_path: str, class_name: str,
        X_train_s, y_train, X_val_s, y_val, X_test_s,
        is_sequence: bool = False, **kwargs,
    ) -> tuple | None:
        """Treina um modelo opcional se disponivel."""
        Cls = _safe_import(module_path, class_name)
        if Cls is None:
            return None
        try:
            logger.info(f"  Treinando {name}...")
            model = Cls(**kwargs)
            model.fit(X_train_s, y_train, X_val_s, y_val)
            val_pred = model.predict(X_val_s)
            test_pred = model.predict(X_test_s)
            if is_sequence:
                val_pred = _align_sequence_preds(val_pred, len(y_val))
                test_pred = _align_sequence_preds(test_pred, len(X_test_s))
            return model, val_pred, test_pred
        except Exception as e:
            logger.warning(f"  {name} falhou: {e}")
            return None

    def _apply_feature_selection(
        self, X: np.ndarray, y: np.ndarray, feature_columns: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Aplica selecao de features se configurado."""
        method = self.config.features.feature_selection_method
        if method == "none":
            return X, feature_columns

        FeatureSelector = _safe_import(
            "src.features.feature_selection", "FeatureSelector"
        )
        if FeatureSelector is None:
            return X, feature_columns

        try:
            selector = FeatureSelector()
            selected = selector.select_features(X, y, feature_columns, method=method)
            if not selected or len(selected) < 5:
                logger.warning(
                    f"Feature selection retornou apenas {len(selected or [])} features, "
                    f"mantendo todas as {len(feature_columns)}"
                )
                return X, feature_columns
            # Filtrar colunas
            indices = [feature_columns.index(f) for f in selected if f in feature_columns]
            X_selected = X[:, indices]
            logger.info(
                f"Feature selection ({method}): {len(feature_columns)} -> {len(selected)} features"
            )
            return X_selected, selected
        except Exception as e:
            logger.warning(f"Feature selection falhou: {e}. Usando todas as features.")
            return X, feature_columns

    def train_coin(
        self, coin: str, df_features: pd.DataFrame, feature_columns: list[str]
    ) -> dict:
        """Treina todos os modelos para uma moeda."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Treinando modelos para {coin}")
        logger.info(f"{'='*60}")

        X, y = self._prepare_data(df_features, feature_columns)
        logger.info(f"Dados: {X.shape[0]} amostras, {X.shape[1]} features")

        # Feature selection (antes do treino)
        X, feature_columns = self._apply_feature_selection(X, y, feature_columns)

        splits = self.walk_forward.split_data(X, y)

        all_fold_metrics = []
        best_models = {}
        best_scaler = None
        conformal_residuals = []

        for fold_idx, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(splits):
            logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
            logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # Scaler fitado apenas no treino
            scaler = FeatureScaler()
            df_train_temp = pd.DataFrame(X_train, columns=feature_columns)
            scaler.fit(df_train_temp, feature_columns)

            X_train_s = scaler.scaler.transform(X_train)
            X_val_s = scaler.scaler.transform(X_val)
            X_test_s = scaler.scaler.transform(X_test)

            fold_preds = {}
            fold_val_preds = {}
            fold_metrics = {"fold": fold_idx}

            # --- XGBoost Regressor ---
            logger.info("  Treinando XGBoost Regressor...")
            xgb_reg = XGBoostRegressor(self.config.xgboost)
            xgb_reg.fit(X_train_s, y_train, X_val_s, y_val)
            fold_val_preds["xgb_reg"] = xgb_reg.predict(X_val_s)
            fold_preds["xgb_reg"] = xgb_reg.predict(X_test_s)

            # --- XGBoost Classifier ---
            xgb_cls = None
            if self.config.xgboost.use_direction:
                logger.info("  Treinando XGBoost Classifier...")
                xgb_cls = XGBoostClassifier(self.config.xgboost)
                xgb_cls.fit(X_train_s, y_train, X_val_s, y_val)
                scale = np.std(y_train) if np.std(y_train) > 0 else 1.0
                fold_val_preds["xgb_cls"] = (xgb_cls.predict(X_val_s) - 0.5) * 2 * scale
                fold_preds["xgb_cls"] = (xgb_cls.predict(X_test_s) - 0.5) * 2 * scale

            # --- LSTM ---
            lstm, lstm_val, lstm_test = self._train_sequence_model(
                "LSTM", "LSTM", X_train_s, y_train, X_val_s, y_val, X_test_s
            )
            fold_val_preds["lstm"] = _align_sequence_preds(lstm_val, len(y_val))
            fold_preds["lstm"] = _align_sequence_preds(lstm_test, len(y_test))

            # --- GRU ---
            gru, gru_val, gru_test = self._train_sequence_model(
                "GRU", "GRU", X_train_s, y_train, X_val_s, y_val, X_test_s
            )
            fold_val_preds["gru"] = _align_sequence_preds(gru_val, len(y_val))
            fold_preds["gru"] = _align_sequence_preds(gru_test, len(y_test))

            # --- LightGBM Regressor (opcional) ---
            lgbm_reg = None
            if self.config.training.use_lightgbm:
                lgbm_params = {
                    k: v for k, v in dataclasses.asdict(self.config.lightgbm).items()
                    if k != "use_direction"
                }
                result = self._train_optional_model(
                    "LightGBM Reg", "src.models.lightgbm_model", "LightGBMRegressor",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                    **lgbm_params,
                )
                if result:
                    lgbm_reg, lgbm_val, lgbm_test = result
                    fold_val_preds["lgbm_reg"] = lgbm_val
                    fold_preds["lgbm_reg"] = lgbm_test

            # --- Random Forest Regressor (opcional) ---
            rf_reg = None
            if self.config.training.use_random_forest:
                result = self._train_optional_model(
                    "Random Forest", "src.models.random_forest", "RandomForestRegressorModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                )
                if result:
                    rf_reg, rf_val, rf_test = result
                    fold_val_preds["rf_reg"] = rf_val
                    fold_preds["rf_reg"] = rf_test

            # --- SVM Regressor (opcional) ---
            svm_reg = None
            if self.config.training.use_svm:
                result = self._train_optional_model(
                    "SVM Reg", "src.models.svm_model", "SVMRegressor",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                )
                if result:
                    svm_reg, svm_val, svm_test = result
                    fold_val_preds["svm_reg"] = svm_val
                    fold_preds["svm_reg"] = svm_test

            # --- CNN-LSTM (opcional) ---
            cnn_lstm = None
            if self.config.training.use_cnn_lstm:
                result = self._train_optional_model(
                    "CNN-LSTM", "src.models.cnn_lstm", "CNNLSTMModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                    is_sequence=True,
                )
                if result:
                    cnn_lstm, cnn_val, cnn_test = result
                    fold_val_preds["cnn_lstm"] = _align_sequence_preds(cnn_val, len(y_val))
                    fold_preds["cnn_lstm"] = _align_sequence_preds(cnn_test, len(y_test))

            # --- TCN (opcional) ---
            tcn = None
            if self.config.training.use_tcn:
                result = self._train_optional_model(
                    "TCN", "src.models.tcn", "TCNModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                    is_sequence=True,
                )
                if result:
                    tcn, tcn_val, tcn_test = result
                    fold_val_preds["tcn"] = _align_sequence_preds(tcn_val, len(y_val))
                    fold_preds["tcn"] = _align_sequence_preds(tcn_test, len(y_test))

            # --- Helformer (opcional) ---
            helformer = None
            if self.config.training.use_helformer:
                result = self._train_optional_model(
                    "Helformer", "src.models.helformer", "HelformerModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                    is_sequence=True,
                )
                if result:
                    helformer, hel_val, hel_test = result
                    fold_val_preds["helformer"] = _align_sequence_preds(hel_val, len(y_val))
                    fold_preds["helformer"] = _align_sequence_preds(hel_test, len(y_test))

            # --- MDN (opcional) ---
            mdn = None
            if self.config.training.use_mdn:
                n_features = X_train_s.shape[1]
                result = self._train_optional_model(
                    "MDN", "src.models.mdn", "MDNModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                    input_size=n_features,
                )
                if result:
                    mdn, mdn_val, mdn_test = result
                    fold_val_preds["mdn"] = mdn_val
                    fold_preds["mdn"] = mdn_test

            # --- BNN (opcional) ---
            bnn = None
            if self.config.training.use_bnn:
                n_features = X_train_s.shape[1]
                result = self._train_optional_model(
                    "BNN", "src.models.bnn", "BNNModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                    input_size=n_features,
                )
                if result:
                    bnn, bnn_val, bnn_test = result
                    fold_val_preds["bnn"] = bnn_val
                    fold_preds["bnn"] = bnn_test

            # --- Mamba SSM (opcional) ---
            mamba = None
            if self.config.training.use_mamba:
                result = self._train_optional_model(
                    "Mamba SSM", "src.models.mamba_model", "MambaModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                    is_sequence=True,
                )
                if result:
                    mamba, mamba_val, mamba_test = result
                    fold_val_preds["mamba"] = _align_sequence_preds(mamba_val, len(y_val))
                    fold_preds["mamba"] = _align_sequence_preds(mamba_test, len(y_test))

            # --- Evidential Regression (opcional) ---
            evidential = None
            if self.config.training.use_evidential:
                n_features = X_train_s.shape[1]
                result = self._train_optional_model(
                    "Evidential", "src.models.evidential", "EvidentialModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                    input_size=n_features,
                )
                if result:
                    evidential, evid_val, evid_test = result
                    fold_val_preds["evidential"] = evid_val
                    fold_preds["evidential"] = evid_test

            # --- Foundation Models (Phase 5) ---
            # Chronos-Bolt (Amazon)
            chronos = None
            if self.config.training.use_chronos:
                result = self._train_optional_model(
                    "Chronos-Bolt", "src.models.chronos_model", "ChronosModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                )
                if result:
                    chronos, chron_val, chron_test = result
                    fold_val_preds["chronos"] = chron_val
                    fold_preds["chronos"] = chron_test

            # TTM (IBM)
            ttm = None
            if self.config.training.use_ttm:
                result = self._train_optional_model(
                    "TTM", "src.models.ttm_model", "TTMModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                )
                if result:
                    ttm, ttm_val, ttm_test = result
                    fold_val_preds["ttm"] = ttm_val
                    fold_preds["ttm"] = ttm_test

            # MOIRAI (Salesforce)
            moirai = None
            if self.config.training.use_moirai:
                result = self._train_optional_model(
                    "MOIRAI", "src.models.moirai_model", "MoiraiModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                )
                if result:
                    moirai, moirai_val, moirai_test = result
                    fold_val_preds["moirai"] = moirai_val
                    fold_preds["moirai"] = moirai_test

            # --- Dual Prediction / CryptoPulse (opcional) ---
            dual_pred = None
            if self.config.training.use_dual_prediction:
                result = self._train_optional_model(
                    "Dual Prediction", "src.models.dual_prediction", "DualPredictionModel",
                    X_train_s, y_train, X_val_s, y_val, X_test_s,
                )
                if result:
                    dual_pred, dual_val, dual_test = result
                    fold_val_preds["dual_pred"] = dual_val
                    fold_preds["dual_pred"] = dual_test

            # --- Ensemble ---
            logger.info("  Treinando Ensemble...")
            min_val_len = min(len(v) for v in fold_val_preds.values())
            min_test_len = min(len(v) for v in fold_preds.values())

            model_names = list(fold_val_preds.keys())
            val_matrix = np.column_stack([
                v[-min_val_len:] for v in fold_val_preds.values()
            ])
            test_matrix = np.column_stack([
                v[-min_test_len:] for v in fold_preds.values()
            ])
            y_val_aligned = y_val[-min_val_len:]
            y_test_aligned = y_test[-min_test_len:]

            # Compute per-model directional accuracy on validation set
            val_dir_accuracies = np.array([
                float(np.mean(
                    np.sign(fold_val_preds[name][-min_val_len:])
                    == np.sign(y_val_aligned)
                ))
                for name in model_names
            ])

            ensemble = EnsembleModel(method="performance_weighted")
            ensemble.fit(
                val_matrix, y_val_aligned,
                test_matrix, y_test_aligned,
                model_names=model_names,
                val_dir_accuracies=val_dir_accuracies,
            )
            ensemble_preds = ensemble.predict(test_matrix)

            # --- Evolutionary Ensemble (opcional) ---
            evo_ensemble = None
            if self.config.training.use_evolutionary_ensemble:
                EvoCls = _safe_import("src.models.evolutionary_ensemble", "EvolutionaryEnsemble")
                if EvoCls:
                    try:
                        logger.info("  Treinando Evolutionary Ensemble...")
                        evo_ensemble = EvoCls()
                        evo_ensemble.fit(
                            val_matrix, y_val_aligned,
                            test_matrix, y_test_aligned,
                            model_names=model_names,
                        )
                        evo_preds = evo_ensemble.predict(test_matrix)
                        evo_dir_acc = float(np.mean(np.sign(evo_preds) == np.sign(y_test_aligned)))
                        ens_dir_acc = float(np.mean(np.sign(ensemble_preds) == np.sign(y_test_aligned)))
                        logger.info(f"  Evo ensemble dir_acc={evo_dir_acc:.4f} vs standard={ens_dir_acc:.4f}")
                        # Usar evolutionary se superior
                        if evo_dir_acc > ens_dir_acc:
                            ensemble_preds = evo_preds
                            logger.info("  -> Usando Evolutionary Ensemble (superior)")
                    except Exception as e:
                        logger.warning(f"  Evolutionary Ensemble falhou: {e}")

            # --- MoE Gating Ensemble (Phase 7 — opcional) ---
            moe = None
            if self.config.training.use_moe_ensemble:
                MoECls = _safe_import("src.models.moe_ensemble", "MoEEnsemble")
                if MoECls:
                    try:
                        logger.info("  Treinando MoE Gating Ensemble...")
                        moe = MoECls(
                            n_models=len(model_names),
                            top_k=min(self.config.moe_ensemble.top_k, len(model_names)),
                        )
                        # MoE usa predictions como input (gating aprende roteamento)
                        moe.fit(val_matrix, y_val_aligned, test_matrix, y_test_aligned)
                        moe_preds = moe.predict(test_matrix)
                        moe_dir_acc = float(np.mean(np.sign(moe_preds) == np.sign(y_test_aligned)))
                        best_dir_acc = float(np.mean(np.sign(ensemble_preds) == np.sign(y_test_aligned)))
                        logger.info(f"  MoE dir_acc={moe_dir_acc:.4f} vs best={best_dir_acc:.4f}")
                        if moe_dir_acc > best_dir_acc:
                            ensemble_preds = moe_preds
                            logger.info("  -> Usando MoE Gating Ensemble (superior)")
                    except Exception as e:
                        logger.warning(f"  MoE Gating falhou: {e}")

            # Coletar residuos para Conformal Prediction
            conformal_residuals.extend(
                (y_test_aligned - ensemble_preds).tolist()
            )

            # Metricas do fold
            fold_metrics.update(
                compute_metrics(y_test_aligned, ensemble_preds, prefix="ensemble_")
            )
            for model_name, preds in fold_preds.items():
                preds_aligned = preds[-min_test_len:]
                fold_metrics.update(
                    compute_metrics(y_test_aligned, preds_aligned, prefix=f"{model_name}_")
                )

            all_fold_metrics.append(fold_metrics)
            logger.info(
                f"  Fold {fold_idx + 1} ensemble: "
                f"RMSE={fold_metrics.get('ensemble_rmse', 0):.6f}, "
                f"Dir.Acc={fold_metrics.get('ensemble_dir_accuracy', 0):.4f}"
            )

        # Modelos finais (ultimo fold)
        best_models = {
            "xgb_reg": xgb_reg,
            "lstm": lstm,
            "gru": gru,
            "ensemble": ensemble,
        }
        if xgb_cls is not None:
            best_models["xgb_cls"] = xgb_cls
        # Novos modelos opcionais
        optional_models = {
            "lgbm_reg": lgbm_reg, "rf_reg": rf_reg, "svm_reg": svm_reg,
            "cnn_lstm": cnn_lstm, "tcn": tcn, "helformer": helformer,
            "mdn": mdn, "bnn": bnn,
            "mamba": mamba, "evidential": evidential, "dual_pred": dual_pred,
            "evo_ensemble": evo_ensemble,
            "chronos": chronos, "ttm": ttm, "moirai": moirai,
            "moe": moe,
        }
        for name, model in optional_models.items():
            if model is not None:
                best_models[name] = model
        best_scaler = scaler

        # Conformal Prediction
        conformal = None
        if self.config.training.use_conformal and conformal_residuals:
            ConformalPredictor = _safe_import("src.models.conformal", "ConformalPredictor")
            if ConformalPredictor:
                conformal = ConformalPredictor(alpha=self.config.conformal.alpha)
                # Calibrar com residuos coletados de todos os folds
                residuals = np.array(conformal_residuals)
                dummy_true = np.zeros(len(residuals))
                conformal.calibrate(dummy_true, dummy_true - residuals)
                best_models["conformal"] = conformal
                logger.info(f"  Conformal Prediction calibrado com {len(residuals)} residuos")

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
            "scaler": best_scaler,
            "fold_metrics": all_fold_metrics,
            "avg_metrics": avg_metrics,
            "feature_columns": feature_columns,
        }

    def _train_emgnn(
        self, features_data: dict[str, pd.DataFrame], feature_columns: list[str],
    ) -> dict | None:
        """Treina EMGNN cross-asset se configurado.

        EMGNN opera sobre multiplos ativos simultaneamente, usando grafos de
        correlacao dinamicos em multiplas escalas temporais.

        Returns:
            Dict com modelo, metricas e coins, ou None se falhar.
        """
        if not self.config.training.use_emgnn:
            return None

        EMGNNModel = _safe_import("src.models.graph_model", "EMGNNModel")
        if EMGNNModel is None:
            return None

        try:
            coins = sorted(features_data.keys())
            if len(coins) < 2:
                logger.info("EMGNN requer >= 2 moedas, pulando")
                return None

            # Alinhar todas as moedas ao mesmo comprimento
            min_len = min(len(features_data[c]) for c in coins)
            if min_len < 50:
                logger.warning(f"EMGNN: dados insuficientes ({min_len} amostras)")
                return None

            # Construir tensor 3D: (timesteps, n_nodes, n_features)
            all_X = []
            all_y = []
            for coin in coins:
                df = features_data[coin].tail(min_len).reset_index(drop=True)
                X_coin = df[feature_columns].values.astype(np.float32)
                y_coin = df["target"].values.astype(np.float32)
                all_X.append(X_coin)
                all_y.append(y_coin)

            # (timesteps, n_nodes, n_features)
            X_3d = np.stack(all_X, axis=1)
            y_2d = np.stack(all_y, axis=1)  # (timesteps, n_nodes)

            # Criar sequencias para 4D: (samples, seq_len, n_nodes, features)
            lookback = min(20, min_len // 4)
            n_samples = min_len - lookback
            if n_samples < 30:
                logger.warning(f"EMGNN: amostras insuficientes apos sequenciamento ({n_samples})")
                return None

            X_4d = np.array([X_3d[i:i+lookback] for i in range(n_samples)])
            y_target = y_2d[lookback:]  # (n_samples, n_nodes)

            # Split train/val/test
            n_train = int(n_samples * 0.7)
            n_val = int(n_samples * 0.15)

            X_train = X_4d[:n_train]
            y_train = y_target[:n_train]
            X_val = X_4d[n_train:n_train+n_val]
            y_val = y_target[n_train:n_train+n_val]
            X_test = X_4d[n_train+n_val:]
            y_test = y_target[n_train+n_val:]

            logger.info(
                f"EMGNN: {len(coins)} moedas, {X_4d.shape}, "
                f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            )

            cfg = self.config.emgnn
            model = EMGNNModel(
                n_nodes=len(coins),
                hidden_size=cfg.hidden_size,
                n_scales=cfg.n_scales,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                batch_size=cfg.batch_size,
                max_epochs=cfg.max_epochs,
                early_stop_patience=cfg.early_stop_patience,
                graph_windows=cfg.scale_windows,
            )
            metrics = model.fit(X_train, y_train, X_val, y_val)

            # Avaliar no test set
            test_preds = model.predict(X_test)
            dir_acc = float(np.mean(np.sign(test_preds) == np.sign(y_test)))
            rmse = float(np.sqrt(np.mean((test_preds - y_test) ** 2)))
            metrics["test_dir_acc"] = dir_acc
            metrics["test_rmse"] = rmse
            logger.info(f"EMGNN test: dir_acc={dir_acc:.4f}, rmse={rmse:.6f}")

            return {
                "model": model,
                "coins": coins,
                "metrics": metrics,
            }
        except Exception as e:
            logger.warning(f"EMGNN cross-asset falhou: {e}")
            return None

    def train_all(self, data: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """Treina modelos para todas as moedas."""
        logger.info("Iniciando feature engineering...")
        features_data = self.feature_pipeline.transform_all(data)
        feature_columns = self.feature_pipeline.feature_columns

        results = {}
        for coin, df in features_data.items():
            if df.empty or "target" not in df.columns:
                logger.warning(f"Dados insuficientes para {coin}")
                continue
            df = df.dropna(subset=["target"])
            results[coin] = self.train_coin(coin, df, feature_columns)

        # EMGNN cross-asset (treina sobre todas as moedas simultaneamente)
        emgnn_result = self._train_emgnn(features_data, feature_columns)
        if emgnn_result is not None:
            # Adicionar modelo EMGNN ao resultado de cada moeda
            for coin in emgnn_result["coins"]:
                if coin in results:
                    results[coin]["models"]["emgnn"] = emgnn_result["model"]
            results["_emgnn_cross_asset"] = emgnn_result

        return results

    def save_models(
        self, results: dict[str, dict], base_dir: Path | None = None
    ) -> None:
        """Salva todos os modelos treinados."""
        if base_dir is None:
            base_dir = self.config.training.models_dir

        # Modelos que usam joblib (sklearn/xgboost/lightgbm/ensemble)
        JOBLIB_MODELS = {
            "conformal", "ensemble", "xgb_reg", "xgb_cls",
            "lgbm_reg", "lgbm_cls", "rf_reg", "rf_cls",
            "svm_reg", "svm_cls", "evo_ensemble",
        }

        for coin, result in results.items():
            # Resultado especial do EMGNN cross-asset
            if coin == "_emgnn_cross_asset":
                emgnn_dir = base_dir / "_emgnn"
                emgnn_dir.mkdir(parents=True, exist_ok=True)
                result["model"].save(emgnn_dir / "emgnn.pt")
                logger.info(f"EMGNN cross-asset salvo em {emgnn_dir}")
                continue

            coin_dir = base_dir / coin
            coin_dir.mkdir(parents=True, exist_ok=True)

            for model_name, model in result["models"].items():
                if model_name in JOBLIB_MODELS:
                    model.save(coin_dir / f"{model_name}.joblib")
                else:
                    model.save(coin_dir / f"{model_name}.pt")

            # Salvar scaler
            joblib.dump(result["scaler"], coin_dir / "scaler.joblib")

            # Salvar feature columns
            with open(coin_dir / "feature_columns.json", "w") as f:
                json.dump(result["feature_columns"], f)

            logger.info(f"Modelos salvos para {coin} em {coin_dir}")
