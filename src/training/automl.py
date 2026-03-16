"""Pipeline AutoML: selecao automatica, tuning e ensemble de modelos."""

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config.settings import Config, config as default_config
from src.data.preprocessor import FeatureScaler
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

# Mapeamento nome -> (modulo, classe, eh_sequencial)
_MODEL_REGISTRY: dict[str, tuple[str, str, bool]] = {
    "xgboost": ("src.models.xgboost_model", "XGBoostRegressor", False),
    "lightgbm": ("src.models.lightgbm_model", "LightGBMRegressor", False),
    "random_forest": ("src.models.random_forest", "RandomForestRegressorModel", False),
    "svm": ("src.models.svm_model", "SVMRegressor", False),
    "lstm": ("src.models.lstm_gru", "SequenceModel", True),
    "gru": ("src.models.lstm_gru", "SequenceModel", True),
    "cnn_lstm": ("src.models.cnn_lstm", "CNNLSTMModel", True),
    "tcn": ("src.models.tcn", "TCNModel", True),
}

# Espacos de busca Optuna para cada modelo
_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "xgboost": {
        "max_depth": ("int", 3, 10),
        "n_estimators": ("int", 100, 1000, 50),
        "learning_rate": ("float_log", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "reg_alpha": ("float", 0.0, 2.0),
        "reg_lambda": ("float", 0.0, 2.0),
    },
    "lightgbm": {
        "n_estimators": ("int", 100, 1000, 50),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("float_log", 0.01, 0.3),
        "num_leaves": ("int", 15, 63),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "reg_alpha": ("float", 0.0, 2.0),
        "reg_lambda": ("float", 0.0, 2.0),
    },
    "random_forest": {
        "n_estimators": ("int", 100, 1000, 50),
        "max_depth": ("int", 5, 20),
        "min_samples_leaf": ("int", 2, 20),
    },
    "svm": {
        "C": ("float_log", 0.01, 100.0),
        "epsilon": ("float_log", 0.001, 0.5),
    },
    "lstm": {
        "hidden_size": ("int", 32, 256, 16),
        "dropout": ("float", 0.1, 0.5),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    "gru": {
        "hidden_size": ("int", 32, 256, 16),
        "dropout": ("float", 0.1, 0.5),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    "cnn_lstm": {
        "dropout": ("float", 0.1, 0.5),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "batch_size": ("categorical", [32, 64, 128]),
    },
    "tcn": {
        "dropout": ("float", 0.1, 0.5),
        "learning_rate": ("float_log", 1e-4, 1e-2),
        "kernel_size": ("categorical", [2, 3, 5, 7]),
        "batch_size": ("categorical", [32, 64, 128]),
    },
}

# Peso relativo: RMSE vs acuracia direcional no score combinado
_RMSE_WEIGHT = 0.6
_DIR_ACC_WEIGHT = 0.4


def _safe_import(module_path: str, class_name: str):
    """Importa uma classe de forma segura, retornando None se indisponivel."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        logger.debug("Classe %s nao disponivel: %s", class_name, exc)
        return None


def _suggest_param(trial: optuna.Trial, name: str, spec: tuple) -> Any:
    """Sugere um hiperparametro ao Optuna conforme a especificacao."""
    kind = spec[0]
    if kind == "int":
        step = spec[3] if len(spec) > 3 else 1
        return trial.suggest_int(name, spec[1], spec[2], step=step)
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Tipo de parametro desconhecido: {kind}")


class AutoMLPipeline:
    """Pipeline AutoML: selecao, tuning e ensemble automaticos de modelos.

    Fluxo completo:
        1. Avaliar todos os modelos disponiveis nos dados recentes
        2. Selecionar os top-K modelos por score combinado (RMSE + dir_acc)
        3. Tunar hiperparametros dos top-3 com Optuna
        4. Treinar ensemble final com os modelos tunados
    """

    def __init__(self, config: Config = default_config):
        self.config = config

    # ------------------------------------------------------------------
    # Mapeamento nome -> classe do modelo
    # ------------------------------------------------------------------

    def _get_model_class(self, name: str):
        """Retorna a classe do modelo correspondente ao nome.

        Para modelos de sequencia (lstm/gru), retorna SequenceModel;
        para os demais, retorna a classe especifica do registro.
        """
        if name not in _MODEL_REGISTRY:
            logger.warning("Modelo '%s' nao encontrado no registro", name)
            return None

        module_path, class_name, _is_seq = _MODEL_REGISTRY[name]
        return _safe_import(module_path, class_name)

    # ------------------------------------------------------------------
    # Instanciacao de modelo com parametros opcionais
    # ------------------------------------------------------------------

    def _instantiate_model(self, name: str, **kwargs):
        """Cria uma instancia do modelo, passando kwargs adequados."""
        cls = self._get_model_class(name)
        if cls is None:
            return None

        try:
            if name == "lstm":
                return cls(cell_type="LSTM", lstm_config=self.config.lstm, **kwargs)
            if name == "gru":
                return cls(cell_type="GRU", lstm_config=self.config.lstm, **kwargs)
            if name == "xgboost":
                return cls(xgb_config=self.config.xgboost, **kwargs)
            # Modelos que aceitam kwargs diretamente
            return cls(**kwargs)
        except Exception as exc:
            logger.warning("Falha ao instanciar '%s': %s", name, exc)
            return None

    # ------------------------------------------------------------------
    # Avaliacao padrao de um modelo
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """Treina e avalia um modelo, retornando RMSE, MAE e acuracia direcional.

        Returns:
            Dicionario com chaves 'rmse', 'mae', 'directional_accuracy' e 'score'.
        """
        try:
            model.fit(X_train, y_train, X_val, y_val)
            preds = model.predict(X_val)

            # Alinhar tamanhos (modelos de sequencia podem retornar menos amostras)
            min_len = min(len(preds), len(y_val))
            preds = preds[:min_len]
            y_aligned = y_val[:min_len]

            metrics = compute_metrics(y_aligned, preds)

            rmse = metrics.get("rmse", float("inf"))
            mae = metrics.get("mae", float("inf"))
            dir_acc = metrics.get("dir_accuracy", 0.0)

            # Score combinado (menor = melhor)
            score = _RMSE_WEIGHT * rmse + _DIR_ACC_WEIGHT * (1.0 - dir_acc)

            return {
                "rmse": rmse,
                "mae": mae,
                "directional_accuracy": dir_acc,
                "score": score,
            }
        except Exception as exc:
            logger.warning("Avaliacao falhou: %s", exc)
            return {
                "rmse": float("inf"),
                "mae": float("inf"),
                "directional_accuracy": 0.0,
                "score": float("inf"),
            }

    # ------------------------------------------------------------------
    # Selecao dos melhores modelos
    # ------------------------------------------------------------------

    def select_best_models(
        self,
        coin: str,
        df: pd.DataFrame,
        top_k: int = 5,
    ) -> list[str]:
        """Avalia todos os modelos disponiveis e retorna os top-K nomes.

        Divide o DataFrame em treino/validacao (80/20 mais recente) e
        treina cada modelo uma vez, ranqueando por score combinado
        (RMSE ponderado + acuracia direcional invertida).

        Args:
            coin: Simbolo da moeda (para logging)
            df: DataFrame com features + coluna 'target'
            top_k: Numero de modelos a retornar

        Returns:
            Lista com os nomes dos top-K modelos, ordenados do melhor ao pior.
        """
        logger.info("AutoML: selecionando melhores modelos para %s...", coin)

        # Preparar dados (sem shuffling — serie temporal)
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["target"].values.astype(np.float32)

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Normalizar features
        scaler = FeatureScaler()
        df_train_temp = pd.DataFrame(X_train, columns=feature_cols)
        scaler.fit(df_train_temp, feature_cols)
        X_train_s = scaler.scaler.transform(X_train)
        X_val_s = scaler.scaler.transform(X_val)

        # Avaliar cada modelo
        resultados: list[tuple[str, float]] = []

        for name in _MODEL_REGISTRY:
            logger.info("  Avaliando %s...", name)
            model = self._instantiate_model(name)
            if model is None:
                logger.info("    %s indisponivel, pulando", name)
                continue

            metrics = self._evaluate_model(model, X_train_s, y_train, X_val_s, y_val)
            resultados.append((name, metrics["score"]))
            logger.info(
                "    %s -> RMSE=%.6f, DirAcc=%.4f, Score=%.6f",
                name,
                metrics["rmse"],
                metrics["directional_accuracy"],
                metrics["score"],
            )

        # Ordenar por score (menor = melhor)
        resultados.sort(key=lambda x: x[1])
        melhores = [nome for nome, _ in resultados[:top_k]]

        logger.info("AutoML: top-%d modelos para %s: %s", top_k, coin, melhores)
        return melhores

    # ------------------------------------------------------------------
    # Tuning de hiperparametros com Optuna
    # ------------------------------------------------------------------

    def auto_tune(
        self,
        coin: str,
        df: pd.DataFrame,
        model_name: str,
        n_trials: int = 30,
    ) -> dict:
        """Otimiza hiperparametros de um modelo especifico via Optuna.

        Usa TPE sampler com MedianPruner. O score objetivo combina RMSE
        e acuracia direcional.

        Args:
            coin: Simbolo da moeda (para logging)
            df: DataFrame com features + coluna 'target'
            model_name: Nome do modelo (chave em _MODEL_REGISTRY)
            n_trials: Numero de trials do Optuna

        Returns:
            Dicionario com 'best_params', 'best_score', 'best_rmse', 'best_dir_acc'.
        """
        logger.info(
            "AutoML: tunando %s para %s (%d trials)...", model_name, coin, n_trials
        )

        if model_name not in _SEARCH_SPACES:
            logger.warning("Espaco de busca nao definido para '%s'", model_name)
            return {"best_params": {}, "best_score": float("inf")}

        # Preparar dados
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["target"].values.astype(np.float32)

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        scaler = FeatureScaler()
        df_train_temp = pd.DataFrame(X_train, columns=feature_cols)
        scaler.fit(df_train_temp, feature_cols)
        X_train_s = scaler.scaler.transform(X_train)
        X_val_s = scaler.scaler.transform(X_val)

        search_space = _SEARCH_SPACES[model_name]

        def _objective(trial: optuna.Trial) -> float:
            """Funcao objetivo: treina modelo com params sugeridos e retorna score."""
            params = {}
            for param_name, spec in search_space.items():
                params[param_name] = _suggest_param(trial, param_name, spec)

            # Instanciar modelo com os params sugeridos
            model = self._instantiate_model(model_name, **params)
            if model is None:
                return float("inf")

            metrics = self._evaluate_model(model, X_train_s, y_train, X_val_s, y_val)

            trial.set_user_attr("rmse", metrics["rmse"])
            trial.set_user_attr("dir_acc", metrics["directional_accuracy"])

            return metrics["score"]

        # Criar e executar estudo Optuna
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            study_name=f"automl_{model_name}_{coin}",
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=self.config.hyperopt.timeout,
            show_progress_bar=False,
        )

        best = study.best_trial
        logger.info(
            "AutoML tuning %s concluido: %d trials, score=%.6f "
            "(RMSE=%.6f, dir_acc=%.4f)",
            model_name,
            len(study.trials),
            best.value,
            best.user_attrs.get("rmse", float("inf")),
            best.user_attrs.get("dir_acc", 0.0),
        )
        logger.info("  Melhores params: %s", best.params)

        return {
            "best_params": best.params,
            "best_score": best.value,
            "best_rmse": best.user_attrs.get("rmse", float("inf")),
            "best_dir_acc": best.user_attrs.get("dir_acc", 0.0),
        }

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------

    def run_auto_pipeline(self, coin: str, df: pd.DataFrame) -> dict:
        """Executa pipeline AutoML completo para uma moeda.

        Fluxo:
            1. Selecionar os 5 melhores modelos
            2. Tunar os top-3 com Optuna
            3. Treinar ensemble com os modelos tunados
            4. Retornar resultados consolidados

        Args:
            coin: Simbolo da moeda
            df: DataFrame com features + coluna 'target'

        Returns:
            Dicionario com 'selected_models', 'tuning_results',
            'ensemble_metrics' e 'trained_models'.
        """
        logger.info("\n" + "=" * 60)
        logger.info("AutoML Pipeline para %s", coin)
        logger.info("=" * 60)

        # Etapa 1: Selecionar melhores modelos
        melhores = self.select_best_models(coin, df, top_k=5)

        if not melhores:
            logger.error("Nenhum modelo disponivel para %s", coin)
            return {"selected_models": [], "error": "Nenhum modelo disponivel"}

        # Etapa 2: Tunar os top-3
        top_para_tunar = melhores[:3]
        tuning_results = {}
        for name in top_para_tunar:
            tuning_results[name] = self.auto_tune(coin, df, name, n_trials=30)

        # Etapa 3: Treinar modelos tunados e montar ensemble
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["target"].values.astype(np.float32)

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        scaler = FeatureScaler()
        df_train_temp = pd.DataFrame(X_train, columns=feature_cols)
        scaler.fit(df_train_temp, feature_cols)
        X_train_s = scaler.scaler.transform(X_train)
        X_val_s = scaler.scaler.transform(X_val)

        # Treinar cada modelo tunado e coletar previsoes de validacao
        modelos_treinados = {}
        val_preds = {}

        for name in top_para_tunar:
            best_params = tuning_results[name].get("best_params", {})
            model = self._instantiate_model(name, **best_params)
            if model is None:
                continue

            try:
                model.fit(X_train_s, y_train, X_val_s, y_val)
                preds = model.predict(X_val_s)
                min_len = min(len(preds), len(y_val))
                val_preds[name] = preds[:min_len]
                modelos_treinados[name] = model
                logger.info("  Modelo tunado %s treinado com sucesso", name)
            except Exception as exc:
                logger.warning("  Falha ao treinar %s tunado: %s", name, exc)

        # Ensemble simples por media ponderada (pesos = 1/score)
        ensemble_metrics = {}
        if val_preds:
            min_len = min(len(v) for v in val_preds.values())
            y_val_aligned = y_val[:min_len]

            # Calcular pesos inversamente proporcionais ao score
            pesos = {}
            for name in val_preds:
                score = tuning_results.get(name, {}).get("best_score", 1.0)
                # Evitar divisao por zero
                pesos[name] = 1.0 / max(score, 1e-8)

            soma_pesos = sum(pesos.values())
            pesos_norm = {k: v / soma_pesos for k, v in pesos.items()}

            # Media ponderada das previsoes
            ensemble_pred = np.zeros(min_len)
            for name, pred in val_preds.items():
                ensemble_pred += pesos_norm[name] * pred[:min_len]

            ensemble_metrics = compute_metrics(y_val_aligned, ensemble_pred, prefix="ensemble_")
            logger.info(
                "Ensemble AutoML: RMSE=%.6f, DirAcc=%.4f",
                ensemble_metrics.get("ensemble_rmse", float("inf")),
                ensemble_metrics.get("ensemble_dir_accuracy", 0.0),
            )

        return {
            "selected_models": melhores,
            "tuning_results": tuning_results,
            "ensemble_metrics": ensemble_metrics,
            "trained_models": modelos_treinados,
            "ensemble_weights": pesos_norm if val_preds else {},
            "scaler": scaler,
            "feature_columns": feature_cols,
        }
