"""Otimizacao de hiperparametros com Optuna para modelos XGBoost e LSTM."""

import logging
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config.settings import Config, config as default_config

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Otimiza hiperparametros dos modelos usando Optuna (TPE + MedianPruner).

    A funcao objetivo combina RMSE de validacao (menor = melhor) com
    acuracia direcional (maior = melhor) em um score unico ponderado.
    """

    # Peso relativo: RMSE vs acuracia direcional no score combinado
    _RMSE_WEIGHT = 0.6
    _DIR_ACC_WEIGHT = 0.4

    def __init__(
        self,
        config: Config = default_config,
        n_trials: int = 50,
        timeout: int = 3600,
    ):
        self.config = config
        self.n_trials = n_trials
        self.timeout = timeout

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    def optimize_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Otimiza hiperparametros do XGBoost via Optuna.

        Espaco de busca:
            max_depth:        [3, 10]
            n_estimators:     [100, 1000]
            learning_rate:    [0.01, 0.3]  (log-uniform)
            subsample:        [0.6, 1.0]
            colsample_bytree: [0.6, 1.0]
            reg_alpha:        [0, 2]
            reg_lambda:       [0, 2]

        Returns:
            Dicionario de params compativel com ``XGBoostConfig``.
        """
        import xgboost as xgb

        def _objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }

            model = xgb.XGBRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_pred = model.predict(X_val)
            rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
            dir_acc = float(np.mean(np.sign(val_pred) == np.sign(y_val)))

            # Score combinado: minimizar RMSE, maximizar dir_acc
            # Normalizamos dir_acc invertendo (1 - dir_acc) para que ambos
            # os termos sejam "menor = melhor".
            score = (
                self._RMSE_WEIGHT * rmse
                + self._DIR_ACC_WEIGHT * (1.0 - dir_acc)
            )

            trial.set_user_attr("rmse", rmse)
            trial.set_user_attr("dir_acc", dir_acc)

            return score

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            study_name="xgboost_hyperopt",
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            _objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        best = study.best_trial
        logger.info(
            "XGBoost hyperopt concluido: %d trials, "
            "melhor score=%.6f (RMSE=%.6f, dir_acc=%.4f)",
            len(study.trials),
            best.value,
            best.user_attrs["rmse"],
            best.user_attrs["dir_acc"],
        )
        logger.info("  Melhores parametros: %s", best.params)

        return best.params

    # ------------------------------------------------------------------
    # LSTM
    # ------------------------------------------------------------------

    def optimize_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Otimiza hiperparametros do LSTM via Optuna.

        Espaco de busca:
            hidden_size:     [32, 256]
            dropout:         [0.1, 0.5]
            learning_rate:   [1e-4, 1e-2]  (log-uniform)
            batch_size:      {32, 64, 128}
            lookback_window: [20, 120]

        Returns:
            Dicionario de params compativel com ``LSTMConfig`` / ``FeatureConfig``.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _build_sequences(
            X: np.ndarray, y: np.ndarray, window: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Cria janelas deslizantes para o RNN."""
            if X.ndim == 3:
                return (
                    torch.FloatTensor(X).to(device),
                    torch.FloatTensor(y).to(device),
                )
            sequences, targets = [], []
            for i in range(window, len(X)):
                sequences.append(X[i - window : i])
                targets.append(y[i])
            if not sequences:
                return torch.empty(0), torch.empty(0)
            return (
                torch.FloatTensor(np.array(sequences)).to(device),
                torch.FloatTensor(np.array(targets)).to(device),
            )

        def _objective(trial: optuna.Trial) -> float:
            hidden_size = trial.suggest_int("hidden_size", 32, 256, step=16)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            lookback_window = trial.suggest_int("lookback_window", 20, 120, step=5)

            # Construir sequencias com o lookback_window proposto
            X_tr_seq, y_tr_seq = _build_sequences(X_train, y_train, lookback_window)
            X_va_seq, y_va_seq = _build_sequences(X_val, y_val, lookback_window)

            if len(X_tr_seq) == 0 or len(X_va_seq) == 0:
                return float("inf")

            input_size = X_tr_seq.shape[2]

            # Rede simples de 2 camadas
            rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=dropout,
                batch_first=True,
            ).to(device)
            fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            ).to(device)

            params_iter = list(rnn.parameters()) + list(fc.parameters())
            optimizer = torch.optim.AdamW(params_iter, lr=lr, weight_decay=1e-4)
            criterion = nn.HuberLoss()

            dataset = TensorDataset(X_tr_seq, y_tr_seq)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Treinar por ate 30 epocas com pruning
            max_epochs = 30
            best_val_loss = float("inf")
            patience, patience_limit = 0, 7

            for epoch in range(max_epochs):
                rnn.train()
                fc.train()
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    out, _ = rnn(X_batch)
                    pred = fc(out[:, -1, :]).squeeze(-1)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(params_iter, 1.0)
                    optimizer.step()

                # Validacao
                rnn.eval()
                fc.eval()
                with torch.no_grad():
                    v_out, _ = rnn(X_va_seq)
                    v_pred = fc(v_out[:, -1, :]).squeeze(-1)
                    val_loss = criterion(v_pred, y_va_seq).item()

                # Optuna pruning
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    best_v_pred = v_pred.cpu().numpy()
                else:
                    patience += 1
                    if patience >= patience_limit:
                        break

            # Metricas finais
            y_val_np = y_va_seq.cpu().numpy()
            if best_v_pred is None or len(best_v_pred) == 0:
                return float("inf")

            rmse = float(np.sqrt(np.mean((best_v_pred - y_val_np) ** 2)))
            dir_acc = float(np.mean(np.sign(best_v_pred) == np.sign(y_val_np)))

            score = (
                self._RMSE_WEIGHT * rmse
                + self._DIR_ACC_WEIGHT * (1.0 - dir_acc)
            )

            trial.set_user_attr("rmse", rmse)
            trial.set_user_attr("dir_acc", dir_acc)

            return score

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
            study_name="lstm_hyperopt",
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            _objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        best = study.best_trial
        logger.info(
            "LSTM hyperopt concluido: %d trials, "
            "melhor score=%.6f (RMSE=%.6f, dir_acc=%.4f)",
            len(study.trials),
            best.value,
            best.user_attrs["rmse"],
            best.user_attrs["dir_acc"],
        )
        logger.info("  Melhores parametros: %s", best.params)

        return best.params

    # ------------------------------------------------------------------
    # Otimizar todos
    # ------------------------------------------------------------------

    def optimize_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Otimiza hiperparametros de XGBoost e LSTM.

        Returns:
            Dicionario com chaves ``"xgboost"`` e ``"lstm"``, cada uma
            contendo os melhores hiperparametros encontrados.
        """
        logger.info("Iniciando otimizacao de hiperparametros (XGBoost + LSTM)...")

        xgb_params = self.optimize_xgboost(X_train, y_train, X_val, y_val)
        lstm_params = self.optimize_lstm(X_train, y_train, X_val, y_val)

        return {
            "xgboost": xgb_params,
            "lstm": lstm_params,
        }
