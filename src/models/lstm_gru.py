"""Modelos LSTM e GRU em PyTorch com MC Dropout para estimativa de incerteza."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import LSTMConfig, FeatureConfig, config as default_config
from src.models.base import BaseModel
from src.models.losses import MADLLoss

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceNet(nn.Module):
    """Rede neural recorrente generica (LSTM ou GRU)."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        dropout: float,
        cell_type: str = "LSTM",
    ):
        super().__init__()
        self.cell_type = cell_type

        rnn_class = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=len(hidden_sizes),
            dropout=dropout if len(hidden_sizes) > 1 else 0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[-1])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        rnn_out, _ = self.rnn(x)
        # Pegar apenas o ultimo timestep
        last_hidden = rnn_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1)


class SequenceModel(BaseModel):
    """Wrapper de treino/inferencia para LSTM ou GRU."""

    def __init__(
        self,
        cell_type: str = "LSTM",
        input_size: int | None = None,
        lstm_config: LSTMConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        self.cell_type = cell_type
        self.cfg = lstm_config or default_config.lstm
        self.feat_cfg = feature_config or default_config.features
        self.input_size = input_size
        self.model: SequenceNet | None = None

    @property
    def name(self) -> str:
        return self.cell_type.lower()

    def _build_model(self, input_size: int) -> SequenceNet:
        self.input_size = input_size
        return SequenceNet(
            input_size=input_size,
            hidden_sizes=self.cfg.hidden_sizes,
            dropout=self.cfg.dropout,
            cell_type=self.cell_type,
        ).to(DEVICE)

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        """Cria sequencias de lookback_window para o RNN.

        Se X ja tem 3 dimensoes (batch, seq, features), retorna como esta.
        Se X tem 2 dimensoes (samples, features), cria janelas deslizantes.
        """
        if X.ndim == 3:
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            if y is not None:
                y_tensor = torch.FloatTensor(y).to(DEVICE)
                return X_tensor, y_tensor
            return X_tensor, None

        window = self.feat_cfg.lookback_window
        sequences = []
        targets = []
        for i in range(window, len(X)):
            sequences.append(X[i - window : i])
            if y is not None:
                targets.append(y[i])

        X_tensor = torch.FloatTensor(np.array(sequences)).to(DEVICE)
        y_tensor = torch.FloatTensor(np.array(targets)).to(DEVICE) if y is not None else None
        return X_tensor, y_tensor

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        X_train_seq, y_train_seq = self._make_sequences(X_train, y_train)
        if X_train_seq is None or y_train_seq is None or len(X_train_seq) == 0:
            logger.warning("Dados insuficientes para treino")
            return {}

        input_size = X_train_seq.shape[2]
        self.model = self._build_model(input_size)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.cfg.scheduler_patience,
            factor=self.cfg.scheduler_factor,
        )
        # Selecionar loss function
        if self.cfg.loss_type == "madl":
            criterion = MADLLoss(alpha=self.cfg.madl_alpha)
        elif self.cfg.loss_type == "directional_huber":
            from src.models.losses import DirectionalHuberLoss
            criterion = DirectionalHuberLoss()
        else:
            criterion = nn.HuberLoss()

        train_dataset = TensorDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True
        )

        # Validation
        X_val_seq, y_val_seq = (None, None)
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._make_sequences(X_val, y_val)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.cfg.max_epochs):
            # Train
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_max_norm
                )
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validate
            if X_val_seq is not None and y_val_seq is not None and len(X_val_seq) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val_seq)
                    val_loss = criterion(val_preds, y_val_seq).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.cfg.early_stop_patience:
                    logger.info(f"  Early stopping na epoca {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                msg = f"  Epoca {epoch + 1}: train_loss={avg_train_loss:.6f}"
                if X_val_seq is not None:
                    msg += f" val_loss={val_loss:.6f}"
                logger.info(msg)

        # Restaurar melhor modelo
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Metricas finais
        metrics = {"train_loss": float(avg_train_loss)}
        if X_val_seq is not None and y_val_seq is not None and len(X_val_seq) > 0:
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_seq).cpu().numpy()
                y_val_np = y_val_seq.cpu().numpy()
            metrics["val_loss"] = float(best_val_loss)
            metrics["val_rmse"] = float(np.sqrt(np.mean((val_preds - y_val_np) ** 2)))
            metrics["val_dir_acc"] = float(np.mean(np.sign(val_preds) == np.sign(y_val_np)))

        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")
        self.model.eval()
        X_seq, _ = self._make_sequences(X)
        with torch.no_grad():
            return self.model(X_seq).cpu().numpy()

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC Dropout: roda N forward passes com dropout ativado."""
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")

        X_seq, _ = self._make_sequences(X)
        self.model.train()  # Ativa dropout

        predictions = []
        for _ in range(self.cfg.mc_dropout_samples):
            with torch.no_grad():
                pred = self.model(X_seq).cpu().numpy()
                predictions.append(pred)

        self.model.eval()
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        # Confianca inversamente proporcional a incerteza
        max_std = std_pred.max() if std_pred.max() > 0 else 1.0
        confidence = 1.0 - np.clip(std_pred / max_std, 0, 1)

        return mean_pred, confidence

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "cell_type": self.cell_type,
            "input_size": self.input_size,
            "hidden_sizes": self.cfg.hidden_sizes,
            "dropout": self.cfg.dropout,
        }, path)
        logger.info(f"Modelo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "SequenceModel":
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        instance = cls(
            cell_type=checkpoint["cell_type"],
            input_size=checkpoint["input_size"],
        )
        instance.model = SequenceNet(
            input_size=checkpoint["input_size"],
            hidden_sizes=checkpoint["hidden_sizes"],
            dropout=checkpoint["dropout"],
            cell_type=checkpoint["cell_type"],
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state"])
        return instance
