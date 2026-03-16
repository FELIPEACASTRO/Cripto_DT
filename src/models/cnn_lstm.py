"""Modelo hibrido 1D-CNN + LSTM para previsao de criptomoedas.

Baseado em estudo da Hue University Vietnam, combina camadas convolucionais
para extracao de padroes locais com LSTM para captura de dependencias temporais
de longo prazo.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import FeatureConfig, config as default_config
from src.models.base import BaseModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNLSTMNet(nn.Module):
    """Rede hibrida 1D-CNN + LSTM.

    Pipeline:
        Input (batch, seq_len, n_features)
        -> Conv1d + BatchNorm + ReLU (64 filtros)
        -> Conv1d + BatchNorm + ReLU (128 filtros)
        -> MaxPool1d(2)
        -> LSTM (2 camadas, hidden=64)
        -> Linear -> Output (retorno previsto)
    """

    def __init__(
        self,
        n_features: int,
        conv1_out: int = 64,
        conv2_out: int = 128,
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features

        # Bloco CNN 1 - extracao de padroes locais
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.relu1 = nn.ReLU()

        # Bloco CNN 2 - padroes mais complexos
        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(conv2_out)
        self.relu2 = nn.ReLU()

        # Pooling para reduzir dimensao temporal
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM para captura de dependencias temporais
        self.lstm = nn.LSTM(
            input_size=conv2_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        # Dropout para regularizacao (usado em MC Dropout)
        self.dropout = nn.Dropout(lstm_dropout)

        # Camada de saida
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            predictions: (batch,)
        """
        # Conv1d espera (batch, channels, seq_len) -> permutar
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)

        # Blocos convolucionais
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        # Pooling
        x = self.pool(x)  # (batch, conv2_out, seq_len // 2)

        # Permutar de volta para LSTM: (batch, seq_len // 2, conv2_out)
        x = x.permute(0, 2, 1)

        # LSTM - usar ultimo hidden state
        lstm_out, (h_n, _) = self.lstm(x)
        # Pegar saida do ultimo timestep
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        # Dropout + FC
        out = self.dropout(last_hidden)
        out = self.fc(out)

        return out.squeeze(-1)


class CNNLSTMModel(BaseModel):
    """Wrapper de treino/inferencia para o modelo CNN-LSTM."""

    def __init__(
        self,
        n_features: int | None = None,
        conv1_out: int = 64,
        conv2_out: int = 128,
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 100,
        early_stop_patience: int = 10,
        grad_clip_max_norm: float = 1.0,
        mc_dropout_samples: int = 30,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        feature_config: FeatureConfig | None = None,
    ):
        self.n_features = n_features
        self.conv1_out = conv1_out
        self.conv2_out = conv2_out
        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.grad_clip_max_norm = grad_clip_max_norm
        self.mc_dropout_samples = mc_dropout_samples
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.feat_cfg = feature_config or default_config.features
        self.model: CNNLSTMNet | None = None

    @property
    def name(self) -> str:
        return "cnn_lstm"

    def _build_model(self, n_features: int) -> CNNLSTMNet:
        """Constroi a rede CNN-LSTM com os parametros configurados."""
        self.n_features = n_features
        return CNNLSTMNet(
            n_features=n_features,
            conv1_out=self.conv1_out,
            conv2_out=self.conv2_out,
            kernel_size=self.kernel_size,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            lstm_dropout=self.lstm_dropout,
        ).to(DEVICE)

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Cria sequencias de janela deslizante para a rede.

        Se X ja tem 3 dimensoes (batch, seq, features), retorna como esta.
        Se X tem 2 dimensoes (samples, features), cria janelas deslizantes.
        """
        if X.ndim == 3:
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            y_tensor = torch.FloatTensor(y).to(DEVICE) if y is not None else None
            return X_tensor, y_tensor

        window = self.feat_cfg.lookback_window
        sequences = []
        targets = []
        for i in range(window, len(X)):
            sequences.append(X[i - window : i])
            if y is not None:
                targets.append(y[i])

        X_tensor = torch.FloatTensor(np.array(sequences)).to(DEVICE)
        y_tensor = (
            torch.FloatTensor(np.array(targets)).to(DEVICE) if y is not None else None
        )
        return X_tensor, y_tensor

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        # Criar sequencias temporais
        X_train_seq, y_train_seq = self._make_sequences(X_train, y_train)
        if X_train_seq is None or y_train_seq is None or len(X_train_seq) == 0:
            logger.warning("Dados insuficientes para treino do CNN-LSTM")
            return {}

        n_features = X_train_seq.shape[2]
        self.model = self._build_model(n_features)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"CNN-LSTM inicializado: {total_params:,} parametros, device={DEVICE}")

        # Otimizador e scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.scheduler_patience,
            factor=self.scheduler_factor,
        )
        criterion = nn.HuberLoss()

        # DataLoader de treino
        train_dataset = TensorDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Validacao
        X_val_seq, y_val_seq = (None, None)
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._make_sequences(X_val, y_val)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # Fase de treino
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            # Fase de validacao
            if X_val_seq is not None and y_val_seq is not None and len(X_val_seq) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val_seq)
                    val_loss = criterion(val_preds, y_val_seq).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stop_patience:
                    logger.info(f"  CNN-LSTM early stopping na epoca {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                msg = f"  CNN-LSTM epoca {epoch + 1}: train_loss={avg_train_loss:.6f}"
                if X_val_seq is not None and y_val_seq is not None and len(X_val_seq) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        _vp = self.model(X_val_seq)
                        _vl = criterion(_vp, y_val_seq).item()
                    msg += f" val_loss={_vl:.6f}"
                logger.info(msg)

        # Restaurar melhor modelo
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Metricas finais
        metrics: dict[str, float] = {"train_loss": avg_train_loss}
        if X_val_seq is not None and y_val_seq is not None and len(X_val_seq) > 0:
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_seq).cpu().numpy()
                y_val_np = y_val_seq.cpu().numpy()
            metrics["val_loss"] = float(best_val_loss)
            metrics["val_rmse"] = float(np.sqrt(np.mean((val_preds - y_val_np) ** 2)))
            metrics["val_dir_acc"] = float(
                np.mean(np.sign(val_preds) == np.sign(y_val_np))
            )

        logger.info(f"  CNN-LSTM metricas: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo CNN-LSTM nao treinado")
        self.model.eval()
        X_seq, _ = self._make_sequences(X)
        with torch.no_grad():
            return self.model(X_seq).cpu().numpy()

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC Dropout: roda N forward passes com dropout ativado para estimar incerteza."""
        if self.model is None:
            raise RuntimeError("Modelo CNN-LSTM nao treinado")

        X_seq, _ = self._make_sequences(X)
        self.model.train()  # Ativa dropout

        predictions = []
        for _ in range(self.mc_dropout_samples):
            with torch.no_grad():
                pred = self.model(X_seq).cpu().numpy()
                predictions.append(pred)

        self.model.eval()
        predictions_arr = np.array(predictions)
        mean_pred = predictions_arr.mean(axis=0)
        std_pred = predictions_arr.std(axis=0)

        # Confianca inversamente proporcional a incerteza
        max_std = std_pred.max() if std_pred.max() > 0 else 1.0
        confidence = 1.0 - np.clip(std_pred / max_std, 0, 1)

        return mean_pred, confidence

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Modelo CNN-LSTM nao treinado")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "n_features": self.n_features,
                "conv1_out": self.conv1_out,
                "conv2_out": self.conv2_out,
                "kernel_size": self.kernel_size,
                "lstm_hidden": self.lstm_hidden,
                "lstm_layers": self.lstm_layers,
                "lstm_dropout": self.lstm_dropout,
            },
            path,
        )
        logger.info(f"CNN-LSTM salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "CNNLSTMModel":
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        instance = cls(
            n_features=checkpoint["n_features"],
            conv1_out=checkpoint["conv1_out"],
            conv2_out=checkpoint["conv2_out"],
            kernel_size=checkpoint["kernel_size"],
            lstm_hidden=checkpoint["lstm_hidden"],
            lstm_layers=checkpoint["lstm_layers"],
            lstm_dropout=checkpoint["lstm_dropout"],
        )
        instance.model = CNNLSTMNet(
            n_features=checkpoint["n_features"],
            conv1_out=checkpoint["conv1_out"],
            conv2_out=checkpoint["conv2_out"],
            kernel_size=checkpoint["kernel_size"],
            lstm_hidden=checkpoint["lstm_hidden"],
            lstm_layers=checkpoint["lstm_layers"],
            lstm_dropout=checkpoint["lstm_dropout"],
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state"])
        return instance
