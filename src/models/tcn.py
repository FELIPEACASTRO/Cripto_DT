"""Temporal Convolutional Network (TCN) para previsao de criptomoedas.

Implementa convolucoes causais dilatadas empilhadas, permitindo que o modelo
capture dependencias temporais de longo prazo sem recorrencia. A causalidade
garante que previsoes dependam apenas de dados passados.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset

from config.settings import FeatureConfig, config as default_config
from src.models.base import BaseModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Chomp1d(nn.Module):
    """Remove o padding extra do final da sequencia para garantir causalidade.

    Apos a convolucao com padding = (kernel_size - 1) * dilation,
    a saida tem elementos extras no final que 'vazam' informacao futura.
    Chomp1d corta esses elementos.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len + chomp_size) -> (batch, channels, seq_len)"""
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Bloco temporal: duas camadas de convolucao causal dilatada com residual.

    Cada camada: Conv1d (causal) -> Chomp1d -> ReLU -> Dropout
    Com conexao residual (1x1 conv se in_channels != out_channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Conexao residual
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializacao normal para convolucoes."""
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len)"""
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TemporalConvNet(nn.Module):
    """Stack de TemporalBlocks com dilation crescente.

    Dilations padrao: [1, 2, 4, 8, 16] -> campo receptivo exponencial.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2**i for i in range(len(num_channels))]

        layers = []
        for i, (out_ch, dil) in enumerate(zip(num_channels, dilations)):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dil, dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_size, seq_len) -> (batch, num_channels[-1], seq_len)"""
        return self.network(x)


class TCNNet(nn.Module):
    """Rede TCN completa: TemporalConvNet -> Linear para previsao de retorno.

    Input: (batch, seq_len, input_size)
    Output: (batch,)
    """

    def __init__(
        self,
        input_size: int,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch,)"""
        # Conv1d espera (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)  # (batch, num_channels[-1], seq_len)
        # Pegar ultimo timestep
        out = out[:, :, -1]  # (batch, num_channels[-1])
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


class TCNModel(BaseModel):
    """Wrapper de treino/inferencia para a TCN."""

    def __init__(
        self,
        input_size: int | None = None,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
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
        self.input_size = input_size
        self.num_channels = num_channels or [64, 64, 64, 64]
        self.kernel_size = kernel_size
        self.dropout = dropout
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
        self.model: TCNNet | None = None

    @property
    def name(self) -> str:
        return "tcn"

    def _build_model(self, input_size: int) -> TCNNet:
        self.input_size = input_size
        return TCNNet(
            input_size=input_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        ).to(DEVICE)

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Cria sequencias de janela deslizante para a TCN.

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
        X_train_seq, y_train_seq = self._make_sequences(X_train, y_train)
        if X_train_seq is None or y_train_seq is None or len(X_train_seq) == 0:
            logger.warning("Dados insuficientes para treino da TCN")
            return {}

        input_size = X_train_seq.shape[2]
        self.model = self._build_model(input_size)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"TCN inicializada: {total_params:,} parametros, device={DEVICE}")

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
            # Treino
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

            # Validacao
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
                    logger.info(f"  TCN early stopping na epoca {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                msg = f"  TCN epoca {epoch + 1}: train_loss={avg_train_loss:.6f}"
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

        logger.info(f"  TCN metricas: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo TCN nao treinado")
        self.model.eval()
        X_seq, _ = self._make_sequences(X)
        with torch.no_grad():
            return self.model(X_seq).cpu().numpy()

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC Dropout: roda N forward passes com dropout ativado."""
        if self.model is None:
            raise RuntimeError("Modelo TCN nao treinado")

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
            raise RuntimeError("Modelo TCN nao treinado")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_size": self.input_size,
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
            },
            path,
        )
        logger.info(f"TCN salva: {path}")

    @classmethod
    def load(cls, path: Path) -> "TCNModel":
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        instance = cls(
            input_size=checkpoint["input_size"],
            num_channels=checkpoint["num_channels"],
            kernel_size=checkpoint["kernel_size"],
            dropout=checkpoint["dropout"],
        )
        instance.model = TCNNet(
            input_size=checkpoint["input_size"],
            num_channels=checkpoint["num_channels"],
            kernel_size=checkpoint["kernel_size"],
            dropout=checkpoint["dropout"],
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state"])
        return instance
