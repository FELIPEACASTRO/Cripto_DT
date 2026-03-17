"""Helformer: Holt-Winters Decomposition + Transformer para previsao de criptomoedas.

Combina decomposicao exponencial (nivel, tendencia, sazonalidade) com
Transformers independentes por componente, permitindo que o modelo
aprenda padroes temporais distintos para cada aspecto da serie.
"""

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import FeatureConfig, LSTMConfig, config as default_config
from src.models.base import BaseModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HoltWintersDecomposition(nn.Module):
    """Decomposicao de Holt-Winters com parametros aprendiveis.

    Decompoe cada feature da serie temporal em tres componentes:
    - Level (nivel): valor medio suavizado
    - Trend (tendencia): direcao e velocidade da mudanca
    - Seasonal (sazonalidade): padrao periodico residual
    """

    def __init__(self, input_size: int, season_period: int = 24):
        super().__init__()
        self.input_size = input_size
        self.season_period = season_period

        # Parametros de suavizacao aprendiveis (um por feature)
        # Inicializados com valores classicos e restritos a (0, 1) via sigmoid
        self.raw_alpha = nn.Parameter(torch.full((input_size,), 0.0))  # sigmoid(0) = 0.5
        self.raw_beta = nn.Parameter(torch.full((input_size,), -1.0))  # sigmoid(-1) ~ 0.27
        self.raw_gamma = nn.Parameter(torch.full((input_size,), -1.0))  # sigmoid(-1) ~ 0.27

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_beta)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gamma)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompoe a serie em nivel, tendencia e sazonalidade.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            level, trend, seasonal: cada um (batch, seq_len, input_size)
        """
        batch_size, seq_len, n_features = x.shape
        alpha = self.alpha.unsqueeze(0)  # (1, input_size)
        beta = self.beta.unsqueeze(0)
        gamma = self.gamma.unsqueeze(0)

        # Inicializacao
        level = x[:, 0, :]  # (batch, input_size)
        trend = torch.zeros_like(level)
        # Buffer sazonal: inicializado com zeros
        seasonal_buffer = torch.zeros(
            batch_size, self.season_period, n_features, device=x.device
        )

        levels = []
        trends = []
        seasonals = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            s_idx = t % self.season_period
            season_prev = seasonal_buffer[:, s_idx, :]

            # Atualizar nivel
            new_level = alpha * (x_t - season_prev) + (1 - alpha) * (level + trend)
            # Atualizar tendencia
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            # Atualizar sazonalidade
            new_seasonal = gamma * (x_t - new_level) + (1 - gamma) * season_prev

            levels.append(new_level)
            trends.append(new_trend)
            seasonals.append(new_seasonal)

            level = new_level
            trend = new_trend
            seasonal_buffer = seasonal_buffer.clone()
            seasonal_buffer[:, s_idx, :] = new_seasonal

        level_out = torch.stack(levels, dim=1)  # (batch, seq_len, input_size)
        trend_out = torch.stack(trends, dim=1)
        seasonal_out = torch.stack(seasonals, dim=1)

        return level_out, trend_out, seasonal_out


class PositionalEncoding(nn.Module):
    """Positional encoding sinusoidal padrao para Transformers."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class HelformerNet(nn.Module):
    """Rede Helformer: Holt-Winters Decomposition + Transformer.

    Pipeline:
        Input -> HW Decomposition -> (level, trend, seasonal)
        Cada componente -> Projecao linear -> Positional Encoding -> TransformerEncoder
        Concatenar saidas -> Fusion FC -> Output (retorno previsto)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 60,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.seq_len = seq_len

        # Decomposicao Holt-Winters
        self.hw_decomp = HoltWintersDecomposition(input_size)

        # Projecoes lineares: input_size -> d_model (uma por componente)
        self.proj_level = nn.Linear(input_size, d_model)
        self.proj_trend = nn.Linear(input_size, d_model)
        self.proj_seasonal = nn.Linear(input_size, d_model)

        # Positional encoding compartilhado
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # TransformerEncoder separado por componente
        encoder_layer_args = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_level = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**encoder_layer_args),
            num_layers=num_layers,
        )
        self.transformer_trend = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**encoder_layer_args),
            num_layers=num_layers,
        )
        self.transformer_seasonal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**encoder_layer_args),
            num_layers=num_layers,
        )

        # Fusion: 3 componentes concatenados -> output
        self.dropout = nn.Dropout(dropout)
        self.fc_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            predictions: (batch,)
        """
        # Decomposicao
        level, trend, seasonal = self.hw_decomp(x)

        # Projecao + Positional Encoding + Transformer por componente
        level_enc = self.pos_encoder(self.proj_level(level))
        trend_enc = self.pos_encoder(self.proj_trend(trend))
        seasonal_enc = self.pos_encoder(self.proj_seasonal(seasonal))

        level_out = self.transformer_level(level_enc)
        trend_out = self.transformer_trend(trend_enc)
        seasonal_out = self.transformer_seasonal(seasonal_enc)

        # Pegar ultimo timestep de cada componente
        level_last = level_out[:, -1, :]  # (batch, d_model)
        trend_last = trend_out[:, -1, :]
        seasonal_last = seasonal_out[:, -1, :]

        # Fusao
        fused = torch.cat([level_last, trend_last, seasonal_last], dim=-1)
        fused = self.dropout(fused)
        output = self.fc_fusion(fused)

        return output.squeeze(-1)


class HelformerModel(BaseModel):
    """Wrapper de treino/inferencia para o Helformer."""

    def __init__(
        self,
        input_size: int | None = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 60,
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
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
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
        self.model: HelformerNet | None = None

    @property
    def name(self) -> str:
        return "helformer"

    def _build_model(self, input_size: int) -> HelformerNet:
        self.input_size = input_size
        return HelformerNet(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            seq_len=self.seq_len,
        ).to(DEVICE)

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Cria sequencias de janela deslizante para o Transformer.

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
            logger.warning("Dados insuficientes para treino do Helformer")
            return {}

        input_size = X_train_seq.shape[2]
        self.model = self._build_model(input_size)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Helformer inicializado: {total_params:,} parametros, device={DEVICE}")

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
                    logger.info(f"  Helformer early stopping na epoca {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                msg = f"  Helformer epoca {epoch + 1}: train_loss={avg_train_loss:.6f}"
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

        logger.info(f"  Helformer metricas: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modelo Helformer nao treinado")
        self.model.eval()
        X_seq, _ = self._make_sequences(X)
        with torch.no_grad():
            return self.model(X_seq).cpu().numpy()

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC Dropout: roda N forward passes com dropout ativado."""
        if self.model is None:
            raise RuntimeError("Modelo Helformer nao treinado")

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
            raise RuntimeError("Modelo Helformer nao treinado")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_size": self.input_size,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "seq_len": self.seq_len,
            },
            path,
        )
        logger.info(f"Helformer salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "HelformerModel":
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        instance = cls(
            input_size=checkpoint["input_size"],
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            seq_len=checkpoint["seq_len"],
        )
        instance.model = HelformerNet(
            input_size=checkpoint["input_size"],
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            seq_len=checkpoint["seq_len"],
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state"])
        return instance
