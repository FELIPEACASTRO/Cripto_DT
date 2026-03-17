"""Mamba SSM (State Space Model) para previsao de series temporais de criptomoedas.

Implementacao pura em PyTorch de uma arquitetura inspirada no Mamba, utilizando
mecanismo de State Space seletivo com parametros B, C, delta dependentes da entrada.
Nao depende do pacote mamba-ssm (que requer Linux/CUDA build).

Referencia: Gu & Dao (2023) - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config.settings import FeatureConfig, config as default_config
from src.models.base import BaseModel
from src.models.losses import MADLLoss

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class MambaConfig:
    d_model: int = 64
    d_state: int = 16
    n_layers: int = 4
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 10
    mc_dropout_samples: int = 30
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    grad_clip_max_norm: float = 1.0
    loss_type: str = "madl"
    madl_alpha: float = 2.0
    d_conv: int = 4  # Kernel size para depthwise conv1d
    expand_factor: int = 2  # Fator de expansao do inner dimension
    dt_rank: str = "auto"  # "auto" -> ceil(d_model / 16)


class SelectiveSSM(nn.Module):
    """Nucleo do mecanismo Selective State Space.

    Implementa a discretizacao A, B com delta dependente da entrada,
    seguida de recorrencia sequencial (scan).
    """

    def __init__(self, d_inner: int, d_state: int, dt_rank: int):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Parametro A continuo (log-space para estabilidade)
        # Inicializacao S4D-Real: A_n = -1/2 + ni
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D eh o skip connection (residual)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Projecoes para parametros dependentes da entrada
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # Projecao de dt_rank para d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Inicializar dt_proj bias para ter deltas pequenos inicialmente
        dt_init_std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Bias inicializado para produzir deltas entre 0.001 e 0.1
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverso do softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_inner)
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape

        # Parametros dependentes da entrada
        x_proj = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        dt, B, C = x_proj.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Projecao e ativacao de delta (step size)
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)  # Garantir delta positivo

        # Discretizacao: A_bar = exp(delta * A), B_bar = delta * B
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negativo para estabilidade
        # A_bar: (batch, seq_len, d_inner, d_state)
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # B_bar: (batch, seq_len, d_inner, d_state)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(2).expand(-1, -1, d_inner, -1)

        # Scan sequencial (recorrencia)
        y = self._selective_scan(x, A_bar, B_bar, C)

        # Skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def _selective_scan(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Recorrencia sequencial: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t; y_t = C_t * h_t

        Args:
            x: (batch, seq_len, d_inner)
            A_bar: (batch, seq_len, d_inner, d_state)
            B_bar: (batch, seq_len, d_inner, d_state)
            C: (batch, seq_len, d_state)
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape

        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # y_t = C_t * h_t -> sum over d_state
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)


class MambaBlock(nn.Module):
    """Bloco Mamba: Linear -> Conv1d -> SiLU -> SSM -> output projection.

    Segue a arquitetura do paper com dual path (x e z) e gating multiplicativo.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand_factor: int, dt_rank: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand_factor

        # Projecao de entrada: projeta para 2*d_inner (x_path e z_path)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Depthwise Conv1d no path principal
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # SSM core
        self.ssm = SelectiveSSM(d_inner, d_state, dt_rank)

        # Projecao de saida
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Projecao dual
        xz = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)  # Cada um: (batch, seq_len, d_inner)

        # Conv1d no x_path: transpor para (batch, d_inner, seq_len)
        x_conv = x_path.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Truncar padding causal
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)

        # Ativacao SiLU (Swish)
        x_conv = F.silu(x_conv)

        # SSM core
        x_ssm = self.ssm(x_conv)

        # Gating multiplicativo com z
        x_ssm = x_ssm * F.silu(z)

        # Projecao de saida
        out = self.out_proj(x_ssm)
        out = self.dropout(out)

        return out


class MambaNet(nn.Module):
    """Rede Mamba completa: stack de MambaBlocks com residual connections e LayerNorm."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        d_state: int,
        n_layers: int,
        d_conv: int,
        expand_factor: int,
        dt_rank: int,
        dropout: float,
    ):
        super().__init__()

        # Projecao de entrada
        self.input_proj = nn.Linear(input_size, d_model)

        # Stack de blocos Mamba com LayerNorm
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                MambaBlock(d_model, d_state, d_conv, expand_factor, dt_rank, dropout)
            )
            self.norms.append(nn.LayerNorm(d_model))

        # Norm final
        self.final_norm = nn.LayerNorm(d_model)

        # Head de regressao
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch,)
        """
        # Projecao de entrada
        h = self.input_proj(x)  # (batch, seq_len, d_model)

        # Stack de MambaBlocks com residual + LayerNorm (pre-norm)
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            h = layer(h) + residual

        # Norm final e pegar ultimo timestep
        h = self.final_norm(h)
        h = h[:, -1, :]  # (batch, d_model)

        # Head de saida
        h = self.dropout(h)
        out = self.fc(h)
        return out.squeeze(-1)


class MambaModel(BaseModel):
    """Wrapper de treino/inferencia para Mamba SSM."""

    def __init__(
        self,
        input_size: int | None = None,
        mamba_config: MambaConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        self.cfg = mamba_config or MambaConfig()
        self.feat_cfg = feature_config or default_config.features
        self.input_size = input_size
        self.model: MambaNet | None = None

        # Calcular dt_rank
        if self.cfg.dt_rank == "auto":
            self._dt_rank = math.ceil(self.cfg.d_model / 16)
        else:
            self._dt_rank = int(self.cfg.dt_rank)

    @property
    def name(self) -> str:
        return "mamba_ssm"

    def _build_model(self, input_size: int) -> MambaNet:
        self.input_size = input_size
        return MambaNet(
            input_size=input_size,
            d_model=self.cfg.d_model,
            d_state=self.cfg.d_state,
            n_layers=self.cfg.n_layers,
            d_conv=self.cfg.d_conv,
            expand_factor=self.cfg.expand_factor,
            dt_rank=self._dt_rank,
            dropout=self.cfg.dropout,
        ).to(DEVICE)

    def _make_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        """Cria sequencias de lookback_window para o modelo.

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

        logger.info(
            f"  Mamba SSM: d_model={self.cfg.d_model}, d_state={self.cfg.d_state}, "
            f"n_layers={self.cfg.n_layers}, dt_rank={self._dt_rank}, "
            f"params={sum(p.numel() for p in self.model.parameters()):,}"
        )

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
        avg_train_loss = float("inf")

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
            "input_size": self.input_size,
            "d_model": self.cfg.d_model,
            "d_state": self.cfg.d_state,
            "n_layers": self.cfg.n_layers,
            "d_conv": self.cfg.d_conv,
            "expand_factor": self.cfg.expand_factor,
            "dt_rank": self._dt_rank,
            "dropout": self.cfg.dropout,
            "lookback_window": self.feat_cfg.lookback_window,
        }, path)
        logger.info(f"Modelo salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "MambaModel":
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        cfg = MambaConfig(
            d_model=checkpoint["d_model"],
            d_state=checkpoint["d_state"],
            n_layers=checkpoint["n_layers"],
            d_conv=checkpoint["d_conv"],
            expand_factor=checkpoint["expand_factor"],
            dropout=checkpoint["dropout"],
        )
        from config.settings import FeatureConfig
        feat_cfg = FeatureConfig(lookback_window=checkpoint.get("lookback_window", 60))
        instance = cls(input_size=checkpoint["input_size"], mamba_config=cfg, feature_config=feat_cfg)
        instance._dt_rank = checkpoint["dt_rank"]
        instance.model = MambaNet(
            input_size=checkpoint["input_size"],
            d_model=checkpoint["d_model"],
            d_state=checkpoint["d_state"],
            n_layers=checkpoint["n_layers"],
            d_conv=checkpoint["d_conv"],
            expand_factor=checkpoint["expand_factor"],
            dt_rank=checkpoint["dt_rank"],
            dropout=checkpoint["dropout"],
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state"])
        return instance
