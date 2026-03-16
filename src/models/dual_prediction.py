"""Modelo de previsao dual inspirado no CryptoPulse.

Arquitetura com dois heads compartilhando um backbone comum:
- Head de regressao: preve magnitude do log-return
- Head de classificacao: preve probabilidade de direcao (up/down)

A previsao final combina ambos: magnitude * sinal da classificacao.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DualPredictionConfig:
    """Configuracao do modelo de previsao dual."""

    hidden_sizes: list[int] | None = None
    head_size: int = 64
    dropout: float = 0.2
    alpha: float = 0.6
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 10
    classification_threshold: float = 0.6

    def __post_init__(self) -> None:
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128]


class DualPredictionNet(nn.Module):
    """Rede neural com backbone compartilhado e dois heads.

    Backbone:
        Linear(features, 256) -> ReLU -> Dropout -> Linear(256, 128) -> ReLU -> Dropout

    Head de regressao:
        Linear(128, 64) -> ReLU -> Linear(64, 1)

    Head de classificacao:
        Linear(128, 64) -> ReLU -> Linear(64, 1) -> Sigmoid
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: list[int],
        head_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Backbone compartilhado
        backbone_layers: list[nn.Module] = []
        in_size = n_features
        for h_size in hidden_sizes:
            backbone_layers.extend([
                nn.Linear(in_size, h_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = h_size
        self.backbone = nn.Sequential(*backbone_layers)

        backbone_out = hidden_sizes[-1]

        # Head de regressao (magnitude do log-return)
        self.regression_head = nn.Sequential(
            nn.Linear(backbone_out, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
        )

        # Head de classificacao (probabilidade de direcao up)
        self.classification_head = nn.Sequential(
            nn.Linear(backbone_out, head_size),
            nn.ReLU(),
            nn.Linear(head_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            (regression_output, classification_output)
            regression_output: shape (batch, 1) -- magnitude prevista
            classification_output: shape (batch, 1) -- prob de up [0, 1]
        """
        features = self.backbone(x)
        reg_out = self.regression_head(features)
        cls_out = self.classification_head(features)
        return reg_out, cls_out


class DualPredictionModel(BaseModel):
    """Modelo de previsao dual com heads de regressao e classificacao.

    Loss combinada:
        loss = alpha * MSE(regression) + (1 - alpha) * BCE(classification)

    Previsao final:
        Se classificacao > threshold: pred = |regression| * +1
        Se classificacao < (1-threshold): pred = |regression| * -1
        Senao: pred = regression (sem correcao)
    """

    def __init__(self, config: DualPredictionConfig | None = None):
        self.config = config or DualPredictionConfig()
        self.net: DualPredictionNet | None = None
        self.n_features: int = 0
        self.train_mean: np.ndarray | None = None
        self.train_std: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "dual_prediction"

    # ------------------------------------------------------------------
    # Normalizacao
    # ------------------------------------------------------------------

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normaliza features com z-score."""
        if fit:
            self.train_mean = X.mean(axis=0)
            self.train_std = X.std(axis=0)
            self.train_std[self.train_std < 1e-10] = 1.0

        assert self.train_mean is not None and self.train_std is not None
        return (X - self.train_mean) / self.train_std

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Treina o modelo dual.

        Args:
            X_train: shape (n_samples, n_features)
            y_train: shape (n_samples,) -- log-returns
            X_val / y_val: dados de validacao para early stopping
        """
        self.n_features = X_train.shape[1]
        assert self.config.hidden_sizes is not None

        # Normalizar
        X_train_norm = self._normalize(X_train, fit=True)

        # Criar rede
        self.net = DualPredictionNet(
            n_features=self.n_features,
            hidden_sizes=self.config.hidden_sizes,
            head_size=self.config.head_size,
            dropout=self.config.dropout,
        ).to(DEVICE)

        # Targets
        y_reg = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        # Classificacao: 1 se retorno > 0, senao 0
        y_cls = torch.tensor(
            (y_train > 0).astype(np.float32), dtype=torch.float32
        ).unsqueeze(1)

        X_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
        train_dataset = TensorDataset(X_tensor, y_reg, y_cls)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Validacao
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_norm = self._normalize(X_val)
            X_val_t = torch.tensor(X_val_norm, dtype=torch.float32)
            y_val_reg = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
            y_val_cls = torch.tensor(
                (y_val > 0).astype(np.float32), dtype=torch.float32
            ).unsqueeze(1)
            val_dataset = TensorDataset(X_val_t, y_val_reg, y_val_cls)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        # Losses e otimizador
        reg_criterion = nn.MSELoss()
        cls_criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.config.learning_rate
        )

        alpha = self.config.alpha
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(
            f"  Treinando {self.name}: features={self.n_features}, "
            f"alpha={alpha}, device={DEVICE}"
        )

        for epoch in range(self.config.max_epochs):
            # --- Treino ---
            self.net.train()
            epoch_reg_loss = 0.0
            epoch_cls_loss = 0.0
            n_batches = 0

            for X_batch, y_reg_batch, y_cls_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_reg_batch = y_reg_batch.to(DEVICE)
                y_cls_batch = y_cls_batch.to(DEVICE)

                optimizer.zero_grad()
                reg_pred, cls_pred = self.net(X_batch)

                loss_reg = reg_criterion(reg_pred, y_reg_batch)
                loss_cls = cls_criterion(cls_pred, y_cls_batch)
                loss = alpha * loss_reg + (1 - alpha) * loss_cls

                loss.backward()
                optimizer.step()

                epoch_reg_loss += loss_reg.item()
                epoch_cls_loss += loss_cls.item()
                n_batches += 1

            avg_reg = epoch_reg_loss / max(n_batches, 1)
            avg_cls = epoch_cls_loss / max(n_batches, 1)

            # --- Validacao ---
            val_loss = None
            if val_loader is not None:
                self.net.eval()
                val_reg_loss = 0.0
                val_cls_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for X_vb, y_vr, y_vc in val_loader:
                        X_vb = X_vb.to(DEVICE)
                        y_vr = y_vr.to(DEVICE)
                        y_vc = y_vc.to(DEVICE)

                        reg_p, cls_p = self.net(X_vb)
                        val_reg_loss += reg_criterion(reg_p, y_vr).item()
                        val_cls_loss += cls_criterion(cls_p, y_vc).item()
                        val_batches += 1

                vr = val_reg_loss / max(val_batches, 1)
                vc = val_cls_loss / max(val_batches, 1)
                val_loss = alpha * vr + (1 - alpha) * vc

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.clone() for k, v in self.net.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stop_patience:
                    logger.info(
                        f"  Early stopping na epoca {epoch + 1} "
                        f"(val_loss={val_loss:.6f})"
                    )
                    self.net.load_state_dict(best_state)
                    break

            if epoch % 20 == 0:
                val_str = f", val_loss={val_loss:.6f}" if val_loss is not None else ""
                logger.info(
                    f"  Epoca {epoch + 1:3d}/{self.config.max_epochs}: "
                    f"reg_loss={avg_reg:.6f}, cls_loss={avg_cls:.6f}{val_str}"
                )

        # --- Metricas finais ---
        metrics = self._compute_metrics(X_train, y_train, prefix="train")
        if X_val is not None and y_val is not None:
            val_metrics = self._compute_metrics(X_val, y_val, prefix="val")
            metrics.update(val_metrics)

        logger.info(f"  {self.name} metricas: {metrics}")
        return metrics

    def _compute_metrics(
        self, X: np.ndarray, y: np.ndarray, prefix: str
    ) -> dict[str, float]:
        """Calcula metricas de avaliacao."""
        preds = self.predict(X)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y)))
        return {f"{prefix}_rmse": rmse, f"{prefix}_dir_acc": dir_acc}

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def _forward_numpy(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Executa forward pass e retorna arrays numpy.

        Returns:
            (regression_preds, classification_probs)
        """
        if self.net is None:
            raise RuntimeError("Modelo nao treinado. Chame fit() primeiro.")

        X_norm = self._normalize(X)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)

        self.net.eval()
        with torch.no_grad():
            reg_out, cls_out = self.net(X_tensor)

        reg_preds = reg_out.cpu().numpy().flatten()
        cls_probs = cls_out.cpu().numpy().flatten()
        return reg_preds, cls_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gera previsoes combinando regressao e classificacao.

        Quando a classificacao tem alta confianca (> threshold ou < 1-threshold),
        usa o sinal da classificacao com a magnitude da regressao.
        """
        reg_preds, cls_probs = self._forward_numpy(X)

        threshold = self.config.classification_threshold
        preds = reg_preds.copy()

        # Alta confianca bullish
        bullish_mask = cls_probs > threshold
        preds[bullish_mask] = np.abs(reg_preds[bullish_mask])

        # Alta confianca bearish
        bearish_mask = cls_probs < (1.0 - threshold)
        preds[bearish_mask] = -np.abs(reg_preds[bearish_mask])

        return preds

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Previsoes com confianca baseada no head de classificacao.

        Confianca = distancia da probabilidade ao ponto de indecisao (0.5),
        escalada para [0, 1].
        """
        preds = self.predict(X)
        _, cls_probs = self._forward_numpy(X)

        # Confianca: |prob - 0.5| * 2 -> [0, 1]
        confidence = np.abs(cls_probs - 0.5) * 2.0
        confidence = np.clip(confidence, 0.0, 1.0)

        return preds, confidence

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Salva modelo em disco."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": self.config,
            "n_features": self.n_features,
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "net_state": self.net.state_dict() if self.net is not None else None,
        }
        torch.save(data, path)
        logger.info(f"Dual prediction salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "DualPredictionModel":
        """Carrega modelo do disco."""
        data = torch.load(path, map_location=DEVICE, weights_only=False)
        instance = cls(config=data["config"])
        instance.n_features = data["n_features"]
        instance.train_mean = data["train_mean"]
        instance.train_std = data["train_std"]

        if data["net_state"] is not None:
            assert instance.config.hidden_sizes is not None
            instance.net = DualPredictionNet(
                n_features=instance.n_features,
                hidden_sizes=instance.config.hidden_sizes,
                head_size=instance.config.head_size,
                dropout=instance.config.dropout,
            ).to(DEVICE)
            instance.net.load_state_dict(data["net_state"])

        return instance
