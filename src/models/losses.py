"""Loss functions customizadas para previsao de series temporais financeiras."""

import torch
import torch.nn as nn


class MADLLoss(nn.Module):
    """Mean Absolute Directional Loss.

    Penaliza mais previsoes que erram a direcao do movimento do que
    previsoes que erram apenas a magnitude. Isso e crucial para trading
    onde acertar a direcao importa mais que acertar o valor exato.

    L = |y - y_hat| * (1 + alpha * 1[sign(y) != sign(y_hat)])

    Args:
        alpha: Peso extra para penalidade direcional (default: 2.0)
        base_loss: Loss base ("mae" ou "huber")
    """

    def __init__(self, alpha: float = 2.0, base_loss: str = "mae"):
        super().__init__()
        self.alpha = alpha
        if base_loss == "huber":
            self.base = nn.HuberLoss(reduction="none")
        else:
            self.base = nn.L1Loss(reduction="none")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Loss base (MAE ou Huber) por amostra
        base_loss = self.base(y_pred, y_true)

        # Penalidade direcional: 1 quando sinais diferem, 0 quando iguais
        sign_mismatch = (torch.sign(y_pred) != torch.sign(y_true)).float()

        # Loss final: base * (1 + alpha * mismatch)
        weighted_loss = base_loss * (1.0 + self.alpha * sign_mismatch)

        return weighted_loss.mean()


class DirectionalHuberLoss(nn.Module):
    """Combinacao de HuberLoss com componente de classificacao direcional.

    L = (1-lambda) * Huber(y, y_hat) + lambda * BCE(sign(y), sigmoid(y_hat))

    Args:
        lambda_dir: Peso do componente direcional (default: 0.3)
        huber_delta: Delta do HuberLoss (default: 1.0)
    """

    def __init__(self, lambda_dir: float = 0.3, huber_delta: float = 1.0):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Componente de regressao
        reg_loss = self.huber(y_pred, y_true)

        # Componente direcional
        true_dir = (y_true > 0).float()
        # Escalar predicoes para logits (magnitude -> confianca direcional)
        pred_logits = y_pred * 100  # amplificar para sigmoid funcionar bem
        dir_loss = self.bce(pred_logits, true_dir)

        return (1 - self.lambda_dir) * reg_loss + self.lambda_dir * dir_loss
