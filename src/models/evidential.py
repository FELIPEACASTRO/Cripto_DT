"""Evidential Deep Learning para regressao com estimativa calibrada de incerteza.

Baseado em Amini et al. (NeurIPS 2020) e inspirado no ProbFM, implementa uma
rede feedforward que produz os parametros de uma distribuicao Normal-Inverse-Gamma
(NIG), permitindo quantificacao de incerteza aleatorica e epistemica em uma
unica passagem forward.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constante para estabilidade numerica
_EPS = 1e-6


# ======================================================================
# Configuracao
# ======================================================================


@dataclass
class EvidentialConfig:
    """Hiperparametros para o modelo Evidential."""

    hidden_sizes: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 150
    early_stop_patience: int = 15
    evidence_coeff: float = 0.1


# ======================================================================
# Rede Neural — Evidential Regression Network
# ======================================================================


class EvidentialNet(nn.Module):
    """Rede que produz os 4 parametros da distribuicao Normal-Inverse-Gamma.

    Saidas: gamma (media), nu (evidencia), alpha (shape), beta (scale).

    Parameters
    ----------
    input_size : int
        Dimensao do vetor de entrada.
    hidden_sizes : list[int]
        Tamanhos das camadas ocultas.
    dropout : float
        Taxa de dropout entre camadas.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        layers: list[nn.Module] = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h_size

        self.shared = nn.Sequential(*layers)

        # Cabeca que produz os 4 parametros NIG
        self.fc_out = nn.Linear(prev_size, 4)

        # Softplus para garantir positividade
        self.softplus = nn.Softplus()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            ``(gamma, nu, alpha, beta)`` — cada um com shape ``(batch,)``.
            - gamma: previsao pontual (sem restricao)
            - nu: evidencia (softplus, > 0)
            - alpha: shape NIG (softplus + 1, > 1)
            - beta: scale NIG (softplus, > 0)
        """
        h = self.shared(x)
        raw = self.fc_out(h)  # (batch, 4)

        gamma = raw[:, 0]
        nu = self.softplus(raw[:, 1])
        alpha = self.softplus(raw[:, 2]) + 1.0
        beta = self.softplus(raw[:, 3])

        return gamma, nu, alpha, beta


# ======================================================================
# Loss: NIG Negative Log-Likelihood + Evidence Regularizer
# ======================================================================


def nig_nll(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood da distribuicao Normal-Inverse-Gamma.

    L = 0.5*log(pi/nu) - alpha*log(Omega)
        + (alpha+0.5)*log((y-gamma)^2*nu + Omega)
        + log(Gamma(alpha)/Gamma(alpha+0.5))

    onde Omega = 2*beta*(1+nu).

    Parameters
    ----------
    gamma, nu, alpha, beta : Tensor (batch,)
        Parametros NIG preditos pela rede.
    y : Tensor (batch,)
        Valores alvo.

    Returns
    -------
    Tensor
        Escalar — media do NLL sobre o batch.
    """
    omega = 2.0 * beta * (1.0 + nu)

    nll = (
        0.5 * torch.log(torch.tensor(np.pi, device=y.device) / (nu + _EPS))
        - alpha * torch.log(omega + _EPS)
        + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega + _EPS)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    return nll.mean()


def evidence_regularizer(
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    y: torch.Tensor,
    coeff: float = 0.1,
) -> torch.Tensor:
    """Regularizador de evidencia que penaliza alta evidencia em predicoes erradas.

    R = coeff * |y - gamma| * (2*nu + alpha)

    Parameters
    ----------
    gamma, nu, alpha : Tensor (batch,)
        Parametros NIG preditos pela rede.
    y : Tensor (batch,)
        Valores alvo.
    coeff : float
        Coeficiente de regularizacao.

    Returns
    -------
    Tensor
        Escalar — media do regularizador sobre o batch.
    """
    reg = coeff * torch.abs(y - gamma) * (2.0 * nu + alpha)
    return reg.mean()


# ======================================================================
# Modelo completo — wrapper BaseModel
# ======================================================================


class EvidentialModel(BaseModel):
    """Wrapper de treino/inferencia para Evidential Deep Learning.

    Produz os parametros de uma distribuicao Normal-Inverse-Gamma (NIG)
    para quantificacao de incerteza em uma unica passagem forward.
    A confianca e derivada diretamente da evidencia (nu), resolvendo o
    problema de confianca 0% observado em abordagens baseadas em variancia.

    Parameters
    ----------
    input_size : int
        Dimensao das features de entrada.
    cfg : EvidentialConfig | None
        Configuracao de hiperparametros. Se None, usa valores padrao.
    """

    def __init__(
        self,
        input_size: int,
        cfg: EvidentialConfig | None = None,
    ) -> None:
        self.input_size = input_size
        self.cfg = cfg or EvidentialConfig()

        self.model: EvidentialNet | None = None

    @property
    def name(self) -> str:
        return "evidential"

    # ------------------------------------------------------------------
    # Utilitarios internos
    # ------------------------------------------------------------------

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(np.asarray(arr, dtype=np.float32)).to(DEVICE)

    def _build_model(self) -> EvidentialNet:
        return EvidentialNet(
            input_size=self.input_size,
            hidden_sizes=self.cfg.hidden_sizes,
            dropout=self.cfg.dropout,
        ).to(DEVICE)

    def _get_nig_params(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retorna (gamma, nu, alpha, beta) como arrays numpy."""
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")
        self.model.eval()
        X_t = self._to_tensor(X)
        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(X_t)
        return (
            gamma.cpu().numpy(),
            nu.cpu().numpy(),
            alpha.cpu().numpy(),
            beta.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Treina o modelo Evidential com NIG NLL + regularizador de evidencia."""
        self.model = self._build_model()

        X_t = self._to_tensor(X_train)
        y_t = self._to_tensor(y_train).ravel()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=max(1, self.cfg.early_stop_patience // 3),
            factor=0.5,
        )

        # Validacao
        X_val_t, y_val_t = None, None
        if X_val is not None and y_val is not None:
            X_val_t = self._to_tensor(X_val)
            y_val_t = self._to_tensor(y_val).ravel()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None

        for epoch in range(self.cfg.max_epochs):
            # --- Treino ---
            self.model.train()
            train_losses: list[float] = []
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                gamma, nu, alpha, beta = self.model(X_batch)

                loss_nll = nig_nll(gamma, nu, alpha, beta, y_batch)
                loss_reg = evidence_regularizer(
                    gamma, nu, alpha, y_batch, coeff=self.cfg.evidence_coeff
                )
                loss = loss_nll + loss_reg

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train = float(np.mean(train_losses))

            # --- Validacao ---
            if X_val_t is not None and y_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    v_gamma, v_nu, v_alpha, v_beta = self.model(X_val_t)
                    val_nll = nig_nll(
                        v_gamma, v_nu, v_alpha, v_beta, y_val_t
                    )
                    val_reg = evidence_regularizer(
                        v_gamma, v_nu, v_alpha, y_val_t,
                        coeff=self.cfg.evidence_coeff,
                    )
                    val_loss = (val_nll + val_reg).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= self.cfg.early_stop_patience:
                    logger.info("  Early stopping na epoca %d", epoch + 1)
                    break
            else:
                # Sem validacao: guardar ultimo estado
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

            if (epoch + 1) % 20 == 0:
                msg = f"  Epoca {epoch + 1}: train_loss={avg_train:.4f}"
                if X_val_t is not None:
                    msg += f" val_loss={val_loss:.4f}"
                logger.info(msg)

        # Restaurar melhor modelo
        if best_state is not None:
            self.model.load_state_dict(best_state)

        metrics: dict[str, float] = {"train_loss": avg_train}
        if X_val_t is not None:
            metrics["val_loss"] = float(best_val_loss)
        logger.info("  Evidential metricas: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Predicao
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna gamma (previsao pontual da distribuicao NIG)."""
        gamma, _, _, _ = self._get_nig_params(X)
        return gamma

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna previsao e confianca baseada na evidencia (nu).

        Confianca = nu / (nu + 1), limitada a [0, 1].
        Quanto maior a evidencia acumulada pela rede, maior a confianca.
        Isso resolve o problema de confianca 0% observado em abordagens
        baseadas em variancia relativa.
        """
        gamma, nu, _, _ = self._get_nig_params(X)

        # Confianca derivada diretamente da evidencia
        confidence = nu / (nu + 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)

        return gamma, confidence

    # ------------------------------------------------------------------
    # Extras: incerteza decomposta
    # ------------------------------------------------------------------

    def predict_uncertainty(
        self, X: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Retorna predicao com decomposicao de incerteza aleatorica e epistemica.

        Returns
        -------
        dict[str, np.ndarray]
            - ``prediction``: gamma (previsao pontual)
            - ``aleatoric``: beta / (alpha - 1) — incerteza aleatorica (ruido nos dados)
            - ``epistemic``: beta / (nu * (alpha - 1)) — incerteza epistemica (falta de dados)
            - ``confidence``: nu / (nu + 1)
        """
        gamma, nu, alpha, beta = self._get_nig_params(X)

        # Incerteza aleatorica: variancia esperada da NIG
        aleatoric = beta / (alpha - 1.0 + _EPS)

        # Incerteza epistemica: variancia da media da NIG
        epistemic = beta / (nu * (alpha - 1.0 + _EPS) + _EPS)

        confidence = np.clip(nu / (nu + 1.0), 0.0, 1.0)

        return {
            "prediction": gamma,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Salva modelo Evidential em disco."""
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_sizes": self.cfg.hidden_sizes,
                "dropout": self.cfg.dropout,
                "learning_rate": self.cfg.learning_rate,
                "batch_size": self.cfg.batch_size,
                "max_epochs": self.cfg.max_epochs,
                "early_stop_patience": self.cfg.early_stop_patience,
                "evidence_coeff": self.cfg.evidence_coeff,
            },
            path,
        )
        logger.info("Evidential salvo: %s", path)

    @classmethod
    def load(cls, path: Path) -> "EvidentialModel":
        """Carrega modelo Evidential do disco."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        cfg = EvidentialConfig(
            hidden_sizes=checkpoint["hidden_sizes"],
            dropout=checkpoint["dropout"],
            learning_rate=checkpoint["learning_rate"],
            batch_size=checkpoint["batch_size"],
            max_epochs=checkpoint["max_epochs"],
            early_stop_patience=checkpoint["early_stop_patience"],
            evidence_coeff=checkpoint["evidence_coeff"],
        )

        instance = cls(
            input_size=checkpoint["input_size"],
            cfg=cfg,
        )
        instance.model = EvidentialNet(
            input_size=checkpoint["input_size"],
            hidden_sizes=cfg.hidden_sizes,
            dropout=cfg.dropout,
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state"])
        logger.info("Evidential carregado: %s", path)
        return instance
