"""Mixture Density Network (MDN) para previsoes probabilisticas multimodais.

Combina uma rede feedforward com uma mistura de Gaussianas para modelar a
distribuicao condicional completa de precos, permitindo amostragem de cenarios
e estimativa de quantis.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constante para estabilidade numerica no log
_LOG_EPS = 1e-8
_SIGMA_MIN = 1e-4


# ======================================================================
# Rede Neural — Mixture Density Network
# ======================================================================


class MixtureDensityNet(nn.Module):
    """Rede que produz parametros de uma mistura de Gaussianas.

    Parameters
    ----------
    input_size : int
        Dimensao do vetor de entrada.
    hidden_size : int
        Tamanho da camada oculta.
    n_mixtures : int
        Numero de componentes na mistura.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        n_mixtures: int = 5,
    ) -> None:
        super().__init__()
        self.n_mixtures = n_mixtures

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Cabeças para os parametros da mistura
        self.fc_pi = nn.Linear(hidden_size, n_mixtures)      # pesos da mistura
        self.fc_mu = nn.Linear(hidden_size, n_mixtures)      # medias
        self.fc_sigma = nn.Linear(hidden_size, n_mixtures)   # desvios padrao

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(pi, mu, sigma)`` — cada um com shape ``(batch, n_mixtures)``.
            ``pi`` passa por softmax, ``sigma`` por exp (com piso).
        """
        h = self.shared(x)

        pi = torch.softmax(self.fc_pi(h), dim=-1)
        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_sigma(h)).clamp(min=_SIGMA_MIN)

        return pi, mu, sigma


# ======================================================================
# Loss: Negative Log-Likelihood da mistura de Gaussianas
# ======================================================================


def mdn_nll_loss(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood para uma mistura de Gaussianas.

    Parameters
    ----------
    pi : Tensor (batch, K)
        Pesos da mistura (somam 1).
    mu : Tensor (batch, K)
        Medias dos componentes.
    sigma : Tensor (batch, K)
        Desvios padrao dos componentes.
    y : Tensor (batch,)
        Valores alvo.

    Returns
    -------
    Tensor
        Escalar — media do NLL sobre o batch.
    """
    y = y.unsqueeze(-1)  # (batch, 1)

    # Log-probabilidade de cada componente Gaussiano
    log_normal = (
        -0.5 * torch.log(2 * torch.tensor(np.pi, device=y.device))
        - torch.log(sigma + _LOG_EPS)
        - 0.5 * ((y - mu) / (sigma + _LOG_EPS)) ** 2
    )  # (batch, K)

    # Log-sum-exp ponderado pelos pesos da mistura
    log_prob = torch.logsumexp(torch.log(pi + _LOG_EPS) + log_normal, dim=-1)

    return -log_prob.mean()


# ======================================================================
# Modelo completo — wrapper BaseModel
# ======================================================================


class MDNModel(BaseModel):
    """Wrapper de treino/inferencia para Mixture Density Network.

    Parameters
    ----------
    input_size : int
        Dimensao das features de entrada.
    hidden_size : int
        Tamanho da camada oculta.
    n_mixtures : int
        Numero de componentes Gaussianos na mistura.
    lr : float
        Learning rate.
    batch_size : int
        Tamanho do mini-batch.
    max_epochs : int
        Numero maximo de epocas.
    patience : int
        Paciencia para early stopping.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        n_mixtures: int = 5,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_mixtures = n_mixtures
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        self.model: MixtureDensityNet | None = None

    @property
    def name(self) -> str:
        return "mdn"

    # ------------------------------------------------------------------
    # Utilitarios internos
    # ------------------------------------------------------------------

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(np.asarray(arr, dtype=np.float32)).to(DEVICE)

    def _build_model(self) -> MixtureDensityNet:
        return MixtureDensityNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_mixtures=self.n_mixtures,
        ).to(DEVICE)

    def _get_mixture_params(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retorna (pi, mu, sigma) como arrays numpy."""
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")
        self.model.eval()
        X_t = self._to_tensor(X)
        with torch.no_grad():
            pi, mu, sigma = self.model(X_t)
        return (
            pi.cpu().numpy(),
            mu.cpu().numpy(),
            sigma.cpu().numpy(),
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
        """Treina a MDN com NLL loss."""
        self.model = self._build_model()

        X_t = self._to_tensor(X_train)
        y_t = self._to_tensor(y_train).ravel()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=max(1, self.patience // 3), factor=0.5
        )

        # Validacao
        X_val_t, y_val_t = None, None
        if X_val is not None and y_val is not None:
            X_val_t = self._to_tensor(X_val)
            y_val_t = self._to_tensor(y_val).ravel()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None

        for epoch in range(self.max_epochs):
            # --- Treino ---
            self.model.train()
            train_losses: list[float] = []
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pi, mu, sigma = self.model(X_batch)
                loss = mdn_nll_loss(pi, mu, sigma, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train = float(np.mean(train_losses))

            # --- Validacao ---
            if X_val_t is not None and y_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    pi_v, mu_v, sigma_v = self.model(X_val_t)
                    val_loss = mdn_nll_loss(pi_v, mu_v, sigma_v, y_val_t).item()

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

                if patience_counter >= self.patience:
                    logger.info("  Early stopping na epoca %d", epoch + 1)
                    break
            else:
                # Sem validacao: guardar ultimo estado
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

            if (epoch + 1) % 20 == 0:
                msg = f"  Epoca {epoch + 1}: train_nll={avg_train:.4f}"
                if X_val_t is not None:
                    msg += f" val_nll={val_loss:.4f}"
                logger.info(msg)

        # Restaurar melhor modelo
        if best_state is not None:
            self.model.load_state_dict(best_state)

        metrics: dict[str, float] = {"train_nll": avg_train}
        if X_val_t is not None:
            metrics["val_nll"] = float(best_val_loss)
        logger.info("  MDN metricas: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Predicao
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna a media ponderada da mistura (previsao pontual)."""
        pi, mu, _ = self._get_mixture_params(X)
        # Media ponderada: sum(pi_k * mu_k)
        return np.sum(pi * mu, axis=-1)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna previsao e confianca baseada na variancia da mistura.

        A confianca e inversamente proporcional a variancia total da mistura.
        """
        pi, mu, sigma = self._get_mixture_params(X)

        # Media ponderada
        mean = np.sum(pi * mu, axis=-1)

        # Variancia total da mistura:
        # Var = sum pi_k * (sigma_k^2 + mu_k^2) - mean^2
        variance = (
            np.sum(pi * (sigma ** 2 + mu ** 2), axis=-1)
            - mean ** 2
        )
        variance = np.clip(variance, 0.0, None)
        std = np.sqrt(variance)

        # Confianca inversamente proporcional ao desvio padrao
        max_std = std.max() if std.max() > 0 else 1.0
        confidence = 1.0 - np.clip(std / max_std, 0.0, 1.0)

        return mean, confidence

    # ------------------------------------------------------------------
    # Amostragem e quantis
    # ------------------------------------------------------------------

    def sample(self, X: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """Amostra da distribuicao preditiva para geracao de cenarios.

        Parameters
        ----------
        X : np.ndarray
            Features de entrada, shape ``(n_obs, input_size)``.
        n_samples : int
            Numero de amostras por observacao.

        Returns
        -------
        np.ndarray
            Shape ``(n_obs, n_samples)``.
        """
        pi, mu, sigma = self._get_mixture_params(X)
        n_obs = pi.shape[0]
        samples = np.zeros((n_obs, n_samples), dtype=np.float64)

        rng = np.random.default_rng()

        for i in range(n_obs):
            # Escolher componente da mistura para cada amostra
            components = rng.choice(
                self.n_mixtures, size=n_samples, p=pi[i]
            )
            # Amostrar de cada componente selecionado
            samples[i] = rng.normal(
                loc=mu[i, components],
                scale=sigma[i, components],
            )

        return samples

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: list[float] | None = None,
    ) -> np.ndarray:
        """Estima quantis da distribuicao preditiva via amostragem.

        Parameters
        ----------
        X : np.ndarray
            Features de entrada.
        quantiles : list[float]
            Lista de quantis desejados (ex: ``[0.05, 0.5, 0.95]``).

        Returns
        -------
        np.ndarray
            Shape ``(n_obs, len(quantiles))``.
        """
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]

        # Usar amostragem Monte Carlo para estimar quantis
        samples = self.sample(X, n_samples=2000)
        return np.quantile(samples, quantiles, axis=1).T  # (n_obs, n_quantiles)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Salva modelo MDN em disco."""
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "n_mixtures": self.n_mixtures,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
            },
            path,
        )
        logger.info("MDN salvo: %s", path)

    @classmethod
    def load(cls, path: Path) -> "MDNModel":
        """Carrega modelo MDN do disco."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        instance = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            n_mixtures=checkpoint["n_mixtures"],
            lr=checkpoint["lr"],
            batch_size=checkpoint["batch_size"],
            max_epochs=checkpoint["max_epochs"],
            patience=checkpoint["patience"],
        )
        instance.model = MixtureDensityNet(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            n_mixtures=checkpoint["n_mixtures"],
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state"])
        logger.info("MDN carregado: %s", path)
        return instance
