"""Bayesian Neural Network (BNN) com variational inference (Bayes by Backprop).

Baseado em estudos da Seoul National University e University of Cagliari,
implementa uma rede neural com pesos probabilisticos para quantificacao
de incerteza epistemica via amostragem de Monte Carlo.
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


# ======================================================================
# Camada Linear Bayesiana
# ======================================================================


class BayesianLinear(nn.Module):
    """Camada linear com pesos probabilisticos (variational inference).

    Cada peso possui parametros mu (media) e rho, onde
    sigma = log(1 + exp(rho)) (softplus). Durante o forward pass,
    os pesos sao amostrados de N(mu, sigma) usando o reparameterization trick.

    Parameters
    ----------
    in_features : int
        Dimensao de entrada.
    out_features : int
        Dimensao de saida.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parametros da posterior q(w|theta) para pesos
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))

        # Parametros da posterior q(b|theta) para bias
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))

        # Inicializacao dos parametros mu (Xavier-like)
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.zeros_(self.bias_mu)

    def _sigma(self, rho: torch.Tensor) -> torch.Tensor:
        """Converte rho em sigma via softplus: sigma = log(1 + exp(rho))."""
        return torch.log1p(torch.exp(rho))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass com amostragem de pesos via reparameterization trick.

        w = mu + sigma * epsilon, onde epsilon ~ N(0, 1).
        """
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)

        # Reparameterization trick
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Calcula a divergencia KL entre posterior q(w|theta) e prior p(w) = N(0,1).

        KL(q || p) = sum[ log(sigma_prior/sigma_q) + (sigma_q^2 + mu_q^2)/(2*sigma_prior^2) - 0.5 ]
        Com prior N(0,1): sigma_prior = 1, simplifica para:
        KL = sum[ -log(sigma_q) + (sigma_q^2 + mu_q^2)/2 - 0.5 ]
        """
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)

        # KL para pesos
        kl_weight = (
            -torch.log(weight_sigma)
            + (weight_sigma ** 2 + self.weight_mu ** 2) / 2.0
            - 0.5
        ).sum()

        # KL para bias
        kl_bias = (
            -torch.log(bias_sigma)
            + (bias_sigma ** 2 + self.bias_mu ** 2) / 2.0
            - 0.5
        ).sum()

        return kl_weight + kl_bias


# ======================================================================
# Rede Bayesiana completa
# ======================================================================


class BayesianNet(nn.Module):
    """Rede neural Bayesiana com camadas BayesianLinear.

    Arquitetura: input -> 128 -> ReLU -> 64 -> ReLU -> 1

    Parameters
    ----------
    input_size : int
        Dimensao do vetor de entrada.
    hidden_size : int
        Tamanho da primeira camada oculta (segunda = hidden_size // 2).
    """

    def __init__(self, input_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.bl1 = BayesianLinear(input_size, hidden_size)
        self.bl2 = BayesianLinear(hidden_size, hidden_size // 2)
        self.bl3 = BayesianLinear(hidden_size // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass retornando output e KL total.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(output, kl_total)`` — output com shape ``(batch, 1)``,
            kl_total e escalar com a soma da KL de todas as camadas.
        """
        x = self.relu(self.bl1(x))
        x = self.relu(self.bl2(x))
        output = self.bl3(x)

        # Soma da KL de todas as camadas Bayesianas
        kl_total = self.bl1.kl_divergence() + self.bl2.kl_divergence() + self.bl3.kl_divergence()

        return output, kl_total


# ======================================================================
# Modelo completo — wrapper BaseModel
# ======================================================================


class BNNModel(BaseModel):
    """Wrapper de treino/inferencia para Bayesian Neural Network.

    Utiliza ELBO loss (MSE + kl_weight * KL_divergence) e inferencia
    por Monte Carlo com multiplos forward passes para estimativa de incerteza.

    Parameters
    ----------
    input_size : int
        Dimensao das features de entrada.
    hidden_size : int
        Tamanho da primeira camada oculta.
    lr : float
        Learning rate.
    batch_size : int
        Tamanho do mini-batch.
    max_epochs : int
        Numero maximo de epocas.
    patience : int
        Paciencia para early stopping.
    n_samples : int
        Numero de forward passes para inferencia Monte Carlo.
    kl_weight : float
        Peso da divergencia KL na ELBO loss.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        n_samples: int = 30,
        kl_weight: float = 1e-3,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.n_samples = n_samples
        self.kl_weight = kl_weight

        self.model: BayesianNet | None = None

    @property
    def name(self) -> str:
        return "bnn"

    # ------------------------------------------------------------------
    # Utilitarios internos
    # ------------------------------------------------------------------

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(np.asarray(arr, dtype=np.float32)).to(DEVICE)

    def _build_model(self) -> BayesianNet:
        return BayesianNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        ).to(DEVICE)

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
        """Treina a BNN com ELBO loss (MSE + kl_weight * KL)."""
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
        mse_fn = nn.MSELoss()

        for epoch in range(self.max_epochs):
            # --- Treino ---
            self.model.train()
            train_losses: list[float] = []
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output, kl = self.model(X_batch)
                mse = mse_fn(output.ravel(), y_batch)
                loss = mse + self.kl_weight * kl
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train = float(np.mean(train_losses))

            # --- Validacao ---
            if X_val_t is not None and y_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output, val_kl = self.model(X_val_t)
                    val_mse = mse_fn(val_output.ravel(), y_val_t)
                    val_loss = (val_mse + self.kl_weight * val_kl).item()

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
                msg = f"  Epoca {epoch + 1}: train_elbo={avg_train:.4f}"
                if X_val_t is not None:
                    msg += f" val_elbo={val_loss:.4f}"
                logger.info(msg)

        # Restaurar melhor modelo
        if best_state is not None:
            self.model.load_state_dict(best_state)

        metrics: dict[str, float] = {"train_elbo": avg_train}
        if X_val_t is not None:
            metrics["val_elbo"] = float(best_val_loss)
        logger.info("  BNN metricas: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Predicao
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna a media de n_samples forward passes (Monte Carlo)."""
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")

        self.model.train()  # Manter estocasticidade nas camadas Bayesianas
        X_t = self._to_tensor(X)

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                output, _ = self.model(X_t)
                predictions.append(output.ravel().cpu().numpy())

        return np.mean(predictions, axis=0)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna previsao e confianca via Monte Carlo com n_samples forward passes.

        A confianca e inversamente proporcional ao desvio padrao das amostras.
        """
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")

        self.model.train()  # Manter estocasticidade nas camadas Bayesianas
        X_t = self._to_tensor(X)

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                output, _ = self.model(X_t)
                predictions.append(output.ravel().cpu().numpy())

        predictions = np.array(predictions)  # (n_samples, n_obs)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)

        # Confianca inversamente proporcional ao desvio padrao
        max_std = std.max() if std.max() > 0 else 1.0
        confidence = 1.0 - np.clip(std / max_std, 0.0, 1.0)

        return mean, confidence

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Salva modelo BNN em disco."""
        if self.model is None:
            raise RuntimeError("Modelo nao treinado")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "n_samples": self.n_samples,
                "kl_weight": self.kl_weight,
            },
            path,
        )
        logger.info("BNN salvo: %s", path)

    @classmethod
    def load(cls, path: Path) -> "BNNModel":
        """Carrega modelo BNN do disco."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        instance = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            lr=checkpoint["lr"],
            batch_size=checkpoint["batch_size"],
            max_epochs=checkpoint["max_epochs"],
            patience=checkpoint["patience"],
            n_samples=checkpoint["n_samples"],
            kl_weight=checkpoint["kl_weight"],
        )
        instance.model = BayesianNet(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state"])
        logger.info("BNN carregado: %s", path)
        return instance
