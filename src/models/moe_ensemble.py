"""Mixture of Experts (MoE) Gating Network para ensemble inteligente.

Inspirado em MIGA (arXiv:2410.02241). Em vez de pesos fixos de ensemble,
aprende a rotear previsoes atraves de modelos especializados com base
nas condicoes de mercado.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class TopKSoftmax(nn.Module):
    """Softmax esparso que ativa apenas os top-K modelos.

    Zera os pesos abaixo do limiar top-K e renormaliza os
    pesos restantes para somar 1.
    """

    def __init__(self, top_k: int = 6):
        super().__init__()
        self.top_k = top_k

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Aplica softmax esparso com selecao top-K.

        Args:
            logits: Tensor de shape (batch, n_models) com logits brutos.

        Returns:
            Tensor de shape (batch, n_models) com pesos normalizados,
            onde apenas os top-K modelos tem peso > 0.
        """
        n_models = logits.shape[-1]
        k = min(self.top_k, n_models)

        # Softmax completo primeiro
        weights = torch.softmax(logits, dim=-1)

        if k >= n_models:
            return weights

        # Encontrar limiar top-K
        topk_vals, _ = torch.topk(weights, k, dim=-1)
        threshold = topk_vals[:, -1:].detach()  # (batch, 1)

        # Mascarar modelos abaixo do limiar
        mask = weights >= threshold
        weights = weights * mask.float()

        # Renormalizar para somar 1
        total = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights / total

        return weights


class GatingNetwork(nn.Module):
    """Rede MLP que mapeia contexto de mercado para pesos de modelos.

    Arquitetura:
        Linear(n_context, 128) -> GELU -> Dropout ->
        Linear(128, 64) -> GELU ->
        Linear(64, n_models) -> TopKSoftmax
    """

    def __init__(
        self,
        n_context_features: int,
        n_models: int,
        top_k: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_context_features = n_context_features
        self.n_models = n_models

        self.mlp = nn.Sequential(
            nn.Linear(n_context_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_models),
        )
        self.top_k_softmax = TopKSoftmax(top_k=top_k)

        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializa pesos com Xavier uniform para estabilidade."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Calcula pesos de gating a partir de features de contexto.

        Args:
            context: Tensor de shape (batch, n_context_features).

        Returns:
            Tensor de shape (batch, n_models) com pesos esparsos.
        """
        logits = self.mlp(context)
        return self.top_k_softmax(logits)


class MoEEnsemble(BaseModel):
    """Mixture of Experts Ensemble que aprende roteamento baseado em mercado.

    Este modelo opera SOBRE as previsoes de outros modelos. Recebe uma
    matriz de previsoes de N modelos + features de contexto de mercado,
    e produz uma previsao ponderada unica via pesos de gating aprendidos.

    Args:
        n_models: Numero de modelos base no ensemble.
        top_k: Numero de modelos a ativar por amostra.
        n_epochs: Numero maximo de epocas de treinamento.
        lr: Taxa de aprendizado para Adam.
        patience: Epocas sem melhoria para early stopping.
        diversity_lambda: Peso da regularizacao de diversidade.
        dropout: Taxa de dropout na rede de gating.
        device: Dispositivo PyTorch ('cpu', 'cuda', etc.).
    """

    N_CONTEXT_FEATURES = 10

    def __init__(
        self,
        n_models: int = 20,
        top_k: int = 6,
        n_epochs: int = 200,
        lr: float = 1e-3,
        patience: int = 20,
        diversity_lambda: float = 0.01,
        dropout: float = 0.2,
        device: str | None = None,
    ):
        self.n_models = n_models
        self.top_k = top_k
        self.n_epochs = n_epochs
        self.lr = lr
        self.patience = patience
        self.diversity_lambda = diversity_lambda
        self.dropout = dropout

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gating_network: GatingNetwork | None = None
        self.context_scaler = StandardScaler()
        self._is_fitted = False
        self._train_metrics: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "moe_ensemble"

    # ------------------------------------------------------------------
    # Extracao de contexto de mercado
    # ------------------------------------------------------------------

    def _extract_market_context(self, X_features: np.ndarray) -> np.ndarray:
        """Extrai ~10 sinais de estado de mercado a partir de features brutas.

        Tenta extrair sinais significativos das features disponiveis.
        Se o array de entrada tiver menos colunas que o necessario,
        preenche com zeros.

        Args:
            X_features: Array de shape (n_samples, n_features) com features brutas
                        OU (n_samples, N_CONTEXT_FEATURES) se ja extraidas.

        Returns:
            Array de shape (n_samples, N_CONTEXT_FEATURES) com contexto de mercado.
        """
        n_samples = X_features.shape[0]
        n_cols = X_features.shape[1] if X_features.ndim > 1 else 1

        if X_features.ndim == 1:
            X_features = X_features.reshape(-1, 1)

        # Se ja tem exatamente N_CONTEXT_FEATURES, assume que ja esta extraido
        if n_cols == self.N_CONTEXT_FEATURES:
            return X_features.copy()

        context = np.zeros((n_samples, self.N_CONTEXT_FEATURES), dtype=np.float32)

        try:
            # Feature 0: Volatilidade (desvio padrao rolling dos retornos)
            if n_cols > 0:
                col = X_features[:, 0]
                returns = np.diff(col, prepend=col[0])
                window = min(20, n_samples)
                vol = np.array([
                    np.std(returns[max(0, i - window):i + 1])
                    for i in range(n_samples)
                ])
                context[:, 0] = vol

            # Feature 1: Forca de tendencia (momentum)
            if n_cols > 0:
                col = X_features[:, 0]
                window = min(10, n_samples)
                momentum = np.array([
                    col[i] - col[max(0, i - window)]
                    for i in range(n_samples)
                ])
                context[:, 1] = momentum

            # Feature 2: Regime de volume (acima/abaixo da media)
            if n_cols > 1:
                vol_col = X_features[:, 1]
                vol_mean = np.mean(vol_col) if np.mean(vol_col) != 0 else 1.0
                context[:, 2] = vol_col / max(abs(vol_mean), 1e-8)
            elif n_cols > 0:
                context[:, 2] = np.abs(X_features[:, 0])

            # Feature 3: Sinal de reversao a media
            if n_cols > 0:
                col = X_features[:, 0]
                window = min(20, n_samples)
                rolling_mean = np.array([
                    np.mean(col[max(0, i - window):i + 1])
                    for i in range(n_samples)
                ])
                mean_rev = col - rolling_mean
                std_val = np.std(mean_rev)
                context[:, 3] = mean_rev / max(std_val, 1e-8)

            # Feature 4: Aceleracao de preco (segunda derivada)
            if n_cols > 0:
                col = X_features[:, 0]
                d1 = np.diff(col, prepend=col[0])
                d2 = np.diff(d1, prepend=d1[0])
                context[:, 4] = d2

            # Feature 5: Razao high-low (proxy de range)
            if n_cols > 2:
                context[:, 5] = X_features[:, 2]
            elif n_cols > 0:
                context[:, 5] = np.abs(np.diff(X_features[:, 0], prepend=X_features[0, 0]))

            # Feature 6: RSI zone (normalizado)
            if n_cols > 3:
                context[:, 6] = X_features[:, 3]
            elif n_cols > 0:
                col = X_features[:, 0]
                returns = np.diff(col, prepend=col[0])
                window = min(14, n_samples)
                rsi = np.zeros(n_samples)
                for i in range(n_samples):
                    chunk = returns[max(0, i - window):i + 1]
                    gains = np.mean(chunk[chunk > 0]) if np.any(chunk > 0) else 0.0
                    losses = -np.mean(chunk[chunk < 0]) if np.any(chunk < 0) else 0.0
                    if losses == 0:
                        rsi[i] = 1.0
                    else:
                        rs = gains / losses
                        rsi[i] = (2.0 * rs / (1.0 + rs)) - 1.0  # Normalizado [-1, 1]
                context[:, 6] = rsi

            # Feature 7: Correlacao com BTC (se disponivel, senao auto-correlacao)
            if n_cols > 4:
                context[:, 7] = X_features[:, 4]
            elif n_cols > 0:
                col = X_features[:, 0]
                lag = min(5, n_samples - 1)
                if lag > 0:
                    corr = np.corrcoef(col[lag:], col[:-lag])[0, 1]
                    context[:, 7] = np.full(n_samples, corr if np.isfinite(corr) else 0.0)

            # Feature 8: Skewness dos retornos recentes
            if n_cols > 0:
                col = X_features[:, 0]
                returns = np.diff(col, prepend=col[0])
                window = min(20, n_samples)
                from scipy.stats import skew as _skew  # noqa: F811
                skew_vals = np.array([
                    _skew(returns[max(0, i - window):i + 1])
                    for i in range(n_samples)
                ])
                context[:, 8] = np.nan_to_num(skew_vals, nan=0.0)

            # Feature 9: Kurtosis dos retornos recentes
            if n_cols > 0:
                col = X_features[:, 0]
                returns = np.diff(col, prepend=col[0])
                window = min(20, n_samples)
                from scipy.stats import kurtosis as _kurt  # noqa: F811
                kurt_vals = np.array([
                    _kurt(returns[max(0, i - window):i + 1])
                    for i in range(n_samples)
                ])
                context[:, 9] = np.nan_to_num(kurt_vals, nan=0.0)

        except Exception as e:
            logger.warning("Erro parcial ao extrair contexto de mercado: %s", e)
            # Fallback: usar as primeiras N colunas diretamente
            n_use = min(n_cols, self.N_CONTEXT_FEATURES)
            context[:, :n_use] = X_features[:, :n_use]

        # Substituir NaN/Inf
        context = np.nan_to_num(context, nan=0.0, posinf=1e6, neginf=-1e6)
        return context

    # ------------------------------------------------------------------
    # Utilitarios de treinamento
    # ------------------------------------------------------------------

    def _diversity_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Penaliza concentracao excessiva num unico modelo.

        Usa entropia negativa: quanto menor a entropia, maior a penalidade.

        Args:
            weights: Tensor de shape (batch, n_models) com pesos de gating.

        Returns:
            Escalar com a penalidade de diversidade (negativo da entropia media).
        """
        # Entropia: H = -sum(w * log(w))
        eps = 1e-8
        entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)
        # Queremos maximizar entropia -> minimizar entropia negativa
        return -entropy.mean()

    def _prepare_tensors(
        self,
        predictions: np.ndarray,
        context: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Converte arrays numpy em tensores PyTorch no dispositivo correto."""
        preds_t = torch.tensor(predictions, dtype=torch.float32, device=self.device)
        ctx_t = torch.tensor(context, dtype=torch.float32, device=self.device)
        tgt_t = None
        if targets is not None:
            tgt_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
        return preds_t, ctx_t, tgt_t

    # ------------------------------------------------------------------
    # Interface BaseModel
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Treina a rede de gating sobre previsoes de validacao.

        X_train deve ser uma matriz concatenada de:
            - Previsoes dos N modelos: shape (n_samples, n_models)
            - Features de contexto de mercado: shape (n_samples, n_context)
        Concatenados horizontalmente: shape (n_samples, n_models + n_context)

        Alternativamente, se X_train tem shape (n_samples, n_models), as
        features de contexto sao extraidas internamente.

        Args:
            X_train: Matriz de previsoes + contexto.
            y_train: Alvos reais.
            X_val: Dados de validacao (mesmo formato que X_train).
            y_val: Alvos de validacao.

        Returns:
            Dicionario com metricas de treinamento.
        """
        logger.info("Iniciando treinamento do MoE Ensemble com %d modelos", self.n_models)

        try:
            # Separar previsoes e contexto
            train_preds, train_context = self._split_predictions_context(X_train)
            n_models_actual = train_preds.shape[1]

            if n_models_actual != self.n_models:
                logger.info(
                    "Ajustando n_models de %d para %d (detectado no input)",
                    self.n_models,
                    n_models_actual,
                )
                self.n_models = n_models_actual

            # Extrair e normalizar contexto
            market_ctx = self._extract_market_context(train_context)
            market_ctx = self.context_scaler.fit_transform(market_ctx)

            # Preparar validacao
            val_preds_np = None
            val_ctx_np = None
            val_targets_np = None
            if X_val is not None and y_val is not None:
                val_preds_np, val_context_raw = self._split_predictions_context(X_val)
                val_ctx_np = self.context_scaler.transform(
                    self._extract_market_context(val_context_raw)
                )
                val_targets_np = y_val

            # Inicializar rede de gating
            self.gating_network = GatingNetwork(
                n_context_features=self.N_CONTEXT_FEATURES,
                n_models=self.n_models,
                top_k=self.top_k,
                dropout=self.dropout,
            ).to(self.device)

            # Preparar tensores
            preds_t, ctx_t, tgt_t = self._prepare_tensors(
                train_preds, market_ctx, y_train
            )

            optimizer = torch.optim.Adam(
                self.gating_network.parameters(), lr=self.lr
            )
            mse_loss = nn.MSELoss()

            best_val_loss = float("inf")
            best_state = None
            epochs_no_improve = 0

            self.gating_network.train()

            for epoch in range(self.n_epochs):
                optimizer.zero_grad()

                # Forward pass
                weights = self.gating_network(ctx_t)  # (n_samples, n_models)
                gated_pred = (weights * preds_t).sum(dim=-1)  # (n_samples,)

                # Loss principal
                loss = mse_loss(gated_pred, tgt_t)

                # Regularizacao de diversidade
                if self.diversity_lambda > 0:
                    div_loss = self.diversity_loss = self._diversity_loss(weights)
                    loss = loss + self.diversity_lambda * div_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.gating_network.parameters(), max_norm=1.0
                )
                optimizer.step()

                # Validacao
                if val_preds_np is not None:
                    val_loss = self._evaluate(
                        val_preds_np, val_ctx_np, val_targets_np
                    )
                else:
                    val_loss = loss.item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.gating_network.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if (epoch + 1) % 50 == 0:
                    logger.info(
                        "Epoca %d/%d - Loss treino: %.6f - Loss val: %.6f",
                        epoch + 1,
                        self.n_epochs,
                        loss.item(),
                        val_loss,
                    )

                if epochs_no_improve >= self.patience:
                    logger.info(
                        "Early stopping na epoca %d (paciencia=%d)",
                        epoch + 1,
                        self.patience,
                    )
                    break

            # Restaurar melhor modelo
            if best_state is not None:
                self.gating_network.load_state_dict(best_state)
                self.gating_network.to(self.device)

            self._is_fitted = True

            # Calcular metricas finais
            self.gating_network.eval()
            with torch.no_grad():
                weights = self.gating_network(ctx_t)
                final_pred = (weights * preds_t).sum(dim=-1)
                train_mse = mse_loss(final_pred, tgt_t).item()

            self._train_metrics = {
                "train_mse": train_mse,
                "best_val_loss": best_val_loss,
                "epochs_trained": epoch + 1,
                "n_models": self.n_models,
                "top_k": self.top_k,
            }

            logger.info(
                "Treinamento MoE concluido - MSE treino: %.6f, MSE val: %.6f, Epocas: %d",
                train_mse,
                best_val_loss,
                epoch + 1,
            )

            return self._train_metrics

        except Exception as e:
            logger.error("Erro durante treinamento do MoE Ensemble: %s", e)
            raise

    def _split_predictions_context(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Separa a matriz de entrada em previsoes de modelos e features de contexto.

        Se X tem mais colunas que n_models, as colunas extras sao contexto.
        Se X tem exatamente n_models colunas, gera contexto sintetico.

        Returns:
            (predictions, context_features)
        """
        n_cols = X.shape[1] if X.ndim > 1 else 1

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if n_cols > self.n_models:
            predictions = X[:, :self.n_models]
            context = X[:, self.n_models:]
        elif n_cols == self.n_models:
            predictions = X
            # Gerar contexto a partir das proprias previsoes
            context = self._context_from_predictions(predictions)
        else:
            # Menos colunas que modelos esperados - preencher
            logger.warning(
                "Input com %d colunas, esperado >= %d. Preenchendo com zeros.",
                n_cols,
                self.n_models,
            )
            predictions = np.zeros((X.shape[0], self.n_models), dtype=np.float32)
            predictions[:, :n_cols] = X
            context = self._context_from_predictions(predictions)

        return predictions, context

    def _context_from_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Gera features de contexto a partir da matriz de previsoes.

        Quando nao ha features de mercado separadas, extrai estatisticas
        da propria distribuicao de previsoes dos modelos.
        """
        n_samples = predictions.shape[0]
        context = np.zeros((n_samples, self.N_CONTEXT_FEATURES), dtype=np.float32)

        # Estatisticas das previsoes entre modelos
        context[:, 0] = np.std(predictions, axis=1)          # Dispersao
        context[:, 1] = np.mean(predictions, axis=1)         # Consenso
        context[:, 2] = np.median(predictions, axis=1)       # Mediana
        context[:, 3] = np.max(predictions, axis=1) - np.min(predictions, axis=1)  # Range
        context[:, 4] = np.percentile(predictions, 75, axis=1) - np.percentile(predictions, 25, axis=1)  # IQR

        # Skewness da distribuicao de previsoes
        mean = context[:, 1:2]
        std = np.maximum(context[:, 0:1], 1e-8)
        centered = predictions - mean
        context[:, 5] = np.mean((centered / std) ** 3, axis=1)

        # Kurtosis
        context[:, 6] = np.mean((centered / std) ** 4, axis=1) - 3.0

        # Fracao de modelos com previsao positiva
        context[:, 7] = np.mean(predictions > 0, axis=1)

        # Diferenca media-mediana (assimetria)
        context[:, 8] = context[:, 1] - context[:, 2]

        # Coeficiente de variacao
        context[:, 9] = context[:, 0] / np.maximum(np.abs(context[:, 1]), 1e-8)

        return np.nan_to_num(context, nan=0.0, posinf=1e6, neginf=-1e6)

    def _evaluate(
        self,
        predictions: np.ndarray,
        context: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Avalia loss no conjunto de validacao."""
        self.gating_network.eval()
        with torch.no_grad():
            preds_t, ctx_t, tgt_t = self._prepare_tensors(
                predictions, context, targets
            )
            weights = self.gating_network(ctx_t)
            gated_pred = (weights * preds_t).sum(dim=-1)
            loss = nn.functional.mse_loss(gated_pred, tgt_t).item()
        self.gating_network.train()
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Gera previsoes ponderadas pelo gating.

        Args:
            X: Matriz de previsoes dos modelos + contexto (mesmo formato do fit).

        Returns:
            Array de previsoes ponderadas de shape (n_samples,).
        """
        if not self._is_fitted or self.gating_network is None:
            raise RuntimeError("Modelo MoE nao foi treinado. Chame fit() primeiro.")

        try:
            predictions, context_raw = self._split_predictions_context(X)
            market_ctx = self._extract_market_context(context_raw)
            market_ctx = self.context_scaler.transform(market_ctx)

            self.gating_network.eval()
            with torch.no_grad():
                preds_t, ctx_t, _ = self._prepare_tensors(predictions, market_ctx)
                weights = self.gating_network(ctx_t)
                gated_pred = (weights * preds_t).sum(dim=-1)

            return gated_pred.cpu().numpy()

        except Exception as e:
            logger.error("Erro durante predicao do MoE Ensemble: %s", e)
            raise

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gera previsoes com confianca baseada na entropia do gating.

        A confianca e derivada da entropia dos pesos de gating:
        - Baixa entropia (pesos concentrados) -> alta confianca
        - Alta entropia (pesos dispersos) -> baixa confianca

        Args:
            X: Matriz de previsoes + contexto.

        Returns:
            (predictions, confidence_scores) onde confidence esta em [0, 1].
        """
        if not self._is_fitted or self.gating_network is None:
            raise RuntimeError("Modelo MoE nao foi treinado. Chame fit() primeiro.")

        try:
            predictions, context_raw = self._split_predictions_context(X)
            market_ctx = self._extract_market_context(context_raw)
            market_ctx = self.context_scaler.transform(market_ctx)

            self.gating_network.eval()
            with torch.no_grad():
                preds_t, ctx_t, _ = self._prepare_tensors(predictions, market_ctx)
                weights = self.gating_network(ctx_t)
                gated_pred = (weights * preds_t).sum(dim=-1)

                # Entropia dos pesos
                eps = 1e-8
                entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)

                # Normalizar: entropia maxima = log(top_k)
                max_entropy = np.log(min(self.top_k, self.n_models))
                if max_entropy > 0:
                    confidence = 1.0 - (entropy / max_entropy).clamp(0.0, 1.0)
                else:
                    confidence = torch.ones_like(entropy)

            return gated_pred.cpu().numpy(), confidence.cpu().numpy()

        except Exception as e:
            logger.error("Erro durante predicao com confianca do MoE: %s", e)
            raise

    def get_gating_weights(self, X: np.ndarray) -> np.ndarray:
        """Retorna os pesos de gating para inspecao/debug.

        Args:
            X: Matriz de previsoes + contexto.

        Returns:
            Array de shape (n_samples, n_models) com pesos de gating.
        """
        if not self._is_fitted or self.gating_network is None:
            raise RuntimeError("Modelo MoE nao foi treinado. Chame fit() primeiro.")

        predictions, context_raw = self._split_predictions_context(X)
        market_ctx = self._extract_market_context(context_raw)
        market_ctx = self.context_scaler.transform(market_ctx)

        self.gating_network.eval()
        with torch.no_grad():
            ctx_t = torch.tensor(
                market_ctx, dtype=torch.float32, device=self.device
            )
            weights = self.gating_network(ctx_t)

        return weights.cpu().numpy()

    def save(self, path: Path) -> None:
        """Salva o modelo MoE em disco.

        Salva:
            - Estado da rede de gating
            - Scaler de contexto
            - Hiperparametros
        """
        if not self._is_fitted or self.gating_network is None:
            raise RuntimeError("Modelo MoE nao foi treinado. Nada para salvar.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import pickle

        state = {
            "gating_state_dict": self.gating_network.state_dict(),
            "context_scaler": pickle.dumps(self.context_scaler),
            "n_models": self.n_models,
            "top_k": self.top_k,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "patience": self.patience,
            "diversity_lambda": self.diversity_lambda,
            "dropout": self.dropout,
            "train_metrics": self._train_metrics,
        }

        torch.save(state, path)
        logger.info("Modelo MoE salvo em %s", path)

    @classmethod
    def load(cls, path: Path) -> "MoEEnsemble":
        """Carrega o modelo MoE do disco.

        Args:
            path: Caminho para o arquivo salvo.

        Returns:
            Instancia de MoEEnsemble com pesos carregados.
        """
        import pickle

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de modelo nao encontrado: {path}")

        state = torch.load(path, map_location="cpu", weights_only=False)

        instance = cls(
            n_models=state["n_models"],
            top_k=state["top_k"],
            n_epochs=state["n_epochs"],
            lr=state["lr"],
            patience=state["patience"],
            diversity_lambda=state["diversity_lambda"],
            dropout=state["dropout"],
        )

        instance.gating_network = GatingNetwork(
            n_context_features=cls.N_CONTEXT_FEATURES,
            n_models=state["n_models"],
            top_k=state["top_k"],
            dropout=state["dropout"],
        ).to(instance.device)

        instance.gating_network.load_state_dict(state["gating_state_dict"])
        instance.context_scaler = pickle.loads(state["context_scaler"])
        instance._train_metrics = state.get("train_metrics", {})
        instance._is_fitted = True

        logger.info("Modelo MoE carregado de %s", path)
        return instance
