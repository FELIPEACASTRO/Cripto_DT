"""EMGNN: Evolving Multiscale Graph Neural Network para previsao cross-asset.

Captura relacoes dinamicas entre criptomoedas em multiplas escalas temporais
usando grafos de correlacao e atencao, combinados com GRU para modelagem
temporal.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicGraphBuilder:
    """Constroi grafos de adjacencia dinamicos baseados em correlacao de retornos."""

    @staticmethod
    def build_correlation_graph(
        returns_matrix: np.ndarray,
        window: int = 20,
        threshold: float = 0.3,
    ) -> np.ndarray:
        """Constroi grafo de adjacencia a partir de correlacao rolling.

        Args:
            returns_matrix: (timesteps, n_nodes) - retornos de cada ativo
            window: janela de correlacao
            threshold: correlacao minima para criar aresta

        Returns:
            adj_matrix: (n_nodes, n_nodes) - matriz de adjacencia normalizada
        """
        n_nodes = returns_matrix.shape[1]

        # Usar ultimos `window` timesteps
        if returns_matrix.shape[0] >= window:
            recent = returns_matrix[-window:]
        else:
            recent = returns_matrix

        # Correlacao de Pearson
        corr = np.corrcoef(recent.T)
        corr = np.nan_to_num(corr, nan=0.0)

        # Aplicar threshold e manter apenas correlacoes positivas fortes
        adj = np.abs(corr)
        adj[adj < threshold] = 0.0

        # Remover auto-loops
        np.fill_diagonal(adj, 0.0)

        # Normalizar por grau (D^-1/2 * A * D^-1/2)
        degree = adj.sum(axis=1)
        degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        return adj_norm.astype(np.float32)

    @staticmethod
    def build_multiscale_graphs(
        returns_matrix: np.ndarray,
        windows: list[int] | None = None,
        threshold: float = 0.3,
    ) -> list[np.ndarray]:
        """Constroi grafos de adjacencia em multiplas escalas temporais.

        Args:
            returns_matrix: (timesteps, n_nodes)
            windows: lista de janelas temporais (default: [5, 10, 20])
            threshold: correlacao minima para criar aresta

        Returns:
            Lista de matrizes de adjacencia, uma por escala
        """
        if windows is None:
            windows = [5, 10, 20]

        graphs = []
        for w in windows:
            adj = DynamicGraphBuilder.build_correlation_graph(
                returns_matrix, window=w, threshold=threshold
            )
            graphs.append(adj)

        return graphs


class GraphAttentionLayer(nn.Module):
    """Camada de atencao em grafo (single-head GAT).

    Computa coeficientes de atencao entre nos vizinhos e realiza
    agregacao ponderada das features.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Vetor de atencao: concatenacao de features dos dois nos
        self.attn = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn.weight)

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass da camada GAT.

        Args:
            h: (batch, n_nodes, in_features) - features dos nos
            adj: (batch, n_nodes, n_nodes) ou (n_nodes, n_nodes) - adjacencia

        Returns:
            h_out: (batch, n_nodes, out_features)
        """
        batch_size, n_nodes, _ = h.shape

        # Projecao linear
        Wh = self.W(h)  # (batch, n_nodes, out_features)

        # Computar coeficientes de atencao para todos os pares
        # Wh_i: (batch, n_nodes, 1, out_features) -> repetido para cada j
        # Wh_j: (batch, 1, n_nodes, out_features) -> repetido para cada i
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, n_nodes, -1, -1)

        # Concatenar pares (i, j)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)  # (batch, n, n, 2*out)
        e = self.leaky_relu(self.attn(concat).squeeze(-1))  # (batch, n, n)

        # Mascarar com adjacencia (vizinhos ausentes -> -inf)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))

        # Softmax normaliza coeficientes de atencao
        attention = F.softmax(e, dim=-1)
        # Onde tudo era -inf, softmax produz nan -> substituir por 0
        attention = torch.nan_to_num(attention, nan=0.0)
        attention = self.dropout(attention)

        # Agregacao ponderada
        h_out = torch.bmm(attention, Wh)  # (batch, n_nodes, out_features)

        return h_out


class EMGNNNet(nn.Module):
    """Evolving Multiscale Graph Neural Network.

    Arquitetura:
        1. Multi-escala: GraphAttentionLayer para cada escala temporal
        2. Temporal: GRU processa a sequencia de snapshots
        3. Fusao: Concatena saidas multi-escala -> FC -> previsao por no
    """

    def __init__(
        self,
        n_nodes: int,
        input_size: int,
        hidden_size: int = 64,
        n_scales: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_scales = n_scales

        # Projecao de input
        self.input_proj = nn.Linear(input_size, hidden_size)

        # GAT por escala
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_size, hidden_size, dropout=dropout)
            for _ in range(n_scales)
        ])

        # Normalizacao por escala
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(n_scales)
        ])

        # GRU temporal (processa sequencia de representacoes de grafo)
        self.gru = nn.GRU(
            input_size=hidden_size * n_scales,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Camada de saida: previsao por no
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj_matrices: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_nodes, input_size) - features temporais por no
            adj_matrices: lista de n_scales tensores, cada um:
                - (batch, n_nodes, n_nodes) ou (n_nodes, n_nodes)

        Returns:
            predictions: (batch, n_nodes) - previsao por ativo
        """
        batch_size, seq_len, n_nodes, _ = x.shape

        # Processar cada timestep
        temporal_features = []

        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch, n_nodes, input_size)
            h_t = F.gelu(self.input_proj(x_t))  # (batch, n_nodes, hidden_size)

            # Multi-escala: aplicar GAT com cada grafo
            scale_outputs = []
            for s in range(self.n_scales):
                adj_s = adj_matrices[s]
                h_s = self.gat_layers[s](h_t, adj_s)
                h_s = self.layer_norms[s](h_s + h_t)  # Residual + LayerNorm
                # Pooling por no: media sobre todos os nos -> representacao global
                # Mantemos por no para previsao per-node
                scale_outputs.append(h_s)  # (batch, n_nodes, hidden_size)

            # Concatenar saidas multi-escala
            multi_scale = torch.cat(scale_outputs, dim=-1)  # (batch, n_nodes, hidden*n_scales)

            # Media sobre nos para obter representacao global do timestep
            graph_repr = multi_scale.mean(dim=1)  # (batch, hidden*n_scales)
            temporal_features.append(graph_repr)

        # Stack temporal e passar pela GRU
        temporal_seq = torch.stack(temporal_features, dim=1)  # (batch, seq_len, hidden*n_scales)
        gru_out, _ = self.gru(temporal_seq)  # (batch, seq_len, hidden)
        last_hidden = gru_out[:, -1, :]  # (batch, hidden)

        # Expandir para cada no e combinar com features do ultimo timestep
        last_x = x[:, -1, :, :]  # (batch, n_nodes, input_size)
        last_h = F.gelu(self.input_proj(last_x))  # (batch, n_nodes, hidden)

        # Somar representacao temporal global com features locais por no
        global_expanded = last_hidden.unsqueeze(1).expand(-1, n_nodes, -1)
        node_repr = last_h + global_expanded  # (batch, n_nodes, hidden)

        # Previsao por no
        node_repr = self.dropout(node_repr)
        predictions = self.fc_out(node_repr).squeeze(-1)  # (batch, n_nodes)

        return predictions


class EMGNNModel(BaseModel):
    """Wrapper de treino/inferencia para o EMGNN."""

    def __init__(
        self,
        n_nodes: int = 10,
        input_size: int | None = None,
        hidden_size: int = 64,
        n_scales: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 100,
        early_stop_patience: int = 10,
        grad_clip_max_norm: float = 1.0,
        mc_dropout_samples: int = 30,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        graph_windows: list[int] | None = None,
        correlation_threshold: float = 0.3,
    ):
        self.n_nodes = n_nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_scales = n_scales
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
        self.graph_windows = graph_windows or [5, 10, 20]
        self.correlation_threshold = correlation_threshold
        self.model: EMGNNNet | None = None
        # Armazenar ultimas adjacencias para inferencia
        self._last_adj_matrices: list[torch.Tensor] | None = None

    @property
    def name(self) -> str:
        return "emgnn"

    def _build_model(self, input_size: int) -> EMGNNNet:
        self.input_size = input_size
        return EMGNNNet(
            n_nodes=self.n_nodes,
            input_size=input_size,
            hidden_size=self.hidden_size,
            n_scales=self.n_scales,
            dropout=self.dropout,
        ).to(DEVICE)

    @staticmethod
    def _adj_to_tensor(adj_list: list[np.ndarray]) -> list[torch.Tensor]:
        """Converte lista de adjacencias numpy para tensores no device."""
        return [torch.FloatTensor(adj).to(DEVICE) for adj in adj_list]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        adj_matrices_train: list[np.ndarray] | None = None,
        adj_matrices_val: list[np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Treina o EMGNN.

        Args:
            X_train: (samples, seq_len, n_nodes, input_size) - 4D
            y_train: (samples, n_nodes) - retornos por ativo
            X_val: dados de validacao (mesmo formato)
            y_val: targets de validacao
            adj_matrices_train: lista de n_scales matrizes (n_nodes, n_nodes)
            adj_matrices_val: adjacencias para validacao (se diferente)
        """
        if X_train.ndim != 4:
            raise ValueError(
                f"EMGNN espera X_train 4D (samples, seq_len, n_nodes, features), "
                f"recebeu shape {X_train.shape}"
            )

        self.n_nodes = X_train.shape[2]
        input_size = X_train.shape[3]
        self.model = self._build_model(input_size)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"EMGNN inicializado: {total_params:,} parametros, device={DEVICE}")

        # Construir grafos default se nao fornecidos
        if adj_matrices_train is None:
            adj_matrices_train = self._build_default_adj(X_train)

        adj_tensors = self._adj_to_tensor(adj_matrices_train)
        self._last_adj_matrices = adj_tensors

        # Tensores de treino
        X_train_t = torch.FloatTensor(X_train).to(DEVICE)
        y_train_t = torch.FloatTensor(y_train).to(DEVICE)

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
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Validacao
        X_val_t, y_val_t = None, None
        adj_val_tensors = adj_tensors  # Usa mesmos grafos por default
        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(DEVICE)
            y_val_t = torch.FloatTensor(y_val).to(DEVICE)
            if adj_matrices_val is not None:
                adj_val_tensors = self._adj_to_tensor(adj_matrices_val)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # Treino
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self.model(X_batch, adj_tensors)
                loss = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            # Validacao
            if X_val_t is not None and y_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val_t, adj_val_tensors)
                    val_loss = criterion(val_preds, y_val_t).item()

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
                    logger.info(f"  EMGNN early stopping na epoca {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                msg = f"  EMGNN epoca {epoch + 1}: train_loss={avg_train_loss:.6f}"
                if X_val_t is not None:
                    msg += f" val_loss={val_loss:.6f}"
                logger.info(msg)

        # Restaurar melhor modelo
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Metricas finais
        metrics: dict[str, float] = {"train_loss": avg_train_loss}
        if X_val_t is not None and y_val_t is not None:
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t, adj_val_tensors).cpu().numpy()
                y_val_np = y_val_t.cpu().numpy()
            metrics["val_loss"] = float(best_val_loss)
            metrics["val_rmse"] = float(np.sqrt(np.mean((val_preds - y_val_np) ** 2)))
            # Acuracia direcional media sobre todos os nos
            metrics["val_dir_acc"] = float(
                np.mean(np.sign(val_preds) == np.sign(y_val_np))
            )

        logger.info(f"  EMGNN metricas: {metrics}")
        return metrics

    def _build_default_adj(self, X: np.ndarray) -> list[np.ndarray]:
        """Constroi grafos de adjacencia default a partir dos dados.

        Usa a media temporal das features como proxy de retornos para correlacao.
        """
        # X: (samples, seq_len, n_nodes, features)
        # Usar feature 0 (tipicamente retorno) como base para correlacao
        returns_proxy = X[:, -1, :, 0]  # (samples, n_nodes) - ultimo timestep, feature 0
        return DynamicGraphBuilder.build_multiscale_graphs(
            returns_proxy,
            windows=self.graph_windows,
            threshold=self.correlation_threshold,
        )

    def predict(
        self,
        X: np.ndarray,
        adj_matrices: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Gera previsoes para cada no (ativo).

        Args:
            X: (samples, seq_len, n_nodes, input_size)
            adj_matrices: grafos de adjacencia (usa ultimos do treino se None)

        Returns:
            predictions: (samples, n_nodes)
        """
        if self.model is None:
            raise RuntimeError("Modelo EMGNN nao treinado")

        self.model.eval()
        X_t = torch.FloatTensor(X).to(DEVICE)

        if adj_matrices is not None:
            adj_tensors = self._adj_to_tensor(adj_matrices)
        elif self._last_adj_matrices is not None:
            adj_tensors = self._last_adj_matrices
        else:
            adj_tensors = self._adj_to_tensor(self._build_default_adj(X))

        with torch.no_grad():
            return self.model(X_t, adj_tensors).cpu().numpy()

    def predict_with_confidence(
        self,
        X: np.ndarray,
        adj_matrices: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC Dropout para estimativa de incerteza.

        Returns:
            (mean_predictions, confidence): cada (samples, n_nodes)
        """
        if self.model is None:
            raise RuntimeError("Modelo EMGNN nao treinado")

        X_t = torch.FloatTensor(X).to(DEVICE)

        if adj_matrices is not None:
            adj_tensors = self._adj_to_tensor(adj_matrices)
        elif self._last_adj_matrices is not None:
            adj_tensors = self._last_adj_matrices
        else:
            adj_tensors = self._adj_to_tensor(self._build_default_adj(X))

        self.model.train()  # Ativa dropout

        predictions = []
        for _ in range(self.mc_dropout_samples):
            with torch.no_grad():
                pred = self.model(X_t, adj_tensors).cpu().numpy()
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
            raise RuntimeError("Modelo EMGNN nao treinado")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Converter adjacencias para numpy para salvar
        adj_numpy = None
        if self._last_adj_matrices is not None:
            adj_numpy = [adj.cpu().numpy() for adj in self._last_adj_matrices]

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "n_nodes": self.n_nodes,
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "n_scales": self.n_scales,
                "dropout": self.dropout,
                "graph_windows": self.graph_windows,
                "correlation_threshold": self.correlation_threshold,
                "adj_matrices": adj_numpy,
            },
            path,
        )
        logger.info(f"EMGNN salvo: {path}")

    @classmethod
    def load(cls, path: Path) -> "EMGNNModel":
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        instance = cls(
            n_nodes=checkpoint["n_nodes"],
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            n_scales=checkpoint["n_scales"],
            dropout=checkpoint["dropout"],
            graph_windows=checkpoint["graph_windows"],
            correlation_threshold=checkpoint["correlation_threshold"],
        )
        instance.model = EMGNNNet(
            n_nodes=checkpoint["n_nodes"],
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            n_scales=checkpoint["n_scales"],
            dropout=checkpoint["dropout"],
        ).to(DEVICE)
        instance.model.load_state_dict(checkpoint["model_state"])

        # Restaurar adjacencias
        if checkpoint.get("adj_matrices") is not None:
            instance._last_adj_matrices = cls._adj_to_tensor(
                checkpoint["adj_matrices"]
            )

        return instance
