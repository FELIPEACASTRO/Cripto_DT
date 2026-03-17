"""MarketGAN para geracao de series temporais financeiras sinteticas.

Implementa Wasserstein GAN com gradient penalty (WGAN-GP) para gerar
sequencias sinteticas de retornos OHLCV, com foco em eventos raros:
crashes, pumps e black swans. Inspirado por arXiv:2601.17773.

A geracao condicional permite especificar regime de mercado
(normal/crash/pump/volatil), possibilitando augmentacao direcionada
para treinar modelos mais robustos em cenarios extremos.
"""

import logging
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class MarketRegime(IntEnum):
    """Regimes de mercado para geracao condicional."""

    NORMAL = 0
    CRASH = 1
    PUMP = 2
    VOLATILE = 3


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Gerador baseado em TCN para series temporais financeiras.

    Recebe um vetor de ruido z (+ embedding opcional de regime) e produz
    uma sequencia sintetica de retornos OHLCV.

    Arquitetura: Linear -> Reshape -> 3x(Conv1d + BatchNorm + GELU)
                 -> Linear -> Tanh
    """

    def __init__(
        self,
        latent_dim: int = 32,
        seq_len: int = 60,
        n_features: int = 5,
        n_regimes: int = 4,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_regimes = n_regimes

        # Embedding de regime condicional
        self.regime_embed = nn.Embedding(n_regimes, 8)
        input_dim = latent_dim + 8

        # Projecao inicial: mapeia ruido + regime para espaco intermediario
        self.fc_in = nn.Linear(input_dim, hidden_channels * seq_len)

        # Blocos convolucionais TCN-like
        self.conv_blocks = nn.Sequential(
            # Bloco 1
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            # Bloco 2
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            # Bloco 3
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )

        # Projecao final para features OHLCV
        self.fc_out = nn.Linear(hidden_channels, n_features)
        self.tanh = nn.Tanh()

    def forward(
        self,
        z: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gera sequencia sintetica a partir de ruido + regime.

        Args:
            z: Vetor de ruido (batch, latent_dim).
            regime: Label de regime (batch,). Se None, usa NORMAL.

        Returns:
            Sequencia sintetica (batch, seq_len, n_features) em [-1, 1].
        """
        batch_size = z.size(0)

        if regime is None:
            regime = torch.zeros(batch_size, dtype=torch.long, device=z.device)

        regime_emb = self.regime_embed(regime)  # (batch, 8)
        x = torch.cat([z, regime_emb], dim=1)   # (batch, latent_dim + 8)

        # Projecao e reshape para formato conv: (batch, channels, seq_len)
        x = self.fc_in(x)  # (batch, hidden_channels * seq_len)
        x = x.view(batch_size, -1, self.seq_len)  # (batch, hidden_channels, seq_len)

        # Blocos convolucionais
        x = self.conv_blocks(x)  # (batch, hidden_channels, seq_len)

        # Transpor para (batch, seq_len, hidden_channels) e projetar
        x = x.transpose(1, 2)
        x = self.fc_out(x)  # (batch, seq_len, n_features)
        x = self.tanh(x)

        return x


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """Discriminador 1D-CNN para sequencias de retornos OHLCV.

    Arquitetura: 3x(Conv1d + LeakyReLU + Dropout) -> Flatten -> Linear -> Sigmoid
    """

    def __init__(
        self,
        seq_len: int = 60,
        n_features: int = 5,
        hidden_channels: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        self.conv_blocks = nn.Sequential(
            # Bloco 1: n_features -> hidden_channels
            nn.Conv1d(n_features, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # Bloco 2: hidden_channels -> hidden_channels * 2
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # Bloco 3: hidden_channels * 2 -> hidden_channels * 2
            nn.Conv1d(
                hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * 2 * seq_len, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifica sequencia como real ou falsa.

        Args:
            x: Sequencia de retornos (batch, seq_len, n_features).

        Returns:
            Probabilidade real/fake (batch, 1).
        """
        # Transpor para formato conv: (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# MarketGAN
# ---------------------------------------------------------------------------

class MarketGAN:
    """WGAN-GP completo para geracao de dados sinteticos de mercado.

    Treina gerador e discriminador com loss de Wasserstein + gradient penalty
    para estabilidade. Suporta geracao condicional por regime de mercado.

    Exemplo de uso::

        gan = MarketGAN(latent_dim=32, seq_len=60)
        gan.fit(df_ohlcv, epochs=500)
        synthetic = gan.generate(n=200)
        crashes = gan.generate_crash_scenarios(n=100)
        augmented_df = gan.augment_training_data(df_ohlcv, n_synthetic=500)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        seq_len: int = 60,
        n_features: int = 5,
        hidden_channels: int = 64,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        batch_size: int = 64,
        n_critic: int = 5,
        gp_lambda: float = 10.0,
        device: Optional[torch.device] = None,
    ):
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.n_features = n_features
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.device = device or DEVICE

        # Redes
        self.generator = Generator(
            latent_dim=latent_dim,
            seq_len=seq_len,
            n_features=n_features,
            hidden_channels=hidden_channels,
        ).to(self.device)

        self.discriminator = Discriminator(
            seq_len=seq_len,
            n_features=n_features,
            hidden_channels=hidden_channels,
        ).to(self.device)

        # Otimizadores Adam com betas para WGAN-GP
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(0.0, 0.9)
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9)
        )

        # Estatisticas de normalizacao (preenchidas por prepare_training_data)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._trained = False

        logger.info(
            "MarketGAN inicializada: latent_dim=%d, seq_len=%d, device=%s",
            latent_dim,
            seq_len,
            self.device,
        )

    # ------------------------------------------------------------------
    # Processamento de dados
    # ------------------------------------------------------------------

    def prepare_training_data(
        self, df: pd.DataFrame
    ) -> torch.Tensor:
        """Converte DataFrame OHLCV em sequencias de retornos normalizados.

        Calcula log-retornos, normaliza por z-score e cria janelas deslizantes
        de tamanho seq_len.

        Args:
            df: DataFrame com colunas OHLCV (open, high, low, close, volume).

        Returns:
            Tensor de sequencias (n_sequences, seq_len, n_features).

        Raises:
            ValueError: Se colunas obrigatorias estiverem ausentes ou dados
                        insuficientes.
        """
        df_cols = [c.lower() for c in df.columns]
        missing = [c for c in OHLCV_COLUMNS if c not in df_cols]
        if missing:
            raise ValueError(
                f"Colunas OHLCV ausentes no DataFrame: {missing}. "
                f"Colunas disponiveis: {list(df.columns)}"
            )

        df_work = df.copy()
        df_work.columns = [c.lower() for c in df_work.columns]
        ohlcv = df_work[OHLCV_COLUMNS].astype(float)

        # Log-retornos (evita divisao por zero)
        returns = np.log(ohlcv / ohlcv.shift(1).replace(0, np.nan)).dropna()

        if len(returns) < self.seq_len + 1:
            raise ValueError(
                f"Dados insuficientes: {len(returns)} linhas apos log-retornos, "
                f"minimo necessario: {self.seq_len + 1}"
            )

        values = returns.values  # (n_rows, 5)

        # Z-score normalizacao
        self._mean = values.mean(axis=0)
        self._std = values.std(axis=0)
        self._std[self._std < 1e-8] = 1e-8  # evita divisao por zero
        normalized = (values - self._mean) / self._std

        # Janelas deslizantes
        sequences = []
        for i in range(len(normalized) - self.seq_len + 1):
            sequences.append(normalized[i : i + self.seq_len])

        tensor = torch.tensor(np.array(sequences), dtype=torch.float32)

        logger.info(
            "Dados de treino preparados: %d sequencias de %d passos x %d features",
            tensor.shape[0],
            tensor.shape[1],
            tensor.shape[2],
        )

        return tensor

    def _assign_regimes(self, sequences: torch.Tensor) -> torch.Tensor:
        """Classifica cada sequencia em um regime de mercado.

        Usa retorno acumulado e volatilidade da coluna close (indice 3)
        para atribuir labels de regime.

        Args:
            sequences: (n_sequences, seq_len, n_features).

        Returns:
            Tensor de labels (n_sequences,).
        """
        close_returns = sequences[:, :, 3].numpy()  # coluna close
        cum_returns = close_returns.sum(axis=1)
        volatilities = close_returns.std(axis=1)

        vol_median = np.median(volatilities)
        labels = np.full(len(sequences), MarketRegime.NORMAL, dtype=np.int64)

        # Crash: retorno acumulado < percentil 10
        crash_threshold = np.percentile(cum_returns, 10)
        labels[cum_returns < crash_threshold] = MarketRegime.CRASH

        # Pump: retorno acumulado > percentil 90
        pump_threshold = np.percentile(cum_returns, 90)
        labels[cum_returns > pump_threshold] = MarketRegime.PUMP

        # Volatil: volatilidade alta mas retorno nao extremo
        vol_mask = (volatilities > vol_median * 1.5) & (
            (labels == MarketRegime.NORMAL)
        )
        labels[vol_mask] = MarketRegime.VOLATILE

        regime_counts = {
            r.name: int((labels == r.value).sum()) for r in MarketRegime
        }
        logger.info("Distribuicao de regimes no treino: %s", regime_counts)

        return torch.tensor(labels, dtype=torch.long)

    # ------------------------------------------------------------------
    # WGAN-GP Training
    # ------------------------------------------------------------------

    def _gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula gradient penalty para WGAN-GP.

        Interpola entre amostras reais e geradas, calcula gradiente do
        discriminador na interpolacao e penaliza desvio da norma L2 = 1.

        Args:
            real: Amostras reais (batch, seq_len, n_features).
            fake: Amostras geradas (batch, seq_len, n_features).

        Returns:
            Gradient penalty escalar.
        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        alpha = alpha.expand_as(real)

        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interpolated = self.discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1.0) ** 2).mean()

        return penalty

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 500,
        verbose: bool = True,
        log_interval: int = 50,
    ) -> dict:
        """Treina a GAN em dados OHLCV reais.

        Usa treinamento WGAN-GP: n_critic passos do discriminador por cada
        passo do gerador, com gradient penalty para estabilidade.

        Args:
            df: DataFrame com colunas OHLCV.
            epochs: Numero de epocas de treinamento.
            verbose: Se True, loga metricas periodicamente.
            log_interval: Intervalo de epocas para logging.

        Returns:
            Dicionario com historico de losses (d_loss, g_loss, gp).
        """
        logger.info("Iniciando treinamento MarketGAN por %d epocas...", epochs)

        sequences = self.prepare_training_data(df)
        regimes = self._assign_regimes(sequences)

        dataset = TensorDataset(sequences, regimes)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        history = {"d_loss": [], "g_loss": [], "gp": []}

        self.generator.train()
        self.discriminator.train()

        for epoch in range(epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_gp = 0.0
            n_batches = 0

            for real_batch, regime_batch in dataloader:
                real_batch = real_batch.to(self.device)
                regime_batch = regime_batch.to(self.device)
                current_batch = real_batch.size(0)

                # ---- Treinar Discriminador (Critico) ----
                for _ in range(self.n_critic):
                    self.opt_d.zero_grad()

                    z = torch.randn(
                        current_batch, self.latent_dim, device=self.device
                    )
                    fake_batch = self.generator(z, regime_batch).detach()

                    d_real = self.discriminator(real_batch)
                    d_fake = self.discriminator(fake_batch)

                    # Wasserstein loss: maximizar E[D(real)] - E[D(fake)]
                    # Equivale a minimizar E[D(fake)] - E[D(real)]
                    gp = self._gradient_penalty(real_batch, fake_batch)
                    d_loss = d_fake.mean() - d_real.mean() + self.gp_lambda * gp

                    d_loss.backward()
                    self.opt_d.step()

                # ---- Treinar Gerador ----
                self.opt_g.zero_grad()

                z = torch.randn(current_batch, self.latent_dim, device=self.device)
                fake_batch = self.generator(z, regime_batch)
                d_fake = self.discriminator(fake_batch)

                # Gerador quer maximizar D(fake) => minimizar -D(fake)
                g_loss = -d_fake.mean()
                g_loss.backward()
                self.opt_g.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_gp += gp.item()
                n_batches += 1

            if n_batches > 0:
                avg_d = epoch_d_loss / n_batches
                avg_g = epoch_g_loss / n_batches
                avg_gp = epoch_gp / n_batches
                history["d_loss"].append(avg_d)
                history["g_loss"].append(avg_g)
                history["gp"].append(avg_gp)

                if verbose and (epoch + 1) % log_interval == 0:
                    logger.info(
                        "Epoca %d/%d - D_loss: %.4f | G_loss: %.4f | GP: %.4f",
                        epoch + 1,
                        epochs,
                        avg_d,
                        avg_g,
                        avg_gp,
                    )

        self._trained = True
        logger.info("Treinamento MarketGAN concluido com sucesso.")

        return history

    def _check_trained(self) -> None:
        """Verifica se o modelo foi treinado antes da geracao."""
        if not self._trained:
            raise RuntimeError(
                "MarketGAN nao foi treinada. Chame fit() antes de gerar dados."
            )

    # ------------------------------------------------------------------
    # Geracao
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        n: int = 100,
        regime: Optional[MarketRegime] = None,
    ) -> np.ndarray:
        """Gera N sequencias sinteticas.

        Args:
            n: Numero de sequencias a gerar.
            regime: Regime de mercado para condicionar. Se None, gera
                    distribuicao mista.

        Returns:
            Array (n, seq_len, n_features) de retornos normalizados.
        """
        self._check_trained()
        self.generator.eval()

        z = torch.randn(n, self.latent_dim, device=self.device)

        if regime is not None:
            regime_labels = torch.full(
                (n,), regime.value, dtype=torch.long, device=self.device
            )
        else:
            regime_labels = torch.randint(
                0, len(MarketRegime), (n,), device=self.device
            )

        synthetic = self.generator(z, regime_labels)
        result = synthetic.cpu().numpy()

        logger.info(
            "Geradas %d sequencias sinteticas (regime=%s)",
            n,
            regime.name if regime else "MISTO",
        )

        return result

    @torch.no_grad()
    def generate_crash_scenarios(self, n: int = 100) -> np.ndarray:
        """Gera cenarios sinteticos de crash.

        Injeta vies negativo e alta volatilidade no espaco latente para
        produzir sequencias que simulam quedas de mercado.

        Args:
            n: Numero de cenarios de crash a gerar.

        Returns:
            Array (n, seq_len, n_features) de retornos de crash.
        """
        self._check_trained()
        self.generator.eval()

        # Ruido com vies negativo e dispersao alta
        z = torch.randn(n, self.latent_dim, device=self.device)
        z = z * 1.5 - 0.8  # alta volatilidade + vies negativo

        regime_labels = torch.full(
            (n,), MarketRegime.CRASH, dtype=torch.long, device=self.device
        )

        synthetic = self.generator(z, regime_labels)
        result = synthetic.cpu().numpy()

        logger.info("Gerados %d cenarios de crash sinteticos", n)

        return result

    @torch.no_grad()
    def generate_pump_scenarios(self, n: int = 100) -> np.ndarray:
        """Gera cenarios sinteticos de pump.

        Injeta vies positivo e alta volatilidade no espaco latente para
        produzir sequencias que simulam rallies explosivos.

        Args:
            n: Numero de cenarios de pump a gerar.

        Returns:
            Array (n, seq_len, n_features) de retornos de pump.
        """
        self._check_trained()
        self.generator.eval()

        # Ruido com vies positivo e dispersao alta
        z = torch.randn(n, self.latent_dim, device=self.device)
        z = z * 1.5 + 0.8  # alta volatilidade + vies positivo

        regime_labels = torch.full(
            (n,), MarketRegime.PUMP, dtype=torch.long, device=self.device
        )

        synthetic = self.generator(z, regime_labels)
        result = synthetic.cpu().numpy()

        logger.info("Gerados %d cenarios de pump sinteticos", n)

        return result

    @torch.no_grad()
    def generate_black_swan(self, n: int = 50) -> np.ndarray:
        """Gera eventos de cauda extrema (black swan, >3 sigma).

        Usa ruido latente extremo para produzir sequencias que representam
        eventos com probabilidade muito baixa mas alto impacto.

        Args:
            n: Numero de eventos black swan a gerar.

        Returns:
            Array (n, seq_len, n_features) de retornos extremos.
        """
        self._check_trained()
        self.generator.eval()

        # Ruido extremo: truncado para >3 sigma em magnitude
        z = torch.randn(n, self.latent_dim, device=self.device)
        # Amplifica para regime de cauda
        z = z * 2.5
        # Garante que componentes tenham magnitude >3 sigma
        sign = z.sign()
        z = sign * torch.clamp(z.abs(), min=3.0)

        # Metade crashes extremos, metade pumps extremos
        regime_labels = torch.zeros(n, dtype=torch.long, device=self.device)
        n_half = n // 2
        regime_labels[:n_half] = MarketRegime.CRASH
        regime_labels[n_half:] = MarketRegime.PUMP

        synthetic = self.generator(z, regime_labels)
        result = synthetic.cpu().numpy()

        logger.info("Gerados %d eventos black swan sinteticos", n)

        return result

    # ------------------------------------------------------------------
    # Conversao de volta para DataFrame
    # ------------------------------------------------------------------

    def to_dataframe(self, synthetic_sequences: np.ndarray) -> pd.DataFrame:
        """Converte sequencias sinteticas de volta para formato OHLCV.

        Desnormaliza retornos, acumula para reconstruir precos e retorna
        DataFrame com colunas OHLCV realistas.

        Args:
            synthetic_sequences: Array (n, seq_len, n_features) de retornos.

        Returns:
            DataFrame com colunas OHLCV reconstruidas.

        Raises:
            RuntimeError: Se estatisticas de normalizacao nao estiverem
                          disponiveis.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "Estatisticas de normalizacao indisponiveis. "
                "Chame fit() ou prepare_training_data() primeiro."
            )

        # Desnormalizar retornos
        denorm = synthetic_sequences * self._std + self._mean

        all_rows = []
        base_price = 100.0  # preco base arbitrario

        for seq_idx in range(denorm.shape[0]):
            seq = denorm[seq_idx]  # (seq_len, 5)

            # Reconstruir precos a partir de log-retornos
            prices = np.zeros_like(seq)
            prices[0] = base_price
            for t in range(1, seq.shape[0]):
                prices[t] = prices[t - 1] * np.exp(seq[t])

            # Garantir consistencia OHLCV: high >= open,close; low <= open,close
            for t in range(seq.shape[0]):
                o, h, l, c, v = prices[t]
                h = max(h, o, c)
                l = min(l, o, c)
                if l <= 0:
                    l = min(o, c) * 0.99
                v = abs(v) * 1e6  # escalar volume para faixa realista
                prices[t] = [o, h, l, c, v]

            for t in range(seq.shape[0]):
                all_rows.append(
                    {
                        "sequence_id": seq_idx,
                        "timestep": t,
                        "open": prices[t, 0],
                        "high": prices[t, 1],
                        "low": prices[t, 2],
                        "close": prices[t, 3],
                        "volume": prices[t, 4],
                    }
                )

        df = pd.DataFrame(all_rows)

        logger.info(
            "Convertidas %d sequencias sinteticas para DataFrame OHLCV (%d linhas)",
            denorm.shape[0],
            len(df),
        )

        return df

    # ------------------------------------------------------------------
    # Augmentacao principal
    # ------------------------------------------------------------------

    def augment_training_data(
        self,
        df: pd.DataFrame,
        n_synthetic: int = 500,
        crash_ratio: float = 0.3,
        pump_ratio: float = 0.2,
        black_swan_ratio: float = 0.1,
    ) -> pd.DataFrame:
        """Metodo principal para augmentar dados de treino existentes.

        Gera sequencias sinteticas com distribuicao ponderada de regimes,
        enfatizando eventos raros (crashes, pumps, black swans).

        Args:
            df: DataFrame OHLCV original.
            n_synthetic: Numero total de sequencias sinteticas.
            crash_ratio: Fracao de cenarios de crash.
            pump_ratio: Fracao de cenarios de pump.
            black_swan_ratio: Fracao de eventos black swan.

        Returns:
            DataFrame combinando dados originais e sinteticos.

        Raises:
            ValueError: Se ratios somarem mais que 1.0.
        """
        total_ratio = crash_ratio + pump_ratio + black_swan_ratio
        if total_ratio > 1.0:
            raise ValueError(
                f"Soma dos ratios ({total_ratio:.2f}) excede 1.0. "
                f"Ajuste crash_ratio, pump_ratio e/ou black_swan_ratio."
            )

        self._check_trained()

        n_crash = int(n_synthetic * crash_ratio)
        n_pump = int(n_synthetic * pump_ratio)
        n_swan = int(n_synthetic * black_swan_ratio)
        n_normal = n_synthetic - n_crash - n_pump - n_swan

        logger.info(
            "Gerando %d sequencias sinteticas: %d normal, %d crash, "
            "%d pump, %d black swan",
            n_synthetic,
            n_normal,
            n_crash,
            n_pump,
            n_swan,
        )

        parts = []

        if n_normal > 0:
            parts.append(self.generate(n_normal, regime=MarketRegime.NORMAL))
        if n_crash > 0:
            parts.append(self.generate_crash_scenarios(n_crash))
        if n_pump > 0:
            parts.append(self.generate_pump_scenarios(n_pump))
        if n_swan > 0:
            parts.append(self.generate_black_swan(n_swan))

        all_synthetic = np.concatenate(parts, axis=0)
        synthetic_df = self.to_dataframe(all_synthetic)
        synthetic_df["source"] = "synthetic"

        # Preparar DataFrame original
        original_df = df.copy()
        original_df["source"] = "real"
        if "sequence_id" not in original_df.columns:
            original_df["sequence_id"] = -1
        if "timestep" not in original_df.columns:
            original_df["timestep"] = range(len(original_df))

        combined = pd.concat([original_df, synthetic_df], ignore_index=True)

        logger.info(
            "Augmentacao concluida: %d linhas originais + %d linhas sinteticas "
            "= %d total",
            len(original_df),
            len(synthetic_df),
            len(combined),
        )

        return combined

    # ------------------------------------------------------------------
    # Validacao estatistica
    # ------------------------------------------------------------------

    def validate_statistics(
        self,
        real_sequences: np.ndarray,
        synthetic_sequences: np.ndarray,
    ) -> dict:
        """Valida que dados sinteticos preservam propriedades estatisticas.

        Verifica:
        - Caudas pesadas (kurtosis similar)
        - Volatility clustering (autocorrelacao de retornos absolutos)
        - Correlacoes entre colunas OHLCV

        Args:
            real_sequences: Array de sequencias reais.
            synthetic_sequences: Array de sequencias sinteticas.

        Returns:
            Dicionario com metricas de validacao.
        """
        from scipy import stats as sp_stats

        results = {}

        # Flatten para comparar distribuicoes marginais
        real_flat = real_sequences.reshape(-1, self.n_features)
        syn_flat = synthetic_sequences.reshape(-1, self.n_features)

        # Kurtosis por feature (caudas pesadas)
        real_kurt = sp_stats.kurtosis(real_flat, axis=0)
        syn_kurt = sp_stats.kurtosis(syn_flat, axis=0)
        results["kurtosis_real"] = dict(zip(OHLCV_COLUMNS, real_kurt.tolist()))
        results["kurtosis_synthetic"] = dict(zip(OHLCV_COLUMNS, syn_kurt.tolist()))
        results["kurtosis_diff"] = dict(
            zip(OHLCV_COLUMNS, (np.abs(real_kurt - syn_kurt)).tolist())
        )

        # Correlacao entre colunas
        real_corr = np.corrcoef(real_flat.T)
        syn_corr = np.corrcoef(syn_flat.T)
        corr_diff = np.abs(real_corr - syn_corr).mean()
        results["mean_correlation_diff"] = float(corr_diff)

        # Volatility clustering: autocorrelacao de |retornos| (coluna close)
        def _autocorr_abs(data: np.ndarray, lag: int = 1) -> float:
            abs_data = np.abs(data)
            if len(abs_data) < lag + 2:
                return 0.0
            c = np.corrcoef(abs_data[:-lag], abs_data[lag:])[0, 1]
            return float(c) if np.isfinite(c) else 0.0

        real_close = real_flat[:, 3]
        syn_close = syn_flat[:, 3]
        results["vol_clustering_real"] = _autocorr_abs(real_close)
        results["vol_clustering_synthetic"] = _autocorr_abs(syn_close)
        results["vol_clustering_diff"] = abs(
            results["vol_clustering_real"] - results["vol_clustering_synthetic"]
        )

        # Kolmogorov-Smirnov test por feature
        ks_results = {}
        for i, col in enumerate(OHLCV_COLUMNS):
            stat, pval = sp_stats.ks_2samp(real_flat[:, i], syn_flat[:, i])
            ks_results[col] = {"statistic": float(stat), "p_value": float(pval)}
        results["ks_test"] = ks_results

        # Avaliacao geral
        quality_score = 1.0 - min(corr_diff, 1.0)
        results["quality_score"] = float(quality_score)

        logger.info(
            "Validacao estatistica: score=%.3f, corr_diff=%.4f, "
            "vol_clustering_diff=%.4f",
            quality_score,
            corr_diff,
            results["vol_clustering_diff"],
        )

        return results

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Salva estado completo do MarketGAN.

        Args:
            path: Caminho do arquivo .pt para salvar.
        """
        state = {
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "opt_g_state": self.opt_g.state_dict(),
            "opt_d_state": self.opt_d.state_dict(),
            "mean": self._mean,
            "std": self._std,
            "trained": self._trained,
            "config": {
                "latent_dim": self.latent_dim,
                "seq_len": self.seq_len,
                "n_features": self.n_features,
                "lr_g": self.lr_g,
                "lr_d": self.lr_d,
                "batch_size": self.batch_size,
                "n_critic": self.n_critic,
                "gp_lambda": self.gp_lambda,
            },
        }
        torch.save(state, path)
        logger.info("MarketGAN salva em %s", path)

    def load(self, path: str) -> None:
        """Carrega estado completo do MarketGAN.

        Args:
            path: Caminho do arquivo .pt para carregar.

        Raises:
            FileNotFoundError: Se o arquivo nao existir.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.generator.load_state_dict(state["generator_state"])
        self.discriminator.load_state_dict(state["discriminator_state"])
        self.opt_g.load_state_dict(state["opt_g_state"])
        self.opt_d.load_state_dict(state["opt_d_state"])
        self._mean = state["mean"]
        self._std = state["std"]
        self._trained = state["trained"]

        logger.info("MarketGAN carregada de %s", path)
