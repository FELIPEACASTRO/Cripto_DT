"""Transfer Entropy entre criptomoedas — causalidade informacional.

Transfer Entropy mede a quantidade de informação que uma série temporal
transfere para outra, capturando relações de causalidade não-linear.

TE(X→Y) alto: X ajuda a prever Y (X lidera Y)
TE(BTC→ALT) >> TE(ALT→BTC): BTC lidera a altcoin

Referência: Schreiber (2000) "Measuring Information Transfer"
           Dimpfl & Peter (2021) "Transfer Entropy in Crypto Markets"
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _discretize(series: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Discretiza série contínua em bins para cálculo de entropia.

    Args:
        series: Série temporal contínua
        n_bins: Número de bins

    Returns:
        Série discretizada (inteiros 0 a n_bins-1)
    """
    # Usar percentis para bins adaptativos
    percentiles = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(series, percentiles)
    thresholds[-1] += 1e-10  # Garantir que o máximo cai no último bin
    discretized = np.digitize(series, thresholds[1:])
    return np.clip(discretized, 0, n_bins - 1)


def _conditional_entropy(x: np.ndarray, y: np.ndarray, n_bins: int = 5) -> float:
    """Calcula H(X|Y) — entropia condicional.

    Args:
        x: Série alvo discretizada
        y: Série condicionante discretizada

    Returns:
        Entropia condicional H(X|Y)
    """
    n = len(x)
    if n == 0:
        return 0.0

    # Calcular distribuição conjunta
    joint = np.zeros((n_bins, n_bins))
    for i in range(n):
        xi = min(int(x[i]), n_bins - 1)
        yi = min(int(y[i]), n_bins - 1)
        joint[xi, yi] += 1

    joint /= n + 1e-10

    # P(Y)
    py = np.sum(joint, axis=0)

    # H(X|Y) = -sum P(x,y) * log(P(x|y))
    h = 0.0
    for xi in range(n_bins):
        for yi in range(n_bins):
            if joint[xi, yi] > 0 and py[yi] > 0:
                p_xy = joint[xi, yi]
                p_x_given_y = p_xy / py[yi]
                h -= p_xy * np.log2(p_x_given_y + 1e-10)

    return h


def _transfer_entropy(
    source: np.ndarray, target: np.ndarray,
    lag: int = 1, n_bins: int = 5,
) -> float:
    """Calcula Transfer Entropy de source → target.

    TE(X→Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})

    Se TE > 0: X ajuda a prever Y (X → Y causalidade informacional)

    Args:
        source: Série fonte (X)
        target: Série alvo (Y)
        lag: Defasagem temporal
        n_bins: Número de bins para discretização

    Returns:
        Transfer Entropy (float >= 0)
    """
    n = len(source)
    if n < lag + 10:
        return 0.0

    # Discretizar
    src_d = _discretize(source, n_bins)
    tgt_d = _discretize(target, n_bins)

    # Séries alinhadas
    y_t = tgt_d[lag:]         # Y no tempo t
    y_past = tgt_d[:-lag]     # Y no tempo t-lag
    x_past = src_d[:-lag]     # X no tempo t-lag

    # H(Y_t | Y_{t-lag})
    h_y_given_ypast = _conditional_entropy(y_t, y_past, n_bins)

    # H(Y_t | Y_{t-lag}, X_{t-lag}) — aproximação via entropia condicional combinada
    # Combinamos y_past e x_past em um único estado
    combined_past = y_past * n_bins + x_past
    combined_past = _discretize(combined_past.astype(float), n_bins)
    h_y_given_both = _conditional_entropy(y_t, combined_past, n_bins)

    te = max(0, h_y_given_ypast - h_y_given_both)
    return te


class TransferEntropyFeatures:
    """Gera features de Transfer Entropy entre criptomoedas.

    Features geradas por par de moedas:
    - te_{source}_{target}_{lag}: TE de source para target com lag
    - net_te_{pair}_{lag}: TE líquida (TE(X→Y) - TE(Y→X))
    - te_btc_influence_{lag}: TE de BTC para moeda alvo

    Features agregadas:
    - te_market_influence: soma de TE de todas moedas para target
    - te_leader_score: quão influente é a moeda no mercado
    - te_information_ratio: razão TE recebida / TE emitida
    """

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 3, 5),
        n_bins: int = 5,
        rolling_window: int = 60,
    ):
        self.lags = lags
        self.n_bins = n_bins
        self.rolling_window = rolling_window

    def transform(
        self,
        coin_data: dict[str, pd.DataFrame],
        target_coin: str,
    ) -> pd.DataFrame:
        """Adiciona features de Transfer Entropy para a moeda alvo.

        Args:
            coin_data: Dict {moeda: DataFrame} com 'log_return' ou 'close'
            target_coin: Nome da moeda alvo

        Returns:
            DataFrame da moeda alvo com TE features
        """
        if target_coin not in coin_data:
            logger.warning(f"Moeda {target_coin} não encontrada")
            return pd.DataFrame()

        df = coin_data[target_coin].copy()
        n = len(df)

        # Obter retornos da moeda alvo
        target_returns = self._get_returns(df)
        if target_returns is None:
            return df

        other_coins = [c for c in coin_data.keys() if c != target_coin]

        if not other_coins:
            return df

        logger.info(
            f"Transfer Entropy: calculando para {target_coin} vs {len(other_coins)} moedas"
        )

        te_received = []  # TE total recebida

        for source_coin in other_coins:
            source_df = coin_data[source_coin]
            source_returns = self._get_returns(source_df)

            if source_returns is None:
                continue

            # Alinhar séries pelo comprimento mínimo
            min_len = min(len(target_returns), len(source_returns))
            tgt = target_returns[:min_len]
            src = source_returns[:min_len]

            for lag in self.lags:
                # TE rolling
                te_values = np.zeros(n)
                net_te_values = np.zeros(n)

                for i in range(self.rolling_window, min_len):
                    window_src = src[i - self.rolling_window:i]
                    window_tgt = tgt[i - self.rolling_window:i]

                    te_s2t = _transfer_entropy(window_src, window_tgt, lag, self.n_bins)
                    te_t2s = _transfer_entropy(window_tgt, window_src, lag, self.n_bins)

                    te_values[i] = te_s2t
                    net_te_values[i] = te_s2t - te_t2s

                # Guardar apenas features agregadas (evitar explosão de features)
                if source_coin == "BTC":
                    df[f"te_btc_influence_lag{lag}"] = te_values[:n]
                    df[f"te_btc_net_lag{lag}"] = net_te_values[:n]

                te_received.append(te_values[:n])

        # Features agregadas
        if te_received:
            te_matrix = np.array(te_received)
            df["te_total_received"] = np.mean(te_matrix, axis=0)
            df["te_max_influence"] = np.max(te_matrix, axis=0)
            df["te_influence_std"] = np.std(te_matrix, axis=0)

            # Information ratio: quão "liderada" é a moeda
            total_received = np.sum(te_matrix, axis=0)
            df["te_follower_score"] = np.where(
                total_received > 0,
                total_received / (np.max(total_received) + 1e-10),
                0
            )
        else:
            df["te_total_received"] = 0.0
            df["te_max_influence"] = 0.0
            df["te_influence_std"] = 0.0
            df["te_follower_score"] = 0.0

        # Preencher NaN
        te_cols = [c for c in df.columns if c.startswith("te_")]
        for col in te_cols:
            df[col] = pd.Series(df[col]).ffill().fillna(0).values

        logger.info(f"Transfer Entropy: {len(te_cols)} features adicionadas")

        return df

    def _get_returns(self, df: pd.DataFrame) -> np.ndarray | None:
        """Obtém retornos de um DataFrame."""
        if "log_return" in df.columns:
            return df["log_return"].fillna(0).values
        elif "close" in df.columns:
            r = np.log(df["close"] / df["close"].shift(1)).fillna(0).values
            return r
        return None
