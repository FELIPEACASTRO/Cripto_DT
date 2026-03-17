"""Features de microestrutura de mercado para previsao de criptomoedas.

Implementa indicadores avancados baseados em pesquisa de microestrutura:
- Expoente de Hurst (persistencia de tendencia)
- Dimensao Fractal (complexidade da serie)
- VPIN estimado (probabilidade de trading informado)
- Amihud Illiquidity Ratio
- Kyle's Lambda estimado (impacto de mercado)
- Roll's Spread estimado
- Realized Volatility Signature Plot features
- Return autocorrelation features

Referencia: Easley, Lopez de Prado & O'Hara (2012) - VPIN
           Amihud (2002) - Illiquidity ratio
           Mandelbrot (1963) - Fractal analysis
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """Calcula o expoente de Hurst via R/S analysis.

    H > 0.5: tendencia persistente (trending)
    H = 0.5: random walk
    H < 0.5: mean-reverting

    Args:
        series: Serie temporal de retornos
        max_lag: Lag maximo para analise R/S

    Returns:
        Expoente de Hurst (float entre 0 e 1)
    """
    n = len(series)
    if n < max_lag * 2:
        return 0.5

    lags = range(2, min(max_lag + 1, n // 2))
    rs_values = []
    lag_values = []

    for lag in lags:
        # Dividir a serie em subseries de tamanho lag
        n_subseries = n // lag
        if n_subseries < 1:
            continue

        rs_lag = []
        for i in range(n_subseries):
            subseries = series[i * lag:(i + 1) * lag]
            mean_sub = np.mean(subseries)
            deviations = np.cumsum(subseries - mean_sub)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(subseries, ddof=1) if len(subseries) > 1 else 1e-10
            if s > 0:
                rs_lag.append(r / s)

        if rs_lag:
            rs_values.append(np.mean(rs_lag))
            lag_values.append(lag)

    if len(rs_values) < 3:
        return 0.5

    # Regressao log-log para estimar H
    log_lags = np.log(lag_values)
    log_rs = np.log(np.array(rs_values) + 1e-10)

    # Least squares
    n_pts = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_lags * log_rs)
    sum_x2 = np.sum(log_lags ** 2)

    denom = n_pts * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 0.5

    h = (n_pts * sum_xy - sum_x * sum_y) / denom
    return np.clip(h, 0.0, 1.0)


def _fractal_dimension(series: np.ndarray, k_max: int = 10) -> float:
    """Calcula dimensao fractal via metodo de Higuchi.

    D proximo de 1: serie suave (trending)
    D proximo de 2: serie complexa (ruidosa)

    Args:
        series: Serie temporal
        k_max: Escala maxima

    Returns:
        Dimensao fractal (float entre 1 e 2)
    """
    n = len(series)
    if n < k_max * 4:
        return 1.5

    lk = []
    x = []

    for k in range(1, min(k_max + 1, n // 4)):
        lm_k = []
        for m in range(1, k + 1):
            # Construir sub-serie
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            sub = series[indices]
            # Comprimento da curva
            length = np.sum(np.abs(np.diff(sub)))
            norm_length = length * (n - 1) / (len(indices) * k)
            lm_k.append(norm_length)

        if lm_k:
            lk.append(np.mean(lm_k))
            x.append(k)

    if len(lk) < 3:
        return 1.5

    log_x = np.log(1.0 / np.array(x))
    log_y = np.log(np.array(lk) + 1e-10)

    # Regressao linear
    n_pts = len(log_x)
    sum_x = np.sum(log_x)
    sum_y = np.sum(log_y)
    sum_xy = np.sum(log_x * log_y)
    sum_x2 = np.sum(log_x ** 2)

    denom = n_pts * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 1.5

    d = (n_pts * sum_xy - sum_x * sum_y) / denom
    return np.clip(d, 1.0, 2.0)


def _vpin_estimate(
    volume: np.ndarray, returns: np.ndarray, bucket_size: int = 10
) -> np.ndarray:
    """Estima VPIN (Volume-Synchronized Probability of Informed Trading).

    Usa classificacao de volume via tick rule (sinal do retorno).
    VPIN alto = alta probabilidade de trading informado = possivel movimento brusco.

    Args:
        volume: Serie de volumes
        returns: Serie de retornos
        bucket_size: Numero de periodos por bucket

    Returns:
        Array de VPIN estimado
    """
    n = len(volume)
    vpin = np.full(n, np.nan)

    if n < bucket_size * 3:
        return vpin

    # Classificar volume como buy/sell pelo sinal do retorno
    buy_volume = np.where(returns > 0, volume, 0)
    sell_volume = np.where(returns < 0, volume, 0)

    # Calcular order imbalance rolling
    for i in range(bucket_size, n):
        window_buy = np.sum(buy_volume[i - bucket_size:i])
        window_sell = np.sum(sell_volume[i - bucket_size:i])
        total_vol = window_buy + window_sell
        if total_vol > 0:
            vpin[i] = abs(window_buy - window_sell) / total_vol

    return vpin


def _amihud_illiquidity(
    returns: np.ndarray, volume: np.ndarray, window: int = 20
) -> np.ndarray:
    """Calcula Amihud Illiquidity Ratio rolling.

    ILLIQ = |return| / volume_dollar
    Alto = iliquido (movimentos de preco por unidade de volume)

    Args:
        returns: Serie de retornos absolutos
        volume: Serie de volumes em USD
        window: Janela rolling

    Returns:
        Array de illiquidity ratio
    """
    n = len(returns)
    illiq = np.full(n, np.nan)

    abs_returns = np.abs(returns)

    for i in range(window, n):
        r_window = abs_returns[i - window:i]
        v_window = volume[i - window:i]
        # Evitar divisao por zero
        valid = v_window > 0
        if valid.sum() > 0:
            ratios = r_window[valid] / v_window[valid]
            illiq[i] = np.mean(ratios)

    return illiq


def _kyle_lambda(
    returns: np.ndarray, volume: np.ndarray, window: int = 20
) -> np.ndarray:
    """Estima Kyle's Lambda (impacto de mercado).

    Lambda = Cov(return, signed_volume) / Var(signed_volume)
    Alto = forte impacto de volume no preco

    Args:
        returns: Serie de retornos
        volume: Serie de volumes
        window: Janela rolling

    Returns:
        Array de Kyle's Lambda estimado
    """
    n = len(returns)
    kyle = np.full(n, np.nan)

    # Volume com sinal (positivo se retorno positivo)
    signed_volume = np.sign(returns) * volume

    for i in range(window, n):
        r_w = returns[i - window:i]
        sv_w = signed_volume[i - window:i]
        var_sv = np.var(sv_w)
        if var_sv > 0:
            cov = np.cov(r_w, sv_w)[0, 1]
            kyle[i] = cov / var_sv

    return kyle


def _roll_spread(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Estima Roll's Spread (bid-ask spread implicito).

    Spread = 2 * sqrt(-Cov(r_t, r_{t-1})) se cov < 0
    Mede custo implicito de transacao.

    Args:
        returns: Serie de retornos
        window: Janela rolling

    Returns:
        Array de spread estimado
    """
    n = len(returns)
    spread = np.full(n, np.nan)

    for i in range(window + 1, n):
        r_current = returns[i - window:i]
        r_lagged = returns[i - window - 1:i - 1]
        cov = np.cov(r_current, r_lagged)[0, 1]
        if cov < 0:
            spread[i] = 2.0 * np.sqrt(-cov)
        else:
            spread[i] = 0.0

    return spread


class MicrostructureFeatures:
    """Gera features de microestrutura de mercado.

    Features geradas:
    - hurst_{window}: Expoente de Hurst rolling
    - fractal_dim_{window}: Dimensao fractal rolling
    - vpin_{bucket}: VPIN estimado
    - amihud_{window}: Illiquidity ratio
    - kyle_lambda_{window}: Impacto de mercado
    - roll_spread_{window}: Bid-ask spread implicito
    - return_autocorr_{lag}: Autocorrelacao de retornos
    - volume_return_corr_{window}: Correlacao volume-retorno
    - trade_intensity_{window}: Intensidade de trading
    - price_efficiency_{window}: Eficiencia informacional
    """

    def __init__(
        self,
        hurst_windows: tuple[int, ...] = (20, 60),
        vpin_buckets: tuple[int, ...] = (10, 20),
        micro_windows: tuple[int, ...] = (10, 20),
        autocorr_lags: tuple[int, ...] = (1, 2, 5),
    ):
        self.hurst_windows = hurst_windows
        self.vpin_buckets = vpin_buckets
        self.micro_windows = micro_windows
        self.autocorr_lags = autocorr_lags

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de microestrutura ao DataFrame.

        Args:
            df: DataFrame com colunas 'close', 'volume', 'log_return'

        Returns:
            DataFrame com features de microestrutura adicionadas
        """
        df = df.copy()

        # Preparar dados
        returns = df["log_return"].fillna(0).values if "log_return" in df.columns else np.zeros(len(df))
        close = df["close"].values if "close" in df.columns else np.ones(len(df))
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
        volume_usd = volume * close  # Volume em USD

        n = len(df)

        # === 1. Expoente de Hurst rolling ===
        for window in self.hurst_windows:
            hurst = np.full(n, 0.5)
            for i in range(window, n):
                hurst[i] = _hurst_exponent(returns[i - window:i], max_lag=min(20, window // 2))
            df[f"hurst_{window}"] = hurst

            # Regime baseado em Hurst
            df[f"hurst_regime_{window}"] = np.where(
                hurst > 0.55, 1,  # Trending
                np.where(hurst < 0.45, -1, 0)  # Mean-reverting / Neutral
            )

        # === 2. Dimensao Fractal rolling ===
        for window in self.hurst_windows:
            fd = np.full(n, 1.5)
            for i in range(window, n):
                fd[i] = _fractal_dimension(close[i - window:i])
            df[f"fractal_dim_{window}"] = fd

        # === 3. VPIN estimado ===
        for bucket in self.vpin_buckets:
            df[f"vpin_{bucket}"] = _vpin_estimate(volume_usd, returns, bucket_size=bucket)

        # === 4. Amihud Illiquidity Ratio ===
        for window in self.micro_windows:
            df[f"amihud_{window}"] = _amihud_illiquidity(returns, volume_usd, window)

        # === 5. Kyle's Lambda ===
        for window in self.micro_windows:
            df[f"kyle_lambda_{window}"] = _kyle_lambda(returns, volume_usd, window)

        # === 6. Roll's Spread ===
        for window in self.micro_windows:
            df[f"roll_spread_{window}"] = _roll_spread(returns, window)

        # === 7. Return Autocorrelation ===
        for lag in self.autocorr_lags:
            autocorr = np.full(n, np.nan)
            window = 20
            for i in range(window + lag, n):
                r1 = returns[i - window:i]
                r2 = returns[i - window - lag:i - lag]
                if np.std(r1) > 0 and np.std(r2) > 0:
                    autocorr[i] = np.corrcoef(r1, r2)[0, 1]
            df[f"return_autocorr_{lag}"] = autocorr

        # === 8. Volume-Return Correlation ===
        for window in self.micro_windows:
            vr_corr = np.full(n, np.nan)
            for i in range(window, n):
                v_w = volume_usd[i - window:i]
                r_w = np.abs(returns[i - window:i])
                if np.std(v_w) > 0 and np.std(r_w) > 0:
                    vr_corr[i] = np.corrcoef(v_w, r_w)[0, 1]
            df[f"volume_return_corr_{window}"] = vr_corr

        # === 9. Trade Intensity (volume acceleration) ===
        for window in self.micro_windows:
            vol_ma = pd.Series(volume_usd).rolling(window).mean().values
            vol_ma_long = pd.Series(volume_usd).rolling(window * 3).mean().values
            with np.errstate(divide='ignore', invalid='ignore'):
                intensity = np.where(vol_ma_long > 0, vol_ma / vol_ma_long, 1.0)
            df[f"trade_intensity_{window}"] = intensity

        # === 10. Price Efficiency (variance ratio) ===
        for window in self.micro_windows:
            # Variance ratio: Var(k-period return) / (k * Var(1-period return))
            # = 1 para random walk, > 1 para trending, < 1 para mean-reverting
            k = window
            eff = np.full(n, 1.0)
            for i in range(k * 2, n):
                r1 = returns[i - k * 2:i]
                var_1 = np.var(r1)
                if var_1 > 0 and i >= k:
                    # k-period returns
                    r_k = np.array([
                        np.sum(returns[j:j + k])
                        for j in range(i - k * 2, i - k + 1)
                    ])
                    var_k = np.var(r_k)
                    eff[i] = var_k / (k * var_1)
            df[f"price_efficiency_{window}"] = eff

        # Preencher NaN
        feature_cols = [c for c in df.columns if any(
            c.startswith(p) for p in [
                "hurst_", "fractal_", "vpin_", "amihud_", "kyle_",
                "roll_spread_", "return_autocorr_", "volume_return_corr_",
                "trade_intensity_", "price_efficiency_",
            ]
        )]
        for col in feature_cols:
            df[col] = df[col].fillna(method="ffill").fillna(0)

        logger.info(
            f"Microstructure features: {len(feature_cols)} features adicionadas"
        )

        return df

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features geradas."""
        names = []
        for w in self.hurst_windows:
            names.extend([f"hurst_{w}", f"hurst_regime_{w}", f"fractal_dim_{w}"])
        for b in self.vpin_buckets:
            names.append(f"vpin_{b}")
        for w in self.micro_windows:
            names.extend([
                f"amihud_{w}", f"kyle_lambda_{w}", f"roll_spread_{w}",
                f"volume_return_corr_{w}", f"trade_intensity_{w}",
                f"price_efficiency_{w}",
            ])
        for lag in self.autocorr_lags:
            names.append(f"return_autocorr_{lag}")
        return names
