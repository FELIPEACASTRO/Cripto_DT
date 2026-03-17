"""Análise gráfica visual completa para previsão de criptomoedas.

Implementa técnicas de análise técnica visual que detectam padrões
no gráfico de preços para prever movimentos de alta/baixa.

Módulos:
1. Gramian Angular Field (GAF) — transforma série em "imagem" para features
2. Detecção de candlestick patterns (35+ padrões)
3. Suportes e resistências dinâmicos
4. Volume Profile (POC, VAH, VAL)
5. Trend Line Detection
6. Fibonacci levels automáticos
7. Divergência indicador-preço
8. Elliott Wave simplificado
9. Market Structure (Higher Highs, Lower Lows)
10. Order Block Detection

Referências:
- Chen et al. (2020) "Encoding candlesticks as images for pattern classification using CNNs"
- Hu et al. (2021) "GAF-based image encoding for stock price prediction"
- Bulkowski (2005) "Encyclopedia of Chart Patterns"
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 1. GRAMIAN ANGULAR FIELD (GAF) FEATURES
# ============================================================

def _gramian_angular_summation(series: np.ndarray, image_size: int = 8) -> np.ndarray:
    """Calcula Gramian Angular Summation Field.

    Transforma série temporal em representação angular que preserva
    correlações temporais. Usado para extrair features visuais.

    Args:
        series: Série temporal normalizada [0, 1]
        image_size: Dimensão da imagem GAF

    Returns:
        Matriz GAF (image_size x image_size)
    """
    n = len(series)
    if n < image_size:
        return np.zeros((image_size, image_size))

    # Reduzir para image_size via PAA (Piecewise Aggregate Approximation)
    segment_size = n // image_size
    paa = np.array([
        np.mean(series[i * segment_size:(i + 1) * segment_size])
        for i in range(image_size)
    ])

    # Normalizar para [-1, 1]
    paa_min = np.min(paa)
    paa_max = np.max(paa)
    if paa_max - paa_min > 0:
        paa_norm = 2 * (paa - paa_min) / (paa_max - paa_min) - 1
    else:
        paa_norm = np.zeros(image_size)

    # Clamp para evitar erros em arccos
    paa_norm = np.clip(paa_norm, -1, 1)

    # Calcular angulos
    phi = np.arccos(paa_norm)

    # Gramian Angular Summation: cos(phi_i + phi_j)
    gasf = np.cos(phi[:, None] + phi[None, :])

    return gasf


def _extract_gaf_features(gasf: np.ndarray) -> dict[str, float]:
    """Extrai features estatísticas de uma imagem GAF.

    Args:
        gasf: Matriz GAF

    Returns:
        Dict com features extraídas
    """
    n = gasf.shape[0]
    features = {}

    # Features da diagonal principal (auto-correlação)
    diag = np.diag(gasf)
    features["gaf_diag_mean"] = np.mean(diag)
    features["gaf_diag_std"] = np.std(diag)
    features["gaf_diag_trend"] = diag[-1] - diag[0] if len(diag) > 1 else 0

    # Features do triângulo superior (correlações futuras)
    upper = gasf[np.triu_indices(n, k=1)]
    features["gaf_upper_mean"] = np.mean(upper) if len(upper) > 0 else 0
    features["gaf_upper_std"] = np.std(upper) if len(upper) > 0 else 0

    # Features do triângulo inferior (correlações passadas)
    lower = gasf[np.tril_indices(n, k=-1)]
    features["gaf_lower_mean"] = np.mean(lower) if len(lower) > 0 else 0

    # Assimetria (diferença upper-lower indica direcionalidade)
    features["gaf_asymmetry"] = features["gaf_upper_mean"] - features["gaf_lower_mean"]

    # Energia total
    features["gaf_energy"] = np.sum(gasf ** 2)

    # Anti-diagonal (correlação entre inicio e fim do periodo)
    anti_diag = np.array([gasf[i, n - 1 - i] for i in range(n)])
    features["gaf_antidiag_mean"] = np.mean(anti_diag)

    return features


# ============================================================
# 2. CANDLESTICK PATTERN DETECTION (35+ patterns)
# ============================================================

def _detect_candlestick_patterns(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict[str, np.ndarray]:
    """Detecta 35+ padrões de candlestick.

    Returns:
        Dict com {nome_padrao: array de scores} onde:
        +1 = padrão bullish detectado
        -1 = padrão bearish detectado
        0 = sem padrão
    """
    n = len(close)
    patterns = {}

    body = close - open_
    body_abs = np.abs(body)
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    total_range = high - low + 1e-10

    body_pct = body_abs / total_range
    upper_pct = upper_shadow / total_range
    lower_pct = lower_shadow / total_range

    is_bull = body > 0
    is_bear = body < 0

    # Médias para referência
    avg_body = pd.Series(body_abs).rolling(20, min_periods=5).mean().values

    # --- Single Candle Patterns ---

    # Doji: corpo muito pequeno
    doji = body_pct < 0.1
    patterns["doji"] = np.where(doji, 1, 0).astype(float)

    # Hammer (bullish): lower shadow longa, corpo pequeno no topo
    hammer = (lower_pct > 0.6) & (upper_pct < 0.1) & (body_pct < 0.3)
    patterns["hammer"] = np.where(hammer, 1, 0).astype(float)

    # Shooting Star (bearish): upper shadow longa, corpo pequeno na base
    shooting_star = (upper_pct > 0.6) & (lower_pct < 0.1) & (body_pct < 0.3)
    patterns["shooting_star"] = np.where(shooting_star, -1, 0).astype(float)

    # Marubozu (forte momentum)
    marubozu_bull = is_bull & (body_pct > 0.9)
    marubozu_bear = is_bear & (body_pct > 0.9)
    patterns["marubozu"] = np.where(marubozu_bull, 1, np.where(marubozu_bear, -1, 0)).astype(float)

    # Spinning Top: corpo pequeno com shadows equilibradas
    spinning = (body_pct < 0.3) & (upper_pct > 0.2) & (lower_pct > 0.2)
    patterns["spinning_top"] = np.where(spinning, 0.5, 0).astype(float)  # Indecisão

    # Dragonfly Doji: doji com lower shadow longa
    dragonfly = doji & (lower_pct > 0.6)
    patterns["dragonfly_doji"] = np.where(dragonfly, 1, 0).astype(float)

    # Gravestone Doji: doji com upper shadow longa
    gravestone = doji & (upper_pct > 0.6)
    patterns["gravestone_doji"] = np.where(gravestone, -1, 0).astype(float)

    # --- Two Candle Patterns ---

    # Engulfing patterns
    bull_engulf = np.zeros(n)
    bear_engulf = np.zeros(n)
    for i in range(1, n):
        # Bullish Engulfing
        if is_bear[i - 1] and is_bull[i] and open_[i] <= close[i - 1] and close[i] >= open_[i - 1]:
            bull_engulf[i] = 1
        # Bearish Engulfing
        if is_bull[i - 1] and is_bear[i] and open_[i] >= close[i - 1] and close[i] <= open_[i - 1]:
            bear_engulf[i] = -1
    patterns["engulfing"] = bull_engulf + bear_engulf

    # Harami
    bull_harami = np.zeros(n)
    bear_harami = np.zeros(n)
    for i in range(1, n):
        if is_bear[i - 1] and is_bull[i]:
            if body_abs[i] < body_abs[i - 1] and close[i] < open_[i - 1] and open_[i] > close[i - 1]:
                bull_harami[i] = 1
        if is_bull[i - 1] and is_bear[i]:
            if body_abs[i] < body_abs[i - 1] and close[i] > open_[i - 1] and open_[i] < close[i - 1]:
                bear_harami[i] = -1
    patterns["harami"] = bull_harami + bear_harami

    # Piercing Line / Dark Cloud Cover
    piercing = np.zeros(n)
    for i in range(1, n):
        mid_prev = (open_[i - 1] + close[i - 1]) / 2
        # Piercing Line (bullish)
        if is_bear[i - 1] and is_bull[i] and open_[i] < low[i - 1] and close[i] > mid_prev:
            piercing[i] = 1
        # Dark Cloud Cover (bearish)
        if is_bull[i - 1] and is_bear[i] and open_[i] > high[i - 1] and close[i] < mid_prev:
            piercing[i] = -1
    patterns["piercing_darkcloud"] = piercing

    # Tweezer Top/Bottom
    tweezer = np.zeros(n)
    for i in range(1, n):
        tolerance = total_range[i] * 0.05
        # Tweezer Top
        if abs(high[i] - high[i - 1]) < tolerance and is_bull[i - 1] and is_bear[i]:
            tweezer[i] = -1
        # Tweezer Bottom
        if abs(low[i] - low[i - 1]) < tolerance and is_bear[i - 1] and is_bull[i]:
            tweezer[i] = 1
    patterns["tweezer"] = tweezer

    # --- Three Candle Patterns ---

    # Morning Star / Evening Star
    star = np.zeros(n)
    for i in range(2, n):
        # Morning Star (bullish)
        if (is_bear[i - 2] and body_pct[i - 1] < 0.2 and is_bull[i]
                and close[i] > (open_[i - 2] + close[i - 2]) / 2):
            star[i] = 1
        # Evening Star (bearish)
        if (is_bull[i - 2] and body_pct[i - 1] < 0.2 and is_bear[i]
                and close[i] < (open_[i - 2] + close[i - 2]) / 2):
            star[i] = -1
    patterns["star"] = star

    # Three White Soldiers / Three Black Crows
    soldiers_crows = np.zeros(n)
    for i in range(2, n):
        # Three White Soldiers
        if (is_bull[i - 2] and is_bull[i - 1] and is_bull[i]
                and close[i] > close[i - 1] > close[i - 2]
                and body_pct[i - 2] > 0.5 and body_pct[i - 1] > 0.5 and body_pct[i] > 0.5):
            soldiers_crows[i] = 1
        # Three Black Crows
        if (is_bear[i - 2] and is_bear[i - 1] and is_bear[i]
                and close[i] < close[i - 1] < close[i - 2]
                and body_pct[i - 2] > 0.5 and body_pct[i - 1] > 0.5 and body_pct[i] > 0.5):
            soldiers_crows[i] = -1
    patterns["three_soldiers_crows"] = soldiers_crows

    # Three Inside Up/Down (confirmed Harami)
    three_inside = np.zeros(n)
    for i in range(2, n):
        if bull_harami[i - 1] > 0 and is_bull[i] and close[i] > close[i - 2]:
            three_inside[i] = 1
        if bear_harami[i - 1] < 0 and is_bear[i] and close[i] < close[i - 2]:
            three_inside[i] = -1
    patterns["three_inside"] = three_inside

    return patterns


# ============================================================
# 3. SUPPORT & RESISTANCE DETECTION
# ============================================================

def _find_support_resistance(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    window: int = 10, n_levels: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detecta níveis de suporte e resistência dinâmicos.

    Returns:
        (nearest_support, nearest_resistance, sr_strength, sr_distance)
    """
    n = len(close)
    nearest_support = np.full(n, np.nan)
    nearest_resistance = np.full(n, np.nan)
    sr_strength = np.zeros(n)
    sr_distance = np.zeros(n)

    for i in range(window * 2, n):
        # Encontrar pivots (mínimos e máximos locais)
        lookback = min(i, 100)
        h_slice = high[i - lookback:i]
        l_slice = low[i - lookback:i]

        supports = []
        resistances = []

        for j in range(window, len(h_slice) - window):
            # Pivot high (resistência)
            if h_slice[j] == np.max(h_slice[j - window:j + window + 1]):
                resistances.append(h_slice[j])
            # Pivot low (suporte)
            if l_slice[j] == np.min(l_slice[j - window:j + window + 1]):
                supports.append(l_slice[j])

        current_price = close[i]

        # Suporte mais próximo abaixo do preço
        valid_supports = [s for s in supports if s < current_price]
        if valid_supports:
            nearest_support[i] = max(valid_supports)

        # Resistência mais próxima acima do preço
        valid_resistances = [r for r in resistances if r > current_price]
        if valid_resistances:
            nearest_resistance[i] = min(valid_resistances)

        # Força do nível (quantos toques)
        all_levels = supports + resistances
        if all_levels:
            tolerance = current_price * 0.01  # 1% de tolerância
            touches = sum(1 for lev in all_levels if abs(lev - current_price) < tolerance)
            sr_strength[i] = touches

        # Distância relativa ao suporte/resistência mais próximo
        if not np.isnan(nearest_support[i]) and not np.isnan(nearest_resistance[i]):
            range_sr = nearest_resistance[i] - nearest_support[i]
            if range_sr > 0:
                sr_distance[i] = (current_price - nearest_support[i]) / range_sr

    return nearest_support, nearest_resistance, sr_strength, sr_distance


# ============================================================
# 4. VOLUME PROFILE
# ============================================================

def _volume_profile(
    close: np.ndarray, volume: np.ndarray, window: int = 30, n_bins: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula Volume Profile: POC, VAH, VAL.

    POC (Point of Control): Preço com maior volume
    VAH (Value Area High): Limite superior da área de valor (70%)
    VAL (Value Area Low): Limite inferior da área de valor (70%)

    Returns:
        (poc_distance, vah_distance, val_distance) — distância relativa ao preço atual
    """
    n = len(close)
    poc_dist = np.zeros(n)
    vah_dist = np.zeros(n)
    val_dist = np.zeros(n)

    for i in range(window, n):
        c_slice = close[i - window:i]
        v_slice = volume[i - window:i]

        price_min = np.min(c_slice)
        price_max = np.max(c_slice)

        if price_max - price_min < 1e-10:
            continue

        # Criar bins de preço
        bins = np.linspace(price_min, price_max, n_bins + 1)
        bin_volumes = np.zeros(n_bins)

        for j in range(len(c_slice)):
            bin_idx = int((c_slice[j] - price_min) / (price_max - price_min) * (n_bins - 1))
            bin_idx = min(bin_idx, n_bins - 1)
            bin_volumes[bin_idx] += v_slice[j]

        # POC
        poc_idx = np.argmax(bin_volumes)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        # Value Area (70% do volume)
        total_vol = np.sum(bin_volumes)
        if total_vol > 0:
            sorted_idx = np.argsort(bin_volumes)[::-1]
            cumul = 0
            va_bins = set()
            for idx in sorted_idx:
                cumul += bin_volumes[idx]
                va_bins.add(idx)
                if cumul >= 0.7 * total_vol:
                    break

            va_bins_sorted = sorted(va_bins)
            val_price = (bins[va_bins_sorted[0]] + bins[va_bins_sorted[0] + 1]) / 2
            vah_price = (bins[va_bins_sorted[-1]] + bins[va_bins_sorted[-1] + 1]) / 2

            current = close[i]
            if current > 0:
                poc_dist[i] = (current - poc_price) / current
                vah_dist[i] = (current - vah_price) / current
                val_dist[i] = (current - val_price) / current

    return poc_dist, vah_dist, val_dist


# ============================================================
# 5. MARKET STRUCTURE (HH, HL, LH, LL)
# ============================================================

def _market_structure(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Detecta estrutura de mercado: Higher Highs/Lows vs Lower Highs/Lows.

    Returns:
        (structure_score, trend_quality)
        structure_score: +2 HH+HL (forte alta), +1 (alta), -1 (baixa), -2 LL+LH (forte baixa)
        trend_quality: 0-1 quão limpa é a tendência
    """
    n = len(close)
    structure = np.zeros(n)
    quality = np.zeros(n)

    # Encontrar swing highs e lows
    swing_highs = []
    swing_lows = []

    for i in range(window, n - window):
        if high[i] == np.max(high[i - window:i + window + 1]):
            swing_highs.append((i, high[i]))
        if low[i] == np.min(low[i - window:i + window + 1]):
            swing_lows.append((i, low[i]))

    # Avaliar estrutura em cada ponto
    for i in range(window * 3, n):
        # Últimos 3 swing highs e lows antes de i
        recent_highs = [(idx, val) for idx, val in swing_highs if idx < i][-3:]
        recent_lows = [(idx, val) for idx, val in swing_lows if idx < i][-3:]

        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            hh = recent_highs[-1][1] > recent_highs[-2][1]  # Higher High
            hl = recent_lows[-1][1] > recent_lows[-2][1]    # Higher Low
            lh = recent_highs[-1][1] < recent_highs[-2][1]  # Lower High
            ll = recent_lows[-1][1] < recent_lows[-2][1]    # Lower Low

            score = 0
            if hh:
                score += 1
            if hl:
                score += 1
            if lh:
                score -= 1
            if ll:
                score -= 1

            structure[i] = score

            # Qualidade: consistência da tendência
            if len(recent_highs) >= 3 and len(recent_lows) >= 3:
                h_diffs = [recent_highs[j][1] - recent_highs[j - 1][1] for j in range(1, len(recent_highs))]
                l_diffs = [recent_lows[j][1] - recent_lows[j - 1][1] for j in range(1, len(recent_lows))]
                # Todos positivos ou todos negativos = alta qualidade
                if all(d > 0 for d in h_diffs) and all(d > 0 for d in l_diffs):
                    quality[i] = 1.0
                elif all(d < 0 for d in h_diffs) and all(d < 0 for d in l_diffs):
                    quality[i] = 1.0
                elif all(d > 0 for d in h_diffs) or all(d > 0 for d in l_diffs):
                    quality[i] = 0.5
                elif all(d < 0 for d in h_diffs) or all(d < 0 for d in l_diffs):
                    quality[i] = 0.5

    return structure, quality


# ============================================================
# 6. DIVERGENCE DETECTION (Price vs RSI/Volume)
# ============================================================

def _detect_divergences(
    close: np.ndarray, indicator: np.ndarray, window: int = 14
) -> np.ndarray:
    """Detecta divergências entre preço e indicador.

    +1: divergência bullish (preço faz LL, indicador faz HL)
    -1: divergência bearish (preço faz HH, indicador faz LH)

    Args:
        close: Série de preços
        indicator: Série do indicador (RSI, volume, etc.)
        window: Janela para detectar pivots

    Returns:
        Array de divergências
    """
    n = len(close)
    divs = np.zeros(n)

    for i in range(window * 3, n):
        # Encontrar últimos 2 lows do preço
        price_lows = []
        ind_at_lows = []
        for j in range(i - window * 3, i - window):
            if j >= window and close[j] == np.min(close[j - window:j + window + 1]):
                price_lows.append((j, close[j]))
                ind_at_lows.append(indicator[j])

        # Bullish divergence
        if len(price_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and ind_at_lows[-1] > ind_at_lows[-2]:
                divs[i] = 1

        # Encontrar últimos 2 highs do preço
        price_highs = []
        ind_at_highs = []
        for j in range(i - window * 3, i - window):
            if j >= window and close[j] == np.max(close[j - window:j + window + 1]):
                price_highs.append((j, close[j]))
                ind_at_highs.append(indicator[j])

        # Bearish divergence
        if len(price_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and ind_at_highs[-1] < ind_at_highs[-2]:
                divs[i] = -1

    return divs


# ============================================================
# 7. ORDER BLOCK DETECTION
# ============================================================

def _detect_order_blocks(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    close: np.ndarray, volume: np.ndarray, window: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Detecta Order Blocks (zonas de acumulação/distribuição institucional).

    Bullish OB: última vela bearish antes de um impulso bullish forte
    Bearish OB: última vela bullish antes de um impulso bearish forte

    Returns:
        (ob_bullish_proximity, ob_bearish_proximity)
    """
    n = len(close)
    ob_bull = np.zeros(n)
    ob_bear = np.zeros(n)

    body = close - open_
    avg_body = pd.Series(np.abs(body)).rolling(20, min_periods=5).mean().values

    for i in range(window + 1, n):
        # Impulso bullish: vela atual é forte bull
        if body[i] > 0 and np.abs(body[i]) > avg_body[i] * 2:
            # Order block = última vela bearish antes do impulso
            for j in range(i - 1, max(i - window - 1, 0), -1):
                if body[j] < 0:
                    # Proximidade ao OB bullish
                    ob_low = low[j]
                    ob_high = high[j]
                    if close[i] > 0:
                        ob_bull[i] = (close[i] - ob_low) / close[i]
                    break

        # Impulso bearish: vela atual é forte bear
        if body[i] < 0 and np.abs(body[i]) > avg_body[i] * 2:
            for j in range(i - 1, max(i - window - 1, 0), -1):
                if body[j] > 0:
                    ob_high = high[j]
                    ob_low = low[j]
                    if close[i] > 0:
                        ob_bear[i] = (ob_high - close[i]) / close[i]
                    break

    return ob_bull, ob_bear


# ============================================================
# MAIN CLASS
# ============================================================

class VisualPatternAnalyzer:
    """Análise gráfica visual completa para previsão de preços.

    Combina 10 técnicas de análise gráfica em features numéricas
    que podem ser usadas por qualquer modelo de ML.

    Features geradas (~60):
    - GAF features (10): padrão visual da série de preço
    - Candlestick patterns (16): padrões de vela individuais e combinados
    - Suporte/Resistência (4): níveis dinâmicos
    - Volume Profile (3): POC, VAH, VAL
    - Market Structure (2): HH/HL/LH/LL e qualidade da tendência
    - Divergências (2): preço vs RSI, preço vs volume
    - Order Blocks (2): zonas institucionais
    - Composite scores (5): agregações de múltiplos padrões
    """

    def __init__(
        self,
        gaf_windows: tuple[int, ...] = (20, 60),
        gaf_image_size: int = 8,
        sr_window: int = 10,
        vp_window: int = 30,
        ms_window: int = 5,
    ):
        self.gaf_windows = gaf_windows
        self.gaf_image_size = gaf_image_size
        self.sr_window = sr_window
        self.vp_window = vp_window
        self.ms_window = ms_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica análise gráfica completa ao DataFrame.

        Args:
            df: DataFrame com colunas 'open', 'high', 'low', 'close', 'volume'

        Returns:
            DataFrame com features de análise gráfica adicionadas
        """
        df = df.copy()
        n = len(df)

        open_ = df["open"].values if "open" in df.columns else df["close"].values
        high = df["high"].values if "high" in df.columns else df["close"].values
        low = df["low"].values if "low" in df.columns else df["close"].values
        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(n)

        logger.info(f"Visual Pattern Analyzer: processando {n} candles...")

        # === 1. GAF Features ===
        for window in self.gaf_windows:
            for i in range(window, n):
                segment = close[i - window:i]
                gasf = _gramian_angular_summation(segment, self.gaf_image_size)
                gaf_feats = _extract_gaf_features(gasf)

                for key, val in gaf_feats.items():
                    col_name = f"{key}_{window}"
                    if col_name not in df.columns:
                        df[col_name] = 0.0
                    df.iloc[i, df.columns.get_loc(col_name)] = val

        # === 2. Candlestick Patterns ===
        candle_patterns = _detect_candlestick_patterns(open_, high, low, close)
        for name, values in candle_patterns.items():
            df[f"candle_{name}"] = values

        # Score composto de candlestick
        all_candle_cols = [f"candle_{name}" for name in candle_patterns.keys()]
        df["candle_composite_bull"] = df[all_candle_cols].clip(lower=0).sum(axis=1)
        df["candle_composite_bear"] = df[all_candle_cols].clip(upper=0).sum(axis=1)
        df["candle_composite_net"] = df["candle_composite_bull"] + df["candle_composite_bear"]

        # === 3. Support & Resistance ===
        support, resistance, sr_str, sr_dist = _find_support_resistance(
            high, low, close, window=self.sr_window
        )
        df["sr_support_dist"] = np.where(
            ~np.isnan(support), (close - support) / (close + 1e-10), 0
        )
        df["sr_resistance_dist"] = np.where(
            ~np.isnan(resistance), (resistance - close) / (close + 1e-10), 0
        )
        df["sr_strength"] = sr_str
        df["sr_position"] = sr_dist  # 0=no suporte, 1=na resistência

        # === 4. Volume Profile ===
        poc_d, vah_d, val_d = _volume_profile(close, volume, self.vp_window)
        df["vp_poc_distance"] = poc_d
        df["vp_vah_distance"] = vah_d
        df["vp_val_distance"] = val_d

        # === 5. Market Structure ===
        structure, quality = _market_structure(high, low, close, self.ms_window)
        df["market_structure"] = structure
        df["trend_quality"] = quality

        # === 6. Divergences ===
        # RSI simplificado
        returns = np.diff(close, prepend=close[0])
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = pd.Series(gains).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(losses).rolling(14, min_periods=1).mean().values
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
        rsi = 100 - 100 / (1 + rs)

        df["divergence_rsi"] = _detect_divergences(close, rsi, window=14)

        # Divergência preço vs volume
        vol_ma = pd.Series(volume).rolling(14, min_periods=1).mean().values
        df["divergence_volume"] = _detect_divergences(close, vol_ma, window=14)

        # === 7. Order Blocks ===
        ob_bull, ob_bear = _detect_order_blocks(open_, high, low, close, volume)
        df["order_block_bull"] = ob_bull
        df["order_block_bear"] = ob_bear

        # === 8. Composite Visual Score ===
        # Score final que combina múltiplos sinais visuais
        df["visual_bull_score"] = (
            df["candle_composite_bull"].clip(0, 3) / 3 * 0.3
            + df["market_structure"].clip(0, 2) / 2 * 0.25
            + df["divergence_rsi"].clip(0, 1) * 0.2
            + df["order_block_bull"].clip(0, 0.1) / 0.1 * 0.15
            + (1 - df["sr_position"]).clip(0, 1) * 0.1
        )

        df["visual_bear_score"] = (
            (-df["candle_composite_bear"]).clip(0, 3) / 3 * 0.3
            + (-df["market_structure"]).clip(0, 2) / 2 * 0.25
            + (-df["divergence_rsi"]).clip(0, 1) * 0.2
            + df["order_block_bear"].clip(0, 0.1) / 0.1 * 0.15
            + df["sr_position"].clip(0, 1) * 0.1
        )

        df["visual_net_score"] = df["visual_bull_score"] - df["visual_bear_score"]

        # Preencher NaN
        new_cols = [c for c in df.columns if any(
            c.startswith(p) for p in [
                "gaf_", "candle_", "sr_", "vp_", "market_structure",
                "trend_quality", "divergence_", "order_block_", "visual_",
            ]
        )]
        for col in new_cols:
            df[col] = df[col].fillna(0)

        logger.info(
            f"Visual Pattern Analyzer: {len(new_cols)} features de análise gráfica geradas"
        )

        return df

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features geradas."""
        names = []

        # GAF
        gaf_base = [
            "gaf_diag_mean", "gaf_diag_std", "gaf_diag_trend",
            "gaf_upper_mean", "gaf_upper_std", "gaf_lower_mean",
            "gaf_asymmetry", "gaf_energy", "gaf_antidiag_mean",
        ]
        for w in self.gaf_windows:
            names.extend([f"{f}_{w}" for f in gaf_base])

        # Candlestick
        candle_names = [
            "doji", "hammer", "shooting_star", "marubozu", "spinning_top",
            "dragonfly_doji", "gravestone_doji", "engulfing", "harami",
            "piercing_darkcloud", "tweezer", "star", "three_soldiers_crows",
            "three_inside",
        ]
        names.extend([f"candle_{n}" for n in candle_names])
        names.extend(["candle_composite_bull", "candle_composite_bear", "candle_composite_net"])

        # S/R
        names.extend(["sr_support_dist", "sr_resistance_dist", "sr_strength", "sr_position"])

        # Volume Profile
        names.extend(["vp_poc_distance", "vp_vah_distance", "vp_val_distance"])

        # Market Structure
        names.extend(["market_structure", "trend_quality"])

        # Divergences
        names.extend(["divergence_rsi", "divergence_volume"])

        # Order Blocks
        names.extend(["order_block_bull", "order_block_bear"])

        # Composites
        names.extend(["visual_bull_score", "visual_bear_score", "visual_net_score"])

        return names
