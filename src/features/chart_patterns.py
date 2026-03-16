"""Deteccao avancada de padroes graficos em dados OHLCV.

Inspirado nas abordagens MORFI (ICCV 2025) e DEJIMA para reconhecimento de
padroes visuais em graficos de candlestick. Utiliza reconhecimento geometrico
matematico diretamente nos dados OHLCV — sem necessidade de processamento
de imagens.

Padroes detectados:
- Head & Shoulders (e inverso)
- Double Top / Double Bottom
- Triangulos (ascendente, descendente, simetrico)
- Flag / Pennant (bandeira de alta e baixa)
- Cup and Handle
- Wedge (cunha ascendente/descendente)
- Niveis de suporte/resistencia
- Retracao de Fibonacci
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Niveis classicos de Fibonacci
_FIBONACCI_LEVELS = np.array([0.236, 0.382, 0.5, 0.618, 0.786])

# Nomes de todas as features geradas
_FEATURE_NAMES = [
    "chart_head_shoulders",
    "chart_double_top",
    "chart_triangle",
    "chart_flag",
    "chart_cup_handle",
    "chart_wedge",
    "chart_pattern_strength",
    "chart_support_level",
    "chart_resistance_level",
    "chart_trendline_slope",
    "chart_pattern_breakout",
    "chart_fibonacci_retrace",
]


def _find_local_extrema(
    arr: np.ndarray, order: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Encontra indices de maximos e minimos locais sem scipy.

    Args:
        arr: serie de precos
        order: numero de pontos de cada lado para comparar

    Returns:
        (indices_maximos, indices_minimos)
    """
    n = len(arr)
    if n < 2 * order + 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    maxima: List[int] = []
    minima: List[int] = []

    for i in range(order, n - order):
        window = arr[i - order: i + order + 1]
        if arr[i] == np.nanmax(window):
            maxima.append(i)
        if arr[i] == np.nanmin(window):
            minima.append(i)

    return np.array(maxima, dtype=int), np.array(minima, dtype=int)


class ChartPatternDetector:
    """Detecta padroes graficos classicos em dados OHLCV.

    Gera 12 features com prefixo ``chart_`` usando reconhecimento geometrico
    em multiplas escalas temporais (janelas de 20, 50 e 100 barras).

    Parameters:
        tolerance: tolerancia percentual para niveis "similares" (default 0.02 = 2%)
        windows: lista de tamanhos de janela para deteccao multi-escala
        extrema_order: ordem para deteccao de extremos locais
    """

    def __init__(
        self,
        tolerance: float = 0.02,
        windows: List[int] | None = None,
        extrema_order: int = 5,
    ):
        self.tolerance = tolerance
        self.windows = windows or [20, 50, 100]
        self.extrema_order = extrema_order

    def get_feature_names(self) -> List[str]:
        """Retorna lista de nomes das features geradas."""
        return list(_FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Deteccao de padroes individuais
    # ------------------------------------------------------------------

    def _detect_head_shoulders(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        peak_idx: np.ndarray,
        trough_idx: np.ndarray,
    ) -> int:
        """Detecta Head & Shoulders (bearish) ou inverso (bullish).

        Returns:
            -1: H&S bearish, +1: H&S inverso (bullish), 0: nao detectado
        """
        # H&S normal: 3 picos onde o do meio e mais alto
        if len(peak_idx) >= 3:
            for i in range(len(peak_idx) - 2):
                left = highs[peak_idx[i]]
                head = highs[peak_idx[i + 1]]
                right = highs[peak_idx[i + 2]]

                if head <= left or head <= right:
                    continue

                # Ombros devem estar em niveis similares
                shoulder_diff = abs(left - right) / max(left, right, 1e-10)
                if shoulder_diff < self.tolerance * 2:
                    # Verificar neckline break
                    if peak_idx[i + 2] < len(closes) - 1:
                        neckline = min(left, right)
                        if closes[-1] < neckline:
                            return -1

        # H&S inverso: 3 vales onde o do meio e mais baixo
        if len(trough_idx) >= 3:
            for i in range(len(trough_idx) - 2):
                left = lows[trough_idx[i]]
                head = lows[trough_idx[i + 1]]
                right = lows[trough_idx[i + 2]]

                if head >= left or head >= right:
                    continue

                shoulder_diff = abs(left - right) / max(left, right, 1e-10)
                if shoulder_diff < self.tolerance * 2:
                    if trough_idx[i + 2] < len(closes) - 1:
                        neckline = max(left, right)
                        if closes[-1] > neckline:
                            return 1

        return 0

    def _detect_double_top_bottom(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        peak_idx: np.ndarray,
        trough_idx: np.ndarray,
    ) -> int:
        """Detecta Double Top (bearish) ou Double Bottom (bullish).

        Returns:
            -1: double top, +1: double bottom, 0: nao detectado
        """
        # Double top: 2 picos em niveis similares
        if len(peak_idx) >= 2:
            for i in range(len(peak_idx) - 1):
                p1 = highs[peak_idx[i]]
                p2 = highs[peak_idx[i + 1]]
                diff = abs(p1 - p2) / max(p1, p2, 1e-10)

                if diff < self.tolerance:
                    # Deve haver um vale entre eles
                    between = trough_idx[
                        (trough_idx > peak_idx[i]) & (trough_idx < peak_idx[i + 1])
                    ]
                    if len(between) > 0:
                        valley = lows[between[0]]
                        if closes[-1] < valley:
                            return -1

        # Double bottom: 2 vales em niveis similares
        if len(trough_idx) >= 2:
            for i in range(len(trough_idx) - 1):
                t1 = lows[trough_idx[i]]
                t2 = lows[trough_idx[i + 1]]
                diff = abs(t1 - t2) / max(t1, t2, 1e-10)

                if diff < self.tolerance:
                    between = peak_idx[
                        (peak_idx > trough_idx[i]) & (peak_idx < trough_idx[i + 1])
                    ]
                    if len(between) > 0:
                        ridge = highs[between[0]]
                        if closes[-1] > ridge:
                            return 1

        return 0

    def _detect_triangle(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        peak_idx: np.ndarray,
        trough_idx: np.ndarray,
    ) -> int:
        """Detecta triangulos (ascendente, descendente, simetrico).

        Returns:
            +1: ascendente (bullish), -1: descendente (bearish), 0: nao detectado
        """
        if len(peak_idx) < 2 or len(trough_idx) < 2:
            return 0

        # Tendencia dos picos (trendline superior)
        peak_vals = highs[peak_idx[-3:]] if len(peak_idx) >= 3 else highs[peak_idx]
        peak_positions = peak_idx[-3:] if len(peak_idx) >= 3 else peak_idx

        # Tendencia dos vales (trendline inferior)
        trough_vals = lows[trough_idx[-3:]] if len(trough_idx) >= 3 else lows[trough_idx]
        trough_positions = trough_idx[-3:] if len(trough_idx) >= 3 else trough_idx

        if len(peak_positions) < 2 or len(trough_positions) < 2:
            return 0

        # Slopes das linhas de tendencia
        upper_slope = np.polyfit(peak_positions.astype(float), peak_vals, 1)[0]
        lower_slope = np.polyfit(trough_positions.astype(float), trough_vals, 1)[0]

        # Normalizar slopes pelo preco medio
        price_mean = np.mean(np.concatenate([peak_vals, trough_vals]))
        if price_mean < 1e-10:
            return 0
        upper_norm = upper_slope / price_mean
        lower_norm = lower_slope / price_mean

        # Convergencia: slopes em direcoes opostas ou um flat
        converging = (upper_norm < -0.0001 and lower_norm > 0.0001) or \
                     (abs(upper_norm) < 0.0002 and lower_norm > 0.0001) or \
                     (upper_norm < -0.0001 and abs(lower_norm) < 0.0002)

        if not converging:
            return 0

        # Ascendente: resistencia flat, suporte subindo
        if abs(upper_norm) < 0.0005 and lower_norm > 0.0002:
            return 1

        # Descendente: suporte flat, resistencia caindo
        if upper_norm < -0.0002 and abs(lower_norm) < 0.0005:
            return -1

        # Simetrico: convergindo (sinal neutro, retorna 0)
        return 0

    def _detect_flag(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> int:
        """Detecta Flag/Pennant (bandeira de alta ou baixa).

        Um flag e um movimento forte (pole) seguido de consolidacao.

        Returns:
            +1: bull flag, -1: bear flag, 0: nao detectado
        """
        n = len(closes)
        if n < 15:
            return 0

        # Dividir em pole (primeiros 40%) e flag (ultimos 60%)
        pole_end = int(n * 0.4)
        pole = closes[:pole_end]
        flag_section = closes[pole_end:]

        if len(pole) < 3 or len(flag_section) < 5:
            return 0

        # Retorno do pole
        pole_return = (pole[-1] - pole[0]) / max(abs(pole[0]), 1e-10)

        # Volatilidade do flag (consolidacao = baixa volatilidade)
        flag_range = (highs[pole_end:].max() - lows[pole_end:].min())
        pole_range = (highs[:pole_end].max() - lows[:pole_end].min())

        if pole_range < 1e-10:
            return 0

        consolidation_ratio = flag_range / pole_range

        # Flag detectado: pole forte + consolidacao
        if abs(pole_return) > 0.03 and consolidation_ratio < 0.5:
            if pole_return > 0:
                return 1  # bull flag
            else:
                return -1  # bear flag

        return 0

    def _detect_cup_handle(
        self,
        closes: np.ndarray,
        lows: np.ndarray,
        peak_idx: np.ndarray,
        trough_idx: np.ndarray,
    ) -> int:
        """Detecta Cup and Handle (padrao bullish).

        Forma de U seguida de pequena consolidacao.

        Returns:
            1: cup and handle detectado, 0: nao detectado
        """
        n = len(closes)
        if n < 20 or len(peak_idx) < 2 or len(trough_idx) < 1:
            return 0

        # Procurar forma U: dois picos com vale entre eles
        for i in range(len(peak_idx) - 1):
            p1_idx = peak_idx[i]
            p2_idx = peak_idx[i + 1]
            p1_val = closes[p1_idx]
            p2_val = closes[p2_idx]

            # Picos devem estar em niveis similares (borda da taca)
            rim_diff = abs(p1_val - p2_val) / max(p1_val, p2_val, 1e-10)
            if rim_diff > self.tolerance * 2:
                continue

            # Encontrar vale mais profundo entre os picos
            between_troughs = trough_idx[
                (trough_idx > p1_idx) & (trough_idx < p2_idx)
            ]
            if len(between_troughs) == 0:
                continue

            cup_bottom_idx = between_troughs[np.argmin(lows[between_troughs])]
            cup_bottom = lows[cup_bottom_idx]
            rim = max(p1_val, p2_val)

            # A taca deve ter profundidade significativa
            depth = (rim - cup_bottom) / max(rim, 1e-10)
            if depth < 0.05:
                continue

            # Formato U: o fundo deve estar aproximadamente no meio
            mid_point = (p1_idx + p2_idx) / 2
            bottom_position = abs(cup_bottom_idx - mid_point) / max(p2_idx - p1_idx, 1)
            if bottom_position > 0.35:
                continue

            # Verificar "handle": pequena queda apos segundo pico
            if p2_idx < n - 3:
                handle_section = closes[p2_idx: min(p2_idx + 10, n)]
                handle_dip = (handle_section.max() - handle_section.min()) / max(rim, 1e-10)
                if handle_dip < depth * 0.5:
                    return 1

        return 0

    def _detect_wedge(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        peak_idx: np.ndarray,
        trough_idx: np.ndarray,
    ) -> int:
        """Detecta Wedge (cunha ascendente=bearish, descendente=bullish).

        Returns:
            -1: rising wedge (bearish), +1: falling wedge (bullish), 0: nao detectado
        """
        if len(peak_idx) < 2 or len(trough_idx) < 2:
            return 0

        peak_vals = highs[peak_idx[-3:]] if len(peak_idx) >= 3 else highs[peak_idx]
        peak_pos = peak_idx[-3:] if len(peak_idx) >= 3 else peak_idx
        trough_vals = lows[trough_idx[-3:]] if len(trough_idx) >= 3 else lows[trough_idx]
        trough_pos = trough_idx[-3:] if len(trough_idx) >= 3 else trough_idx

        if len(peak_pos) < 2 or len(trough_pos) < 2:
            return 0

        upper_slope = np.polyfit(peak_pos.astype(float), peak_vals, 1)[0]
        lower_slope = np.polyfit(trough_pos.astype(float), trough_vals, 1)[0]

        price_mean = np.mean(np.concatenate([peak_vals, trough_vals]))
        if price_mean < 1e-10:
            return 0

        upper_norm = upper_slope / price_mean
        lower_norm = lower_slope / price_mean

        # Rising wedge: ambas as linhas subindo, mas convergindo
        if upper_norm > 0.0001 and lower_norm > 0.0001:
            if upper_norm < lower_norm * 1.5:  # convergindo
                return -1  # bearish

        # Falling wedge: ambas caindo, mas convergindo
        if upper_norm < -0.0001 and lower_norm < -0.0001:
            if lower_norm > upper_norm * 1.5:  # convergindo
                return 1  # bullish

        return 0

    # ------------------------------------------------------------------
    # Suporte, resistencia e Fibonacci
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_support_resistance(
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        peak_idx: np.ndarray,
        trough_idx: np.ndarray,
    ) -> Tuple[float, float]:
        """Calcula distancia normalizada ao suporte e resistencia mais proximos.

        Returns:
            (distancia_suporte, distancia_resistencia) normalizadas pelo preco atual
        """
        current = closes[-1]
        if current < 1e-10:
            return 0.0, 0.0

        # Niveis de resistencia: picos recentes
        if len(peak_idx) > 0:
            resistance_levels = highs[peak_idx]
            above = resistance_levels[resistance_levels > current]
            if len(above) > 0:
                nearest_res = above.min()
                res_dist = (nearest_res - current) / current
            else:
                res_dist = 0.0
        else:
            res_dist = 0.0

        # Niveis de suporte: vales recentes
        if len(trough_idx) > 0:
            support_levels = lows[trough_idx]
            below = support_levels[support_levels < current]
            if len(below) > 0:
                nearest_sup = below.max()
                sup_dist = (current - nearest_sup) / current
            else:
                sup_dist = 0.0
        else:
            sup_dist = 0.0

        return sup_dist, res_dist

    @staticmethod
    def _compute_fibonacci(closes: np.ndarray) -> float:
        """Retorna o nivel de Fibonacci mais proximo da posicao atual.

        Calcula retracao entre o maximo e minimo recentes.
        """
        if len(closes) < 5:
            return 0.5

        high = np.nanmax(closes)
        low = np.nanmin(closes)
        rng = high - low

        if rng < 1e-10:
            return 0.5

        current = closes[-1]
        retrace = (high - current) / rng

        # Encontrar nivel de Fibonacci mais proximo
        distances = np.abs(_FIBONACCI_LEVELS - retrace)
        nearest_idx = np.argmin(distances)
        return float(_FIBONACCI_LEVELS[nearest_idx])

    @staticmethod
    def _compute_trendline_slope(closes: np.ndarray) -> float:
        """Calcula slope da trendline dominante normalizado."""
        n = len(closes)
        if n < 3:
            return 0.0

        x = np.arange(n, dtype=float)
        slope = np.polyfit(x, closes, 1)[0]

        price_mean = np.mean(closes)
        if price_mean < 1e-10:
            return 0.0

        return float(slope / price_mean)

    # ------------------------------------------------------------------
    # Pipeline de deteccao por janela
    # ------------------------------------------------------------------

    def _detect_patterns_window(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> dict:
        """Executa todas as deteccoes em uma janela de dados.

        Returns:
            Dicionario com sinais de cada padrao
        """
        peak_idx, trough_idx = _find_local_extrema(
            closes, order=self.extrema_order
        )

        hs = self._detect_head_shoulders(
            highs, lows, closes, peak_idx, trough_idx
        )
        dt = self._detect_double_top_bottom(
            highs, lows, closes, peak_idx, trough_idx
        )
        tri = self._detect_triangle(highs, lows, peak_idx, trough_idx)
        flag = self._detect_flag(closes, highs, lows)
        cup = self._detect_cup_handle(closes, lows, peak_idx, trough_idx)
        wedge = self._detect_wedge(highs, lows, peak_idx, trough_idx)

        sup_dist, res_dist = self._compute_support_resistance(
            closes, highs, lows, peak_idx, trough_idx
        )
        fib = self._compute_fibonacci(closes)
        slope = self._compute_trendline_slope(closes)

        # Breakout: preco cruza suporte ou resistencia recente
        breakout = 0
        if len(peak_idx) > 0 and closes[-1] > highs[peak_idx].max():
            breakout = 1
        elif len(trough_idx) > 0 and closes[-1] < lows[trough_idx].min():
            breakout = -1

        # Forca composta: quantos padroes foram detectados
        signals = [hs, dt, tri, flag, cup, wedge]
        active = sum(1 for s in signals if s != 0)
        strength = min(active / 3.0, 1.0)

        return {
            "chart_head_shoulders": hs,
            "chart_double_top": dt,
            "chart_triangle": tri,
            "chart_flag": flag,
            "chart_cup_handle": cup,
            "chart_wedge": wedge,
            "chart_pattern_strength": strength,
            "chart_support_level": sup_dist,
            "chart_resistance_level": res_dist,
            "chart_trendline_slope": slope,
            "chart_pattern_breakout": breakout,
            "chart_fibonacci_retrace": fib,
        }

    # ------------------------------------------------------------------
    # Interface publica
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de padroes graficos ao DataFrame.

        Executa deteccao em multiplas escalas (janelas) e combina os
        resultados. Para cada barra, utiliza a maior janela disponivel.

        Espera colunas: open, high, low, close, volume

        Args:
            df: DataFrame OHLCV

        Returns:
            DataFrame com 12 novas colunas ``chart_*``
        """
        df = df.copy()
        n = len(df)

        if n == 0 or "high" not in df.columns:
            return df

        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)

        # Inicializar colunas
        for feat in _FEATURE_NAMES:
            df[feat] = 0.0

        min_window = min(self.windows)

        for i in range(n):
            if i < min_window - 1:
                continue

            # Usar a maior janela que cabe
            best_result = None
            for w in sorted(self.windows):
                if i < w - 1:
                    continue
                start = i - w + 1
                try:
                    result = self._detect_patterns_window(
                        highs[start: i + 1],
                        lows[start: i + 1],
                        closes[start: i + 1],
                    )
                    best_result = result
                except Exception:
                    # Falha silenciosa — manter resultado da janela menor
                    pass

            if best_result is not None:
                for feat, val in best_result.items():
                    df.at[df.index[i], feat] = val

        logger.info(
            f"  ChartPatterns: {len(_FEATURE_NAMES)} features adicionadas "
            f"({n} amostras, janelas={self.windows})"
        )

        return df
