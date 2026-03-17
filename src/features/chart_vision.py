"""Features visuais de graficos OHLCV inspiradas no VISTA.

Gera features baseadas em padroes de candlestick e perfil de volume,
com opcao de renderizar imagens PNG para analise por VLMs.
"""

import logging
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend nao-interativo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Nomes das features geradas
_FEATURE_NAMES = [
    "chart_consecutive_green",
    "chart_consecutive_red",
    "chart_body_ratio",
    "chart_upper_wick_ratio",
    "chart_lower_wick_ratio",
    "chart_pattern_score",
    "chart_volume_profile",
    "chart_pattern_bullish",
    "chart_pattern_bearish",
    "chart_momentum_score",
]

# Tamanho da janela para medias moveis de candles
_CANDLE_WINDOW = 10


class ChartVisionFeatures:
    """Gera features visuais a partir de dados OHLCV.

    Features computadas diretamente dos dados (sem dependencia de VLM):
    - chart_consecutive_green / red: contagem de candles consecutivos
    - chart_body_ratio: razao corpo/range media
    - chart_upper_wick_ratio / lower_wick_ratio: razao pavios/range
    - chart_pattern_score: score composto de padroes (-1 a 1)
    - chart_volume_profile: assimetria de volume relativo ao VWAP
    - chart_pattern_bullish / bearish: indicadores de padrao
    - chart_momentum_score: momentum baseado em candles

    Opcionalmente gera imagens PNG de candlestick para analise por VLMs.
    """

    def __init__(self, generate_images: bool = False, image_dir: str | None = None):
        """
        Args:
            generate_images: se True, gera PNGs de candlestick
            image_dir: diretorio para salvar imagens (usa tempdir se None)
        """
        self.generate_images = generate_images
        self.image_dir = Path(image_dir) if image_dir else None
        self._image_paths: list[Path] = []

    def get_feature_names(self) -> list[str]:
        """Retorna lista de nomes das features geradas."""
        return list(_FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Features numericas
    # ------------------------------------------------------------------

    @staticmethod
    def _consecutive_candles(closes: np.ndarray, opens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calcula contagem de candles consecutivos verdes/vermelhos para cada posicao."""
        n = len(closes)
        green_counts = np.zeros(n, dtype=np.float64)
        red_counts = np.zeros(n, dtype=np.float64)

        g_run = 0
        r_run = 0
        for i in range(n):
            if closes[i] >= opens[i]:
                g_run += 1
                r_run = 0
            else:
                r_run += 1
                g_run = 0
            green_counts[i] = g_run
            red_counts[i] = r_run

        return green_counts, red_counts

    @staticmethod
    def _candle_ratios(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int = _CANDLE_WINDOW,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula razoes de corpo e pavios sobre janela movel.

        Returns:
            (body_ratio, upper_wick_ratio, lower_wick_ratio)
        """
        n = len(opens)
        body_ratio = np.full(n, np.nan)
        upper_wick = np.full(n, np.nan)
        lower_wick = np.full(n, np.nan)

        total_range = highs - lows
        body = np.abs(closes - opens)
        upper = highs - np.maximum(opens, closes)
        lower = np.minimum(opens, closes) - lows

        # Evitar divisao por zero
        safe_range = np.where(total_range > 1e-10, total_range, np.nan)

        body_r = body / safe_range
        upper_r = upper / safe_range
        lower_r = lower / safe_range

        for i in range(window - 1, n):
            start = i - window + 1
            body_ratio[i] = np.nanmean(body_r[start: i + 1])
            upper_wick[i] = np.nanmean(upper_r[start: i + 1])
            lower_wick[i] = np.nanmean(lower_r[start: i + 1])

        return body_ratio, upper_wick, lower_wick

    @staticmethod
    def _volume_profile(
        closes: np.ndarray,
        volumes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        window: int = _CANDLE_WINDOW,
    ) -> np.ndarray:
        """Calcula assimetria do perfil de volume relativo ao VWAP.

        Valores positivos indicam mais volume acima do VWAP (distribuicao),
        negativos indicam acumulacao.
        """
        n = len(closes)
        profile = np.full(n, np.nan)

        for i in range(window - 1, n):
            start = i - window + 1
            slc = slice(start, i + 1)

            v = volumes[slc]
            c = closes[slc]
            total_vol = v.sum()

            if total_vol < 1e-10:
                profile[i] = 0.0
                continue

            # VWAP da janela
            typical_price = (highs[slc] + lows[slc] + c) / 3.0
            vwap = np.sum(typical_price * v) / total_vol

            # Volume acima e abaixo do VWAP
            above_mask = c > vwap
            vol_above = v[above_mask].sum()
            vol_below = v[~above_mask].sum()

            # Assimetria normalizada [-1, 1]
            if total_vol > 0:
                profile[i] = (vol_above - vol_below) / total_vol
            else:
                profile[i] = 0.0

        return profile

    @staticmethod
    def _pattern_score(
        green_counts: np.ndarray,
        red_counts: np.ndarray,
        body_ratio: np.ndarray,
        upper_wick: np.ndarray,
        lower_wick: np.ndarray,
    ) -> np.ndarray:
        """Score composto de padroes de candlestick, de -1 (bearish) a 1 (bullish).

        Combina:
        - Sequencias de candles verdes/vermelhos
        - Padroes de reversal (martelo, estrela cadente)
        - Proporcao corpo/pavios
        """
        n = len(green_counts)
        score = np.full(n, np.nan)

        for i in range(n):
            if np.isnan(body_ratio[i]):
                continue

            s = 0.0

            # Componente de tendencia (candles consecutivos)
            s += min(green_counts[i], 5) * 0.1  # max +0.5
            s -= min(red_counts[i], 5) * 0.1  # max -0.5

            # Componente de pavios (martelo / estrela cadente)
            br = body_ratio[i]
            uw = upper_wick[i]
            lw = lower_wick[i]

            # Martelo (bullish): pavio inferior longo, corpo pequeno
            if lw > 0.5 and br < 0.3:
                s += 0.25

            # Estrela cadente (bearish): pavio superior longo, corpo pequeno
            if uw > 0.5 and br < 0.3:
                s -= 0.25

            # Corpos grandes indicam convicao
            if br > 0.7:
                if green_counts[i] > 0:
                    s += 0.15
                elif red_counts[i] > 0:
                    s -= 0.15

            score[i] = np.clip(s, -1.0, 1.0)

        return score

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features visuais de candlestick ao DataFrame.

        Espera colunas: open, high, low, close, volume
        """
        df = df.copy()

        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        volumes = df["volume"].values.astype(np.float64)

        # Candles consecutivos
        green_counts, red_counts = self._consecutive_candles(closes, opens)
        df["chart_consecutive_green"] = green_counts
        df["chart_consecutive_red"] = red_counts

        # Razoes de corpo e pavios
        body_ratio, upper_wick, lower_wick = self._candle_ratios(
            opens, highs, lows, closes
        )
        df["chart_body_ratio"] = body_ratio
        df["chart_upper_wick_ratio"] = upper_wick
        df["chart_lower_wick_ratio"] = lower_wick

        # Score de padrao
        pattern = self._pattern_score(
            green_counts, red_counts, body_ratio, upper_wick, lower_wick
        )
        df["chart_pattern_score"] = pattern

        # Perfil de volume
        df["chart_volume_profile"] = self._volume_profile(
            closes, volumes, highs, lows
        )

        # Features derivadas de alto nivel
        df["chart_pattern_bullish"] = np.where(pattern > 0.3, 1.0, 0.0)
        df["chart_pattern_bearish"] = np.where(pattern < -0.3, 1.0, 0.0)

        # Momentum score: combinacao de tendencia + volume
        momentum = (
            0.6 * np.nan_to_num(pattern, nan=0.0)
            + 0.4 * np.nan_to_num(df["chart_volume_profile"].values, nan=0.0)
        )
        df["chart_momentum_score"] = np.clip(momentum, -1.0, 1.0)

        logger.info(
            f"  ChartVision: {len(_FEATURE_NAMES)} features adicionadas "
            f"({len(df)} amostras)"
        )

        # Gerar imagens opcionalmente
        if self.generate_images:
            self._generate_chart_images(df)

        return df

    # ------------------------------------------------------------------
    # Geracao de imagens de candlestick
    # ------------------------------------------------------------------

    def _generate_chart_images(
        self,
        df: pd.DataFrame,
        window: int = 30,
        step: int = 1,
    ) -> list[Path]:
        """Gera imagens PNG de candlestick para janelas deslizantes.

        Args:
            df: DataFrame com OHLCV
            window: numero de candles por imagem
            step: passo entre janelas

        Returns:
            Lista de caminhos das imagens geradas
        """
        if self.image_dir is None:
            self.image_dir = Path(tempfile.mkdtemp(prefix="chart_vision_"))
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self._image_paths = []
        n = len(df)

        for start in range(0, n - window + 1, step):
            end = start + window
            chunk = df.iloc[start:end]

            img_path = self.image_dir / f"chart_{start:06d}.png"
            self._render_candlestick(chunk, img_path)
            self._image_paths.append(img_path)

        logger.info(
            f"  ChartVision: {len(self._image_paths)} imagens geradas em {self.image_dir}"
        )
        return self._image_paths

    @staticmethod
    def _render_candlestick(chunk: pd.DataFrame, path: Path) -> None:
        """Renderiza um grafico de candlestick minimalista em PNG (240x180px)."""
        fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=100)

        opens = chunk["open"].values
        highs = chunk["high"].values
        lows = chunk["low"].values
        closes = chunk["close"].values

        n = len(chunk)
        width = 0.6

        for i in range(n):
            color = "#26a69a" if closes[i] >= opens[i] else "#ef5350"

            # Pavio (high-low)
            ax.plot(
                [i, i],
                [lows[i], highs[i]],
                color=color,
                linewidth=0.5,
            )

            # Corpo (open-close)
            body_bottom = min(opens[i], closes[i])
            body_height = abs(closes[i] - opens[i])
            rect = mpatches.FancyBboxPatch(
                (i - width / 2, body_bottom),
                width,
                max(body_height, (highs[i] - lows[i]) * 0.01),
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor=color,
                linewidth=0.3,
            )
            ax.add_patch(rect)

        # Estilo minimalista (sem eixos/texto para ML)
        ax.set_xlim(-1, n)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(
            path,
            dpi=100,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="white",
        )
        plt.close(fig)

    @property
    def image_paths(self) -> list[Path]:
        """Caminhos das imagens geradas na ultima chamada a transform()."""
        return list(self._image_paths)
