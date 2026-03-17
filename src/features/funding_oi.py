"""Features de Funding Rate e Open Interest para perpetual futures.

Funding Rate e Open Interest são indicadores extremamente valiosos para
prever movimentos de preço em criptomoedas:

- Funding Rate alto positivo → mercado overleveraged long → possível queda
- Funding Rate negativo → mercado overleveraged short → possível alta
- OI crescendo + preço subindo → tendência confirmada (dinheiro novo)
- OI caindo + preço subindo → short squeeze (insustentável)
- OI crescendo + preço caindo → distribuição (bearish)

Como não temos acesso direto à API de futuros, criamos proxies
baseados em dados OHLCV spot que capturam dinâmicas similares.

Referência: Bybit Research (2023) "Funding Rate as a Contrarian Indicator"
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundingOIFeatures:
    """Gera features proxy de Funding Rate e Open Interest.

    Como dados reais de futuros nem sempre estão disponíveis,
    usa proxies derivados de OHLCV spot:
    - Volume acceleration como proxy de OI
    - Price-volume divergence como proxy de funding pressure
    - Consecutive directional days como proxy de leverage buildup
    """

    def __init__(
        self,
        windows: tuple[int, ...] = (3, 7, 14),
        extreme_threshold: float = 2.0,
    ):
        self.windows = windows
        self.extreme_threshold = extreme_threshold

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de funding rate e OI proxy.

        Args:
            df: DataFrame com 'close', 'volume', 'high', 'low'

        Returns:
            DataFrame com features adicionadas
        """
        df = df.copy()
        n = len(df)

        close = df["close"].values if "close" in df.columns else np.ones(n)
        volume = df["volume"].values if "volume" in df.columns else np.ones(n)
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close

        returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
        log_returns = np.log(close / np.roll(close, 1))
        log_returns[0] = 0

        # === 1. Funding Rate Proxy ===
        # Baseado em consecutive directional pressure + volume
        for window in self.windows:
            # Contagem de dias consecutivos na mesma direção
            consec = np.zeros(n)
            for i in range(1, n):
                if returns[i] > 0 and returns[i - 1] > 0:
                    consec[i] = consec[i - 1] + 1
                elif returns[i] < 0 and returns[i - 1] < 0:
                    consec[i] = consec[i - 1] - 1
                else:
                    consec[i] = np.sign(returns[i])

            # Funding proxy = pressão direcional acumulada
            funding_proxy = pd.Series(consec).rolling(window, min_periods=1).mean().values
            df[f"funding_proxy_{window}"] = funding_proxy

            # Funding extremo (possível reversão)
            std_funding = pd.Series(funding_proxy).rolling(60, min_periods=10).std().values
            df[f"funding_extreme_{window}"] = np.where(
                std_funding > 0,
                np.abs(funding_proxy) / (std_funding + 1e-10),
                0
            )

            # Sinal de reversão quando funding está extremo
            df[f"funding_reversal_signal_{window}"] = np.where(
                df[f"funding_extreme_{window}"] > self.extreme_threshold,
                -np.sign(funding_proxy),  # Contrarian
                0
            )

        # === 2. Open Interest Proxy ===
        # Volume como proxy de OI (volume crescente = mais posições abertas)
        vol_usd = volume * close

        for window in self.windows:
            # Volume trend (proxy de OI direction)
            vol_ma_short = pd.Series(vol_usd).rolling(window, min_periods=1).mean().values
            vol_ma_long = pd.Series(vol_usd).rolling(window * 4, min_periods=1).mean().values

            oi_proxy = np.where(vol_ma_long > 0, vol_ma_short / vol_ma_long - 1, 0)
            df[f"oi_proxy_{window}"] = oi_proxy

            # OI-Price divergence (classificação de regime)
            price_trend = pd.Series(close).pct_change(window).fillna(0).values

            # OI up + Price up = tendência confirmada (1)
            # OI up + Price down = distribuição (-1)
            # OI down + Price up = short squeeze (0.5, insustentável)
            # OI down + Price down = capitulação (-0.5)
            regime = np.zeros(n)
            for i in range(window, n):
                oi_up = oi_proxy[i] > 0.05
                oi_down = oi_proxy[i] < -0.05
                price_up = price_trend[i] > 0
                price_down = price_trend[i] < 0

                if oi_up and price_up:
                    regime[i] = 1.0    # Tendência confirmada
                elif oi_up and price_down:
                    regime[i] = -1.0   # Distribuição
                elif oi_down and price_up:
                    regime[i] = 0.5    # Short squeeze
                elif oi_down and price_down:
                    regime[i] = -0.5   # Capitulação

            df[f"oi_price_regime_{window}"] = regime

        # === 3. Leverage Proxy ===
        # Alta volatilidade + alto volume = alto leverage
        for window in self.windows:
            volatility = pd.Series(log_returns).rolling(window, min_periods=1).std().values
            vol_norm = pd.Series(vol_usd).rolling(window, min_periods=1).mean().values
            vol_long = pd.Series(vol_usd).rolling(window * 4, min_periods=1).mean().values

            vol_ratio = np.where(vol_long > 0, vol_norm / vol_long, 1)

            # Leverage proxy = volatilidade * volume_ratio
            leverage = volatility * vol_ratio
            df[f"leverage_proxy_{window}"] = leverage

            # Liquidation risk (leverage acima do normal)
            lev_mean = pd.Series(leverage).rolling(60, min_periods=10).mean().values
            lev_std = pd.Series(leverage).rolling(60, min_periods=10).std().values
            df[f"liquidation_risk_{window}"] = np.where(
                lev_std > 0,
                (leverage - lev_mean) / (lev_std + 1e-10),
                0
            )

        # === 4. Long/Short Pressure ===
        # Baseado em posição do close no range high-low
        for window in self.windows:
            # Pressão buying: close próximo do high consistentemente
            close_position = (close - low) / (high - low + 1e-10)
            buying_pressure = pd.Series(close_position).rolling(window, min_periods=1).mean().values
            df[f"buying_pressure_{window}"] = buying_pressure

            # Imbalance: diferença entre buying e selling pressure
            selling_pressure = 1 - buying_pressure
            df[f"pressure_imbalance_{window}"] = buying_pressure - selling_pressure

        # === 5. Squeeze Detection ===
        # Bollinger Band width contraction + volume spike = squeeze
        bb_window = 20
        bb_ma = pd.Series(close).rolling(bb_window, min_periods=5).mean().values
        bb_std = pd.Series(close).rolling(bb_window, min_periods=5).std().values
        bb_width = np.where(bb_ma > 0, 2 * bb_std / bb_ma, 0)

        # Width compression
        bb_width_ma = pd.Series(bb_width).rolling(60, min_periods=10).mean().values
        df["squeeze_compression"] = np.where(
            bb_width_ma > 0,
            bb_width / (bb_width_ma + 1e-10),
            1
        )

        # Squeeze detectado quando width está muito comprimido
        df["squeeze_detected"] = np.where(df["squeeze_compression"] < 0.5, 1, 0).astype(float)

        # Preencher NaN
        new_cols = [c for c in df.columns if any(
            c.startswith(p) for p in [
                "funding_", "oi_", "leverage_", "liquidation_",
                "buying_pressure_", "pressure_imbalance_", "squeeze_",
            ]
        )]
        for col in new_cols:
            df[col] = pd.Series(df[col]).ffill().fillna(0).values

        logger.info(f"Funding/OI features: {len(new_cols)} features adicionadas")

        return df

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features geradas."""
        names = []
        for w in self.windows:
            names.extend([
                f"funding_proxy_{w}", f"funding_extreme_{w}", f"funding_reversal_signal_{w}",
                f"oi_proxy_{w}", f"oi_price_regime_{w}",
                f"leverage_proxy_{w}", f"liquidation_risk_{w}",
                f"buying_pressure_{w}", f"pressure_imbalance_{w}",
            ])
        names.extend(["squeeze_compression", "squeeze_detected"])
        return names
