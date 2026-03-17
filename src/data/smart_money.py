"""Rastreamento de Smart Money — inteligencia on-chain e heuristicas OHLCV.

Inspirado em SocialScan e aixbt (aiagentstore.ai).
Complementa o whale_monitor.py com metricas de fluxo institucional,
acumulacao/distribuicao e concentracao de baleias.

Todas as features funcionam a partir de dados OHLCV sozinhos;
APIs externas (Blockchain.com, Etherscan) sao enriquecimento opcional.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ── Enderecos conhecidos de smart money (exemplos publicos) ──────────────
# Usados quando APIs on-chain estao disponiveis para enriquecer dados
KNOWN_SMART_MONEY: dict[str, list[dict[str, str]]] = {
    "BTC": [
        {"label": "Grayscale Cold Wallet", "address": "bc1qe7nk2nlnjewjga5kkal5eryaj2ckgpx9gvc4x7"},
        {"label": "MicroStrategy Treasury", "address": "bc1qazcm763858nkj2dz7g3vafgk2ys9xv8vy7x7s"},
        {"label": "Binance Cold Wallet", "address": "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo"},
    ],
    "ETH": [
        {"label": "Binance Hot Wallet", "address": "0x28C6c06298d514Db089934071355E5743bf21d60"},
        {"label": "Jump Trading", "address": "0xf584F8728B874a6a5c7A8d4d387C9aae9172D621"},
        {"label": "Wintermute", "address": "0x0000000000000000000000000000000000000000"},
    ],
}

# Limiares de deteccao
_ACCUMULATION_LOOKBACK = 14       # Janela para detectar acumulacao (dias)
_DISTRIBUTION_LOOKBACK = 14       # Janela para detectar distribuicao (dias)
_MFI_PERIOD = 14                  # Periodo padrao para Money Flow Index
_EXCHANGE_FLOW_PERIOD = 20        # Janela para estimativa de fluxo de exchange
_CONCENTRATION_PERIOD = 30        # Janela para concentracao de baleias
_VOLUME_WHALE_MULTIPLIER = 2.0    # Multiplo para considerar volume de "smart money"


class SmartMoneyTracker:
    """Rastreia fluxos de smart money usando dados OHLCV e APIs on-chain.

    Gera features de inteligencia institucional que complementam o WhaleMonitor.
    Todas as features sao computaveis a partir de dados OHLCV;
    APIs on-chain (Blockchain.com, Etherscan) enriquecem quando disponiveis.
    """

    # URLs de APIs externas
    BLOCKCHAIN_API_BASE = "https://blockchain.info"
    ETHERSCAN_API_BASE = "https://api.etherscan.io/api"

    def __init__(self, etherscan_api_key: str | None = None) -> None:
        """Inicializa o rastreador de smart money.

        Args:
            etherscan_api_key: Chave da API Etherscan (opcional).
                               Se None, features ETH on-chain nao serao buscadas.
        """
        self._etherscan_api_key = etherscan_api_key
        # Cache de dados on-chain para evitar chamadas repetidas
        self._onchain_cache: dict[str, Any] = {}

    # ══════════════════════════════════════════════════════════════════════
    # API PUBLICA
    # ══════════════════════════════════════════════════════════════════════

    def add_smart_money_features(
        self, df: pd.DataFrame, coin: str
    ) -> pd.DataFrame:
        """Adiciona todas as features de smart money ao DataFrame OHLCV.

        Features geradas:
            - smart_money_flow: indice de fluxo de smart money (-1 a 1)
            - smart_money_accumulation: indicador de acumulacao (0 a 1)
            - smart_money_distribution: indicador de distribuicao (0 a 1)
            - exchange_flow_ratio: proxy de fluxo de exchange (inflow/outflow)
            - whale_concentration: razao de concentracao de baleias (0+)

        Args:
            df: DataFrame com colunas OHLCV (open, high, low, close, volume)
            coin: Simbolo da moeda (ex: BTC, ETH, SOL)

        Returns:
            DataFrame com features de smart money adicionadas
        """
        df = df.copy()

        # Verificar colunas minimas necessarias
        colunas_necessarias = {"close", "volume"}
        if not colunas_necessarias.issubset(df.columns):
            logger.error(
                f"Colunas minimas ausentes ({colunas_necessarias - set(df.columns)}) "
                "— retornando features vazias"
            )
            return self._add_empty_features(df)

        # Garantir tipos numericos
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Calcular cada feature ────────────────────────────────────────
        df = self._calc_smart_money_flow(df)
        df = self._calc_accumulation(df)
        df = self._calc_distribution(df)
        df = self._calc_exchange_flow_ratio(df)
        df = self._calc_whale_concentration(df)

        # ── Tentar enriquecer com dados on-chain (opcional) ──────────────
        df = self._enrich_with_onchain(df, coin)

        logger.info(
            f"Features de smart money calculadas para {coin}: {len(df)} linhas | "
            f"flow medio={df['smart_money_flow'].mean():.3f}, "
            f"acum medio={df['smart_money_accumulation'].mean():.3f}"
        )

        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Retorna nomes das features de smart money.

        Returns:
            Lista com nomes das features:
            - smart_money_flow: score de fluxo de smart money (-1 a 1)
            - smart_money_accumulation: indicador de acumulacao (0 a 1)
            - smart_money_distribution: indicador de distribuicao (0 a 1)
            - exchange_flow_ratio: proxy de fluxo exchange (inflow/outflow)
            - whale_concentration: razao de concentracao de baleias
        """
        return [
            "smart_money_flow",
            "smart_money_accumulation",
            "smart_money_distribution",
            "exchange_flow_ratio",
            "whale_concentration",
        ]

    # ══════════════════════════════════════════════════════════════════════
    # CALCULO DE FEATURES (baseado em OHLCV)
    # ══════════════════════════════════════════════════════════════════════

    def _calc_smart_money_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula Smart Money Flow Index — variante do MFI ponderada por tamanho.

        Logica: o Money Flow Index classico usa preco tipico * volume.
        Aqui ponderamos pelo ratio de volume vs media, dando mais peso
        a periodos com volume anomalamente alto (proxy de smart money).

        Resultado normalizado entre -1 (distribuicao forte) e 1 (acumulacao forte).
        """
        close = df["close"]
        volume = df["volume"]

        # Preco tipico (se high/low disponiveis, senao usar close)
        if "high" in df.columns and "low" in df.columns:
            typical_price = (df["high"] + df["low"] + close) / 3.0
        else:
            typical_price = close

        # Money flow bruto: preco tipico * volume
        raw_money_flow = typical_price * volume

        # Peso de smart money: volume relativo a media movel
        vol_ma = volume.rolling(window=_MFI_PERIOD, min_periods=1).mean()
        smart_weight = (volume / vol_ma.replace(0, np.nan)).fillna(1.0)

        # Ponderar money flow pelo peso de smart money
        # Valores altos de smart_weight amplificam o sinal
        weighted_flow = raw_money_flow * smart_weight

        # Separar fluxo positivo (preco subiu) e negativo (preco caiu)
        price_change = close.diff()
        positive_flow = weighted_flow.where(price_change > 0, 0.0)
        negative_flow = weighted_flow.where(price_change < 0, 0.0)

        # Somar fluxos em janela rolante
        pos_sum = positive_flow.rolling(window=_MFI_PERIOD, min_periods=1).sum()
        neg_sum = negative_flow.abs().rolling(window=_MFI_PERIOD, min_periods=1).sum()

        # Money Flow Ratio e normalizacao para [-1, 1]
        total_flow = pos_sum + neg_sum
        # Evitar divisao por zero
        safe_total = total_flow.replace(0, np.nan)
        # Score: (positivo - negativo) / total -> range [-1, 1]
        df["smart_money_flow"] = (
            (pos_sum - neg_sum) / safe_total
        ).fillna(0.0).clip(-1.0, 1.0)

        return df

    def _calc_accumulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta acumulacao de smart money via divergencia preco-volume.

        Acumulacao ocorre quando volume sobe mas preco nao sobe proporcionalmente
        (ou cai pouco). Sinal de compra silenciosa por grandes players.

        Usa indicador Accumulation/Distribution adaptado + confirmacao por volume.
        Resultado normalizado entre 0 (sem acumulacao) e 1 (acumulacao forte).
        """
        close = df["close"]
        volume = df["volume"]

        # ── Componente 1: Accumulation/Distribution Line (ADL) ───────────
        if "high" in df.columns and "low" in df.columns:
            high = df["high"]
            low = df["low"]
            # Money Flow Multiplier: posicao do close dentro do range [low, high]
            hl_range = (high - low).replace(0, np.nan)
            mf_multiplier = ((close - low) - (high - close)) / hl_range
            mf_multiplier = mf_multiplier.fillna(0.0)
        else:
            # Sem high/low: usar sinal da variacao de preco
            mf_multiplier = np.sign(close.diff()).fillna(0.0)

        # ADL: money flow multiplier * volume (acumulado)
        mf_volume = mf_multiplier * volume
        adl = mf_volume.cumsum()

        # Variacao da ADL normalizada em janela rolante
        adl_change = adl.diff(periods=_ACCUMULATION_LOOKBACK)
        adl_std = adl.rolling(window=_ACCUMULATION_LOOKBACK, min_periods=1).std()
        adl_zscore = (adl_change / adl_std.replace(0, np.nan)).fillna(0.0)

        # ── Componente 2: divergencia preco-volume ───────────────────────
        # Volume subindo + preco estavel ou caindo = acumulacao
        vol_change_pct = volume.pct_change(periods=_ACCUMULATION_LOOKBACK).fillna(0.0)
        price_change_pct = close.pct_change(periods=_ACCUMULATION_LOOKBACK).fillna(0.0)

        # Divergencia: volume subindo mais que preco (ou preco caindo)
        divergence = (vol_change_pct - price_change_pct).clip(lower=0.0)

        # ── Combinar componentes e normalizar para [0, 1] ───────────────
        # ADL positivo + divergencia positiva = acumulacao
        raw_accum = adl_zscore.clip(lower=0.0) * 0.6 + divergence * 0.4

        # Normalizar com sigmoid suave para range [0, 1]
        df["smart_money_accumulation"] = self._sigmoid_normalize(raw_accum)

        return df

    def _calc_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta distribuicao de smart money (venda institucional).

        Distribuicao ocorre quando preco sobe mas volume cai (ou preco
        sobe com volume alto seguido de queda — distribuicao apos pump).

        Resultado normalizado entre 0 (sem distribuicao) e 1 (distribuicao forte).
        """
        close = df["close"]
        volume = df["volume"]

        # ── Componente 1: On Balance Volume (OBV) divergencia ────────────
        # OBV acumula volume com sinal da variacao de preco
        obv_sign = np.sign(close.diff()).fillna(0.0)
        obv = (obv_sign * volume).cumsum()

        # Divergencia: preco subindo mas OBV caindo = distribuicao
        price_trend = close.rolling(
            window=_DISTRIBUTION_LOOKBACK, min_periods=1
        ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0.0, raw=True)

        obv_trend = obv.rolling(
            window=_DISTRIBUTION_LOOKBACK, min_periods=1
        ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0.0, raw=True)

        # Normalizar tendencias para comparacao
        price_trend_norm = price_trend / close.rolling(
            window=_DISTRIBUTION_LOOKBACK, min_periods=1
        ).mean().replace(0, np.nan)
        price_trend_norm = price_trend_norm.fillna(0.0)

        obv_trend_norm = obv_trend / obv.abs().rolling(
            window=_DISTRIBUTION_LOOKBACK, min_periods=1
        ).mean().replace(0, np.nan)
        obv_trend_norm = obv_trend_norm.fillna(0.0)

        # Divergencia negativa: preco sobe mas OBV cai
        divergence = (price_trend_norm - obv_trend_norm).clip(lower=0.0)

        # ── Componente 2: volume decrescente em alta ─────────────────────
        vol_declining = (-volume.pct_change(periods=_DISTRIBUTION_LOOKBACK)).clip(lower=0.0).fillna(0.0)
        price_rising = close.pct_change(periods=_DISTRIBUTION_LOOKBACK).clip(lower=0.0).fillna(0.0)
        sell_pressure = vol_declining * price_rising

        # ── Combinar e normalizar para [0, 1] ───────────────────────────
        raw_dist = divergence * 0.6 + sell_pressure * 100 * 0.4
        df["smart_money_distribution"] = self._sigmoid_normalize(raw_dist)

        return df

    def _calc_exchange_flow_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estima fluxo de exchange (inflow vs outflow) via padroes OHLCV.

        Proxy baseado em:
        - Volume alto + preco caindo = provavel inflow para exchange (venda)
        - Volume alto + preco subindo = provavel outflow de exchange (compra)

        Ratio > 1 significa mais inflow (pressao vendedora)
        Ratio < 1 significa mais outflow (pressao compradora)
        Ratio = 1 significa equilibrio
        """
        close = df["close"]
        volume = df["volume"]

        # Classificar cada candle como inflow ou outflow proxy
        price_change = close.diff().fillna(0.0)

        # Volume em dias de queda = proxy de inflow (depositando para vender)
        inflow_proxy = volume.where(price_change < 0, 0.0)
        # Volume em dias de alta = proxy de outflow (retirando apos comprar)
        outflow_proxy = volume.where(price_change > 0, 0.0)

        # Se temos high/low, usar posicao do close no range para refinar
        if "high" in df.columns and "low" in df.columns:
            hl_range = (df["high"] - df["low"]).replace(0, np.nan)
            # Close perto do low = mais inflow; close perto do high = mais outflow
            close_position = ((close - df["low"]) / hl_range).fillna(0.5)

            # Ponderar inflow/outflow pela posicao do close
            inflow_weight = (1.0 - close_position)  # Mais peso quando close perto do low
            outflow_weight = close_position           # Mais peso quando close perto do high

            inflow_proxy = volume * inflow_weight
            outflow_proxy = volume * outflow_weight

        # Somar em janela rolante
        inflow_sum = inflow_proxy.rolling(
            window=_EXCHANGE_FLOW_PERIOD, min_periods=1
        ).sum()
        outflow_sum = outflow_proxy.rolling(
            window=_EXCHANGE_FLOW_PERIOD, min_periods=1
        ).sum()

        # Ratio: inflow / outflow (evitar divisao por zero)
        df["exchange_flow_ratio"] = (
            inflow_sum / outflow_sum.replace(0, np.nan)
        ).fillna(1.0).clip(0.1, 10.0)  # Limitar extremos

        return df

    def _calc_whale_concentration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estima concentracao de baleias via desbalanceamento de order book.

        Proxy: a relacao entre volume anomalo e volume total indica
        quanto do mercado e movido por poucos grandes players.

        Concentracao alta = poucas entidades movem muito volume.
        """
        volume = df["volume"]

        # Media e desvio padrao rolante do volume
        vol_ma = volume.rolling(window=_CONCENTRATION_PERIOD, min_periods=1).mean()
        vol_std = volume.rolling(window=_CONCENTRATION_PERIOD, min_periods=1).std()

        # Volume acima de 2x a media = proxy de "whale volume"
        whale_threshold = vol_ma + _VOLUME_WHALE_MULTIPLIER * vol_std.fillna(0.0)
        whale_volume = volume.where(volume > whale_threshold, 0.0)

        # Razao de concentracao: volume de baleias / volume total na janela
        whale_vol_sum = whale_volume.rolling(
            window=_CONCENTRATION_PERIOD, min_periods=1
        ).sum()
        total_vol_sum = volume.rolling(
            window=_CONCENTRATION_PERIOD, min_periods=1
        ).sum()

        df["whale_concentration"] = (
            whale_vol_sum / total_vol_sum.replace(0, np.nan)
        ).fillna(0.0).clip(0.0, 1.0)

        # Amplificar sinal: usar variacao de amplitude como proxy de
        # impacto de grandes ordens no preco
        if "high" in df.columns and "low" in df.columns:
            amplitude = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).fillna(0.0)
            amp_ma = amplitude.rolling(window=_CONCENTRATION_PERIOD, min_periods=1).mean()
            amp_ratio = (amplitude / amp_ma.replace(0, np.nan)).fillna(1.0)

            # Concentracao ajustada: peso extra quando amplitude tambem e alta
            df["whale_concentration"] = (
                df["whale_concentration"] * 0.7
                + (amp_ratio / amp_ratio.quantile(0.95) if amp_ratio.quantile(0.95) > 0 else amp_ratio).clip(0, 1) * 0.3
            ).clip(0.0, 1.0)

        return df

    # ══════════════════════════════════════════════════════════════════════
    # ENRIQUECIMENTO VIA APIs ON-CHAIN (opcional)
    # ══════════════════════════════════════════════════════════════════════

    def _enrich_with_onchain(
        self, df: pd.DataFrame, coin: str
    ) -> pd.DataFrame:
        """Tenta enriquecer features com dados on-chain de APIs externas.

        Nao altera features se APIs falharem — OHLCV e suficiente.
        """
        coin_upper = coin.upper()

        if coin_upper == "BTC":
            self._enrich_btc(df)
        elif coin_upper == "ETH" and self._etherscan_api_key:
            self._enrich_eth(df)
        else:
            logger.debug(
                f"Sem API on-chain para {coin} — usando apenas heuristicas OHLCV"
            )

        return df

    def _enrich_btc(self, df: pd.DataFrame) -> None:
        """Enriquece com metricas BTC do Blockchain.com (nao critico)."""
        try:
            # Buscar hash rate como proxy de saude da rede
            resp = requests.get(
                f"{self.BLOCKCHAIN_API_BASE}/q/hashrate", timeout=10
            )
            resp.raise_for_status()
            hashrate = float(resp.text)
            self._onchain_cache["btc_hashrate"] = hashrate
            logger.info(f"BTC hashrate obtido: {hashrate:.0f}")

            # Buscar dificuldade de mineracao
            resp_diff = requests.get(
                f"{self.BLOCKCHAIN_API_BASE}/q/getdifficulty", timeout=10
            )
            resp_diff.raise_for_status()
            difficulty = float(resp_diff.text)
            self._onchain_cache["btc_difficulty"] = difficulty
            logger.info(f"BTC difficulty obtida: {difficulty:.0f}")

        except Exception as e:
            logger.warning(f"Falha ao buscar metricas BTC on-chain: {e}")

    def _enrich_eth(self, df: pd.DataFrame) -> None:
        """Enriquece com dados Etherscan para ETH (requer API key)."""
        if not self._etherscan_api_key:
            return

        try:
            # Buscar supply de ETH como referencia
            params = {
                "module": "stats",
                "action": "ethsupply",
                "apikey": self._etherscan_api_key,
            }
            resp = requests.get(
                self.ETHERSCAN_API_BASE, params=params, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "1":
                supply = float(data["result"]) / 1e18  # Wei para ETH
                self._onchain_cache["eth_supply"] = supply
                logger.info(f"ETH supply obtido: {supply:.2f}")
        except Exception as e:
            logger.warning(f"Falha ao buscar metricas ETH on-chain: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # UTILITARIOS
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _sigmoid_normalize(series: pd.Series, k: float = 2.0) -> pd.Series:
        """Normaliza uma serie para [0, 1] usando funcao sigmoid.

        Args:
            series: Serie com valores arbitrarios
            k: Fator de escala (maior = mais sensivel)

        Returns:
            Serie normalizada entre 0 e 1
        """
        # Padronizar (z-score) antes de aplicar sigmoid
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.5, index=series.index)

        z = (series - mean) / std
        return 1.0 / (1.0 + np.exp(-k * z))

    def _add_empty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de smart money com valores neutros (fallback)."""
        df["smart_money_flow"] = 0.0
        df["smart_money_accumulation"] = 0.5
        df["smart_money_distribution"] = 0.5
        df["exchange_flow_ratio"] = 1.0
        df["whale_concentration"] = 0.0
        return df

    def get_known_addresses(self, coin: str) -> list[dict[str, str]]:
        """Retorna lista de enderecos conhecidos de smart money para uma moeda.

        Args:
            coin: Simbolo da moeda (BTC, ETH)

        Returns:
            Lista de dicts com 'label' e 'address'
        """
        return KNOWN_SMART_MONEY.get(coin.upper(), [])

    def get_onchain_cache(self) -> dict[str, Any]:
        """Retorna dados on-chain em cache (se disponiveis).

        Returns:
            Dict com metricas on-chain obtidas nas ultimas chamadas
        """
        return self._onchain_cache.copy()
