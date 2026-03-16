"""Phase 10: Inteligencia Regional Multilingue para analise cripto.

Agrega sinais de mercados regionais de diferentes zonas geograficas,
estimando premiums regionais e efeitos de calendario a partir de dados
diarios OHLCV. Como temos apenas dados diarios, sessoes de trading sao
estimadas via padroes de dia-da-semana e distribuicao de volume.

Ref: Estudos academicos sobre Kimchi premium, efeito weekend em cripto,
e padroes sazonais de mercados asiaticos (USP/FGV, KAIST).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Total de features geradas: 24
_FEATURE_NAMES = [
    # Trading Session Analysis (12)
    "reg_asia_session_return",
    "reg_europe_session_return",
    "reg_us_session_return",
    "reg_asia_volume_ratio",
    "reg_europe_volume_ratio",
    "reg_us_volume_ratio",
    "reg_session_momentum",
    "reg_overnight_gap",
    "reg_cross_session_vol",
    "reg_asia_lead_indicator",
    "reg_weekend_effect",
    "reg_time_of_day_bias",
    # Regional Premium Indicators (6)
    "reg_kimchi_premium_proxy",
    "reg_japan_premium_proxy",
    "reg_latam_volume_signal",
    "reg_india_wazirx_proxy",
    "reg_global_spread",
    "reg_arbitrage_opportunity",
    # Calendar Effects (6)
    "reg_day_of_week",
    "reg_month_effect",
    "reg_quarter_end",
    "reg_lunar_new_year",
    "reg_golden_week",
    "reg_tax_season",
]


class RegionalIntelligence:
    """Inteligencia regional: sessoes, premiums e efeitos de calendario.

    Com dados diarios OHLCV, estimamos features de sessao via:
    - Dia da semana como proxy de atividade regional dominante
      (segunda = Asia lidera abertura; sexta = US fecha posicoes)
    - Distribuicao de volume rolling em janelas curtas/longas
    - Razao high-low intraday como proxy de volatilidade por sessao

    Gera 24 features com prefixo ``reg_``.
    """

    # Pesos de sessao por dia da semana (proxy empírico)
    # Seg: Asia abre primeiro apos weekend; Sex: US domina fechamento
    _ASIA_DOW_WEIGHTS = np.array([0.35, 0.30, 0.25, 0.25, 0.20, 0.15, 0.15])
    _EUROPE_DOW_WEIGHTS = np.array([0.30, 0.35, 0.35, 0.35, 0.30, 0.10, 0.10])
    _US_DOW_WEIGHTS = np.array([0.25, 0.30, 0.35, 0.35, 0.40, 0.10, 0.10])

    def __init__(self, lookback: int = 20, premium_lookback: int = 30):
        self.lookback = lookback
        self.premium_lookback = premium_lookback

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona 24 features regionais ao DataFrame.

        Args:
            df: DataFrame com colunas OHLCV (open, high, low, close, volume)
                e opcionalmente ``log_return`` e ``timestamp``.

        Returns:
            DataFrame com features ``reg_*`` adicionadas.
        """
        df = df.copy()
        n = len(df)
        logger.info("RegionalIntelligence: processando %d linhas", n)

        # Garantir retorno disponivel
        if "log_return" in df.columns:
            ret = df["log_return"].astype(np.float64)
        elif "close" in df.columns:
            ret = np.log(df["close"] / df["close"].shift(1))
            logger.info("Coluna 'log_return' ausente; calculada a partir de close.")
        else:
            logger.warning("Sem coluna close/log_return. Features regionais serao zero.")
            for col in _FEATURE_NAMES:
                df[col] = np.float32(0.0)
            return df

        # Extrair dia-da-semana (0=Monday..6=Sunday)
        dow = self._extract_dow(df)

        # --- Blocos de features ---
        df = self._session_analysis(df, ret, dow)
        df = self._regional_premiums(df, ret)
        df = self._calendar_effects(df, ret, dow)

        # Preencher NaN residual com 0
        for col in _FEATURE_NAMES:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(np.float32)
            else:
                df[col] = np.float32(0.0)

        logger.info("RegionalIntelligence: %d features adicionadas.", len(_FEATURE_NAMES))
        return df

    def get_feature_names(self) -> list[str]:
        """Retorna nomes de todas as features geradas."""
        return list(_FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dow(df: pd.DataFrame) -> pd.Series:
        """Extrai dia-da-semana do indice ou coluna timestamp."""
        if "timestamp" in df.columns:
            return pd.to_datetime(df["timestamp"]).dt.dayofweek
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.dayofweek.to_series(index=df.index)
        # Fallback: assume dias consecutivos a partir de segunda
        return pd.Series(np.arange(len(df)) % 7, index=df.index)

    @staticmethod
    def _safe_div(num: pd.Series, den: pd.Series, fill: float = 0.0) -> pd.Series:
        """Divisao segura substituindo inf/nan por *fill*."""
        result = num / den.replace(0, np.nan)
        return result.fillna(fill)

    # ------------------------------------------------------------------
    # 1. Trading Session Analysis (12 features)
    # ------------------------------------------------------------------

    def _session_analysis(
        self, df: pd.DataFrame, ret: pd.Series, dow: pd.Series,
    ) -> pd.DataFrame:
        lb = self.lookback

        # Pesos de sessao por dia da semana
        asia_w = dow.map(lambda d: self._ASIA_DOW_WEIGHTS[int(d) % 7])
        europe_w = dow.map(lambda d: self._EUROPE_DOW_WEIGHTS[int(d) % 7])
        us_w = dow.map(lambda d: self._US_DOW_WEIGHTS[int(d) % 7])

        # -- Retorno estimado por sessao (retorno diario * peso da sessao) --
        df["reg_asia_session_return"] = (ret * asia_w).rolling(lb).mean()
        df["reg_europe_session_return"] = (ret * europe_w).rolling(lb).mean()
        df["reg_us_session_return"] = (ret * us_w).rolling(lb).mean()

        # -- Volume ratio por sessao --
        vol = df.get("volume", pd.Series(np.ones(len(df)), index=df.index))
        vol_ma = vol.rolling(lb).mean()
        df["reg_asia_volume_ratio"] = self._safe_div(vol * asia_w, vol_ma, 1.0)
        df["reg_europe_volume_ratio"] = self._safe_div(vol * europe_w, vol_ma, 1.0)
        df["reg_us_volume_ratio"] = self._safe_div(vol * us_w, vol_ma, 1.0)

        # -- Session momentum: sessao com maior retorno absoluto domina --
        asia_abs = (ret * asia_w).abs()
        europe_abs = (ret * europe_w).abs()
        us_abs = (ret * us_w).abs()
        total_abs = asia_abs + europe_abs + us_abs + 1e-12
        # +1 = Asia domina, 0 = equilibrio, -1 = US domina
        df["reg_session_momentum"] = (asia_abs - us_abs) / total_abs

        # -- Overnight gap: diferenca entre close e proximo open --
        if "open" in df.columns and "close" in df.columns:
            gap = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
            df["reg_overnight_gap"] = gap.fillna(0.0)
        else:
            df["reg_overnight_gap"] = 0.0

        # -- Cross session volatility --
        # Diferenca de volatilidade: short-term (5d) vs medium-term (lookback)
        vol_short = ret.rolling(5).std()
        vol_med = ret.rolling(lb).std()
        df["reg_cross_session_vol"] = self._safe_div(vol_short - vol_med, vol_med, 0.0)

        # -- Asia lead indicator: retorno de segunda preve direcao semanal? --
        is_monday = (dow == 0).astype(float)
        monday_ret = ret * is_monday
        # Correlacao rolling entre retorno de segunda e retorno semanal
        weekly_ret = ret.rolling(5).sum()
        monday_ma = monday_ret.rolling(lb * 5).mean()
        weekly_ma = weekly_ret.rolling(lb * 5).mean()
        cov = (monday_ret * weekly_ret).rolling(lb * 5).mean() - monday_ma * weekly_ma
        std_m = monday_ret.rolling(lb * 5).std()
        std_w = weekly_ret.rolling(lb * 5).std()
        df["reg_asia_lead_indicator"] = self._safe_div(cov, std_m * std_w, 0.0).clip(-1, 1)

        # -- Weekend effect: media de retorno em fim de semana vs dia util --
        is_weekend = dow.isin([5, 6]).astype(float)
        is_weekday = 1.0 - is_weekend
        weekend_ret = (ret * is_weekend).rolling(lb * 4).sum()
        weekday_ret = (ret * is_weekday).rolling(lb * 4).sum()
        weekend_n = is_weekend.rolling(lb * 4).sum().clip(lower=1)
        weekday_n = is_weekday.rolling(lb * 4).sum().clip(lower=1)
        df["reg_weekend_effect"] = (weekend_ret / weekend_n) - (weekday_ret / weekday_n)

        # -- Time of day bias: proxy via intraday range position --
        if {"open", "high", "low", "close"}.issubset(df.columns):
            hl_range = df["high"] - df["low"]
            co_pos = df["close"] - df["open"]
            df["reg_time_of_day_bias"] = self._safe_div(co_pos, hl_range.clip(lower=1e-8), 0.0)
        else:
            df["reg_time_of_day_bias"] = 0.0

        return df

    # ------------------------------------------------------------------
    # 2. Regional Premium Indicators (6 features)
    # ------------------------------------------------------------------

    def _regional_premiums(self, df: pd.DataFrame, ret: pd.Series) -> pd.DataFrame:
        lb = self.premium_lookback

        vol = df.get("volume", pd.Series(np.ones(len(df)), index=df.index))

        # Volume-weighted price deviation como proxy de premium regional
        # Premium alto => volume anomalo + retorno extremo vs media
        ret_ma = ret.rolling(lb).mean()
        ret_std = ret.rolling(lb).std().clip(lower=1e-8)
        ret_z = (ret - ret_ma) / ret_std

        vol_ma = vol.rolling(lb).mean().clip(lower=1e-8)
        vol_ratio = vol / vol_ma

        # -- Kimchi premium proxy --
        # Combina z-score de retorno com anomalia de volume
        # Academicamente o premium medio eh ~1.24%; usamos z * vol_ratio como proxy
        df["reg_kimchi_premium_proxy"] = (ret_z * vol_ratio * 0.0124).clip(-0.1, 0.1)

        # -- Japan premium proxy --
        # Japao tem padroes similares mas menos pronunciados (~0.5% tipico)
        vol_ratio_fast = vol / vol.rolling(max(5, lb // 4)).mean().clip(lower=1e-8)
        df["reg_japan_premium_proxy"] = (ret_z * vol_ratio_fast * 0.005).clip(-0.05, 0.05)

        # -- LatAm volume signal --
        # Mercados latam correlacionam com alta volatilidade e volume tardio
        vol_change = vol.pct_change(5).fillna(0.0)
        ret_vol = ret.rolling(5).std().fillna(0.0)
        df["reg_latam_volume_signal"] = (vol_change * ret_vol).clip(-0.5, 0.5)

        # -- India (WazirX) premium proxy --
        # India premium tipicamente segue padroes de Kimchi mas com lag
        kimchi = df["reg_kimchi_premium_proxy"]
        df["reg_india_wazirx_proxy"] = kimchi.shift(1).fillna(0.0) * 0.8

        # -- Global spread: max premium - min premium --
        premium_cols = [
            "reg_kimchi_premium_proxy",
            "reg_japan_premium_proxy",
            "reg_india_wazirx_proxy",
        ]
        premiums = df[premium_cols]
        df["reg_global_spread"] = premiums.max(axis=1) - premiums.min(axis=1)

        # -- Arbitrage opportunity: spread excede threshold --
        spread_ma = df["reg_global_spread"].rolling(lb).mean()
        spread_std = df["reg_global_spread"].rolling(lb).std().clip(lower=1e-8)
        spread_z = (df["reg_global_spread"] - spread_ma) / spread_std
        df["reg_arbitrage_opportunity"] = (spread_z > 2.0).astype(np.float32)

        return df

    # ------------------------------------------------------------------
    # 3. Calendar Effects (6 features)
    # ------------------------------------------------------------------

    def _calendar_effects(
        self, df: pd.DataFrame, ret: pd.Series, dow: pd.Series,
    ) -> pd.DataFrame:
        # Extrair mes e dia do ano para sazonalidade
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = df.index.to_series()
        else:
            # Fallback: sem info temporal, usar features neutras
            for col in _FEATURE_NAMES[18:]:  # ultimas 6 features
                df[col] = np.float32(0.0)
            # Ainda gera day_of_week usando dow
            angle = 2.0 * np.pi * dow / 7.0
            df["reg_day_of_week"] = np.sin(angle).astype(np.float32)
            return df

        month = ts.dt.month
        day = ts.dt.day
        day_of_year = ts.dt.dayofyear

        # -- Day of week: codificacao ciclica sin --
        angle_dow = 2.0 * np.pi * dow / 7.0
        df["reg_day_of_week"] = np.sin(angle_dow)

        # -- Month effect: sazonalidade mensal (sin ciclico) --
        angle_month = 2.0 * np.pi * month / 12.0
        df["reg_month_effect"] = np.sin(angle_month)

        # -- Quarter end rebalancing --
        # Ultimos 5 dias de cada trimestre (Mar, Jun, Set, Dez)
        is_quarter_month = month.isin([3, 6, 9, 12])
        is_late_month = day >= 25
        df["reg_quarter_end"] = (is_quarter_month & is_late_month).astype(np.float32)

        # -- Lunar New Year effect (Jan 20 - Feb 20 aprox) --
        # Mercados asiaticos podem ter pressao de venda pre-feriado
        lunar_window = ((month == 1) & (day >= 20)) | ((month == 2) & (day <= 20))
        # Intensidade baseada no retorno medio historico neste periodo
        lunar_mask = lunar_window.astype(float)
        lunar_ret_avg = (ret * lunar_mask).rolling(60).mean()
        df["reg_lunar_new_year"] = lunar_mask * lunar_ret_avg.abs().clip(upper=0.05)

        # -- Golden Week effect --
        # Japao: 29 Abr - 5 Mai; China: 1-7 Out
        golden_japan = (month == 4) & (day >= 29) | (month == 5) & (day <= 5)
        golden_china = (month == 10) & (day <= 7)
        golden_mask = (golden_japan | golden_china).astype(float)
        golden_ret_avg = (ret * golden_mask).rolling(60).mean()
        df["reg_golden_week"] = golden_mask * golden_ret_avg.abs().clip(upper=0.05)

        # -- Tax season selling pressure --
        # EUA: Abr 1-15 (Tax Day); Japao: Feb 16 - Mar 15
        tax_us = (month == 4) & (day <= 15)
        tax_japan = ((month == 2) & (day >= 16)) | ((month == 3) & (day <= 15))
        tax_mask = (tax_us | tax_japan).astype(float)
        # Sinal negativo indica pressao vendedora tipica
        tax_ret_avg = (ret * tax_mask).rolling(60).mean()
        df["reg_tax_season"] = tax_mask * tax_ret_avg.clip(-0.05, 0.0)

        return df
