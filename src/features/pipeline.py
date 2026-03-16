"""Pipeline de feature engineering: orquestra todos os modulos de features."""

import logging

import pandas as pd

from config.settings import Config, config as default_config
from src.data.preprocessor import DataPreprocessor
from src.data.sentiment import SentimentCollector
from src.features.technical import TechnicalFeatures
from src.features.lag_features import LagFeatures
from src.features.market import MarketFeatures
from src.features.wavelet import WaveletDenoiser

logger = logging.getLogger(__name__)


# --- Importacoes seguras para componentes opcionais ---

def _safe_import(module_path: str, class_name: str, dep_name: str = ""):
    """Importa uma classe de forma segura, retornando None se falhar."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        if dep_name:
            logger.debug(f"{class_name} nao disponivel (instale {dep_name})")
        return None


class FeaturePipeline:
    """Orquestra toda a engenharia de features.

    Pipeline completo (18 etapas):
    1. Preprocessamento (clean, returns, target)
    2. Wavelet denoising
    3. Indicadores tecnicos (MACD, RSI, Bollinger, Ichimoku, ADX, etc.)
    4. Lag features (retornos defasados, rolling stats)
    5. Market features (correlacao BTC, dominancia)
    6. Sentiment (Fear & Greed Index)
    7. CEEMDAN decomposition
    8. Regime detection (HMM)
    9. On-chain features
    10. Volatility features (GARCH, Parkinson, Garman-Klass)
    11. HAR volatility (realized variance multi-escala)
    12. Bubble detection (GSADF)
    13. Anomaly detection (Z-score)
    14. Macro variables (VIX, DXY, S&P500, etc.)
    15. Whale monitoring (volume anomalies)
    16. NLP sentiment (news analysis)
    17. Cross-crypto features (retornos cruzados) — chamado externamente
    18. Limpeza final (dropna)
    """

    # Colunas que NAO sao features (metadata e targets)
    NON_FEATURE_COLS = {
        "timestamp", "open", "high", "low", "close", "volume",
        "log_return", "pct_return", "direction", "target",
        # Colunas intermediarias do wavelet
        "close_denoised", "high_denoised", "low_denoised", "open_denoised",
    }

    def __init__(self, config: Config = default_config):
        self.config = config
        self.preprocessor = DataPreprocessor()
        self.wavelet = WaveletDenoiser(
            wavelet=config.features.wavelet_name,
            threshold_mode=config.features.wavelet_threshold_mode,
        )
        self.technical = TechnicalFeatures(config.features)
        self.lag = LagFeatures(config.features)
        self.market = MarketFeatures(config.features)
        self.sentiment = SentimentCollector()
        self._feature_columns: list[str] = []
        self._fg_df: pd.DataFrame | None = None

        # --- Componentes opcionais (Fase 1) ---
        self._decomposer = None
        self._regime_detector = None
        self._onchain = None

        if config.features.use_decomposition:
            Cls = _safe_import("src.features.decomposition", "SignalDecomposer", "EMD-signal")
            if Cls:
                self._decomposer = Cls(
                    method=config.features.decomposition_method,
                    n_imfs=config.features.n_imfs,
                )

        if config.features.use_regime:
            Cls = _safe_import("src.features.regime", "RegimeDetector", "hmmlearn")
            if Cls:
                self._regime_detector = Cls(n_regimes=config.features.n_regimes)

        if config.features.use_onchain:
            Cls = _safe_import("src.data.onchain", "OnchainCollector")
            if Cls:
                self._onchain = Cls()

        # --- Novos componentes (Pesquisa 2024-2026) ---
        self._volatility = None
        self._har = None
        self._bubble = None
        self._anomaly = None
        self._macro = None
        self._whale = None
        self._nlp_sentiment = None

        if config.features.use_volatility:
            Cls = _safe_import("src.features.volatility", "VolatilityFeatures")
            if Cls:
                self._volatility = Cls()

        if config.features.use_har:
            Cls = _safe_import("src.features.har_volatility", "HARVolatility")
            if Cls:
                self._har = Cls()

        if config.features.use_bubble:
            Cls = _safe_import("src.features.bubble", "BubbleDetector")
            if Cls:
                self._bubble = Cls()

        if config.features.use_anomaly:
            Cls = _safe_import("src.features.anomaly", "AnomalyDetector")
            if Cls:
                self._anomaly = Cls()

        if config.features.use_macro:
            Cls = _safe_import("src.data.macro", "MacroCollector")
            if Cls:
                self._macro = Cls()

        if config.features.use_whale:
            Cls = _safe_import("src.data.whale_monitor", "WhaleMonitor")
            if Cls:
                self._whale = Cls()

        if config.features.use_nlp_sentiment:
            Cls = _safe_import("src.data.nlp_sentiment", "NLPSentimentAnalyzer")
            if Cls:
                self._nlp_sentiment = Cls()

    def _get_fear_greed(self) -> pd.DataFrame:
        """Busca Fear & Greed uma unica vez e cacheia."""
        if self._fg_df is None:
            if self.config.features.use_sentiment:
                self._fg_df = self.sentiment.fetch_fear_greed(
                    days=self.config.data.history_days
                )
            else:
                self._fg_df = pd.DataFrame()
        return self._fg_df

    def transform(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame | None = None,
        is_btc: bool = False,
        coin: str = "",
    ) -> pd.DataFrame:
        """Aplica pipeline completo de features."""
        logger.info(f"Feature pipeline: {len(df)} linhas de entrada")

        # 1. Preprocessamento
        df = self.preprocessor.prepare(
            df,
            horizon=self.config.training.prediction_horizon,
            target_type=self.config.training.target_type,
        )

        # 2. Wavelet denoising
        if self.config.features.use_wavelet:
            df = self.wavelet.transform(df)

        # 3. Indicadores tecnicos
        df = self.technical.transform(df)

        # 4. Lag features
        df = self.lag.transform(df)

        # 5. Market features
        btc_ref = None if is_btc else btc_df
        df = self.market.transform(df, btc_df=btc_ref)

        # 6. Sentiment features (Fear & Greed Index)
        fg_df = self._get_fear_greed()
        if not fg_df.empty:
            df = self.sentiment.add_sentiment_features(df, fg_df)

        # 7. CEEMDAN decomposition features
        if self._decomposer is not None:
            try:
                df = self._decomposer.transform(df)
            except Exception as e:
                logger.warning(f"Decomposition falhou: {e}")

        # 8. Regime detection features
        if self._regime_detector is not None and "log_return" in df.columns:
            try:
                returns = df["log_return"].dropna().values
                self._regime_detector.fit(returns)
                df = self._regime_detector.transform(df)
            except Exception as e:
                logger.warning(f"Regime detection falhou: {e}")

        # 9. On-chain features
        if self._onchain is not None and coin:
            try:
                df = self._onchain.add_onchain_features(df, coin)
            except Exception as e:
                logger.warning(f"On-chain features falhou para {coin}: {e}")

        # 10. Volatility features (GARCH, Parkinson, Garman-Klass)
        if self._volatility is not None:
            try:
                df = self._volatility.transform(df)
            except Exception as e:
                logger.warning(f"Volatility features falhou: {e}")

        # 11. HAR volatility features
        if self._har is not None:
            try:
                df = self._har.transform(df)
            except Exception as e:
                logger.warning(f"HAR volatility falhou: {e}")

        # 12. Bubble detection (GSADF)
        if self._bubble is not None:
            try:
                df = self._bubble.transform(df)
            except Exception as e:
                logger.warning(f"Bubble detection falhou: {e}")

        # 13. Anomaly detection (Z-score)
        if self._anomaly is not None:
            try:
                df = self._anomaly.transform(df)
            except Exception as e:
                logger.warning(f"Anomaly detection falhou: {e}")

        # 14. Macro variables
        if self._macro is not None:
            try:
                df = self._macro.add_macro_features(df)
            except Exception as e:
                logger.warning(f"Macro features falhou: {e}")

        # 15. Whale monitoring
        if self._whale is not None:
            try:
                df = self._whale.add_whale_features(df, coin)
            except Exception as e:
                logger.warning(f"Whale features falhou para {coin}: {e}")

        # 16. NLP sentiment (news)
        if self._nlp_sentiment is not None and coin:
            try:
                df = self._nlp_sentiment.add_nlp_features(df, coin)
            except Exception as e:
                logger.warning(f"NLP sentiment falhou para {coin}: {e}")

        # 17. Limpeza final — remover linhas com NaN (warm-up dos indicadores)
        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        n_removed = n_before - len(df)
        logger.info(
            f"  Removidas {n_removed} linhas (warm-up). "
            f"Restam {len(df)} linhas com {len(self.get_feature_columns(df))} features"
        )

        # Cache feature columns
        self._feature_columns = self.get_feature_columns(df)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Retorna lista de colunas que sao features (exclui metadata/target)."""
        return [c for c in df.columns if c not in self.NON_FEATURE_COLS]

    @property
    def feature_columns(self) -> list[str]:
        """Colunas de features (disponivel apos transform)."""
        return self._feature_columns

    def transform_all(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Transforma dados de multiplas moedas."""
        # Preprocessar BTC primeiro (para correlacao)
        btc_df = None
        if "BTC" in data and not data["BTC"].empty:
            btc_df = self.preprocessor.prepare(
                data["BTC"],
                horizon=self.config.training.prediction_horizon,
                target_type=self.config.training.target_type,
            )

        results = {}
        for coin, df in data.items():
            if df.empty:
                logger.warning(f"Dados vazios para {coin}, pulando")
                continue
            logger.info(f"Processando features para {coin}...")
            is_btc = coin == "BTC"
            results[coin] = self.transform(
                df, btc_df=btc_df, is_btc=is_btc, coin=coin
            )

        # Cross-crypto features (requer dados de todas as moedas)
        if self.config.features.use_cross_crypto:
            try:
                Cls = _safe_import("src.features.cross_crypto", "CrossCryptoFeatures")
                if Cls:
                    cross = Cls()
                    for coin in list(results.keys()):
                        try:
                            results[coin] = cross.transform(results, coin)
                        except Exception as e:
                            logger.warning(f"Cross-crypto falhou para {coin}: {e}")
            except Exception as e:
                logger.warning(f"Cross-crypto features falhou: {e}")

        return results
