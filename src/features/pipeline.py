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

    Pipeline completo (29 etapas):
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
    17. Smart Money features (Fase 3 - flow, acumulacao, distribuicao)
    18. Narrative Detection (Fase 3 - 8 narrativas cripto)
    19. Advanced Crawler features (Fase 3 - news count, buzz, fear&greed)
    20. Regional Market features (Kimchi premium, sessoes regionais)
    21. Chart Vision features (VISTA-inspired candlestick patterns)
    22. Blockchain NLP features (crypto sentiment lexicon)
    23. Multilingual Sentiment (CN, KR, JP, VN, ID, AR)
    24. Chart Pattern Detection (Phase 8 — H&S, triangles, flags, etc.)
    25. Regional Intelligence (Phase 10 — sessions, premiums, calendar)
    26. FinBERT + Twitter-RoBERTa Sentiment (Phase 6 NLP)
    27. Timeframe Fusion (agrega 1h/4h em features diarias)
    28. Cross-crypto features (retornos cruzados) — chamado externamente
    29. Limpeza final (dropna) + Feature Selection (no trainer)
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

        # --- Modulos Fase 3 (aiagentstore.ai) ---
        self._smart_money = None
        self._narrative_detector = None
        self._advanced_crawler = None

        Cls = _safe_import("src.data.smart_money", "SmartMoneyTracker")
        if Cls:
            self._smart_money = Cls()

        Cls = _safe_import("src.data.narrative_detector", "NarrativeDetector")
        if Cls:
            self._narrative_detector = Cls()

        Cls = _safe_import("src.data.advanced_crawler", "AdvancedNewsCrawler")
        if Cls:
            self._advanced_crawler = Cls()

        # --- Modulos da pesquisa Oriental (IA Leste Asiatico) ---
        self._regional_markets = None
        self._chart_vision = None
        self._blockchain_nlp = None
        self._multilingual_sentiment = None

        if config.features.use_regional_markets:
            Cls = _safe_import("src.features.regional_markets", "RegionalMarketFeatures")
            if Cls:
                self._regional_markets = Cls()

        if config.features.use_chart_vision:
            Cls = _safe_import("src.features.chart_vision", "ChartVisionFeatures")
            if Cls:
                self._chart_vision = Cls()

        if config.features.use_blockchain_nlp:
            Cls = _safe_import("src.features.blockchain_nlp", "BlockchainNLPFeatures")
            if Cls:
                self._blockchain_nlp = Cls()

        if config.features.use_multilingual_sentiment:
            Cls = _safe_import("src.data.multilingual_sentiment", "MultilingualSentimentAnalyzer")
            if Cls:
                self._multilingual_sentiment = Cls()

        # --- Timeframe Fusion (multi-timeframe) ---
        self._timeframe_fusion = None
        if config.features.use_multi_timeframe:
            Cls = _safe_import("src.features.timeframe_fusion", "TimeframeFusion")
            if Cls:
                self._timeframe_fusion = Cls(config)

        # --- Phase 6: NLP Sentiment (FinBERT + Twitter-RoBERTa) ---
        self._finbert_sentiment = None
        if config.features.use_finbert_sentiment:
            Cls = _safe_import("src.features.finbert_sentiment", "NLPSentimentFeatures")
            if Cls:
                self._finbert_sentiment = Cls()

        # --- Phase 8: Chart Pattern Detection ---
        self._chart_patterns = None
        if config.features.use_chart_patterns:
            Cls = _safe_import("src.features.chart_patterns", "ChartPatternDetector")
            if Cls:
                self._chart_patterns = Cls()

        # --- Phase 10: Regional Intelligence ---
        self._regional_intelligence = None
        if config.features.use_regional_intelligence:
            Cls = _safe_import("src.features.regional_intelligence", "RegionalIntelligence")
            if Cls:
                self._regional_intelligence = Cls()

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
        df_1h: pd.DataFrame | None = None,
        df_4h: pd.DataFrame | None = None,
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

        # 17. Smart Money features (Fase 3)
        if self._smart_money is not None and coin:
            try:
                df = self._smart_money.add_smart_money_features(df, coin)
            except Exception as e:
                logger.warning(f"Smart Money falhou para {coin}: {e}")

        # 18. Narrative Detection features (Fase 3)
        if self._narrative_detector is not None and coin:
            try:
                df = self._narrative_detector.add_narrative_features(df, coin)
            except Exception as e:
                logger.warning(f"Narrative detection falhou para {coin}: {e}")

        # 19. Advanced Crawler features (Fase 3)
        if self._advanced_crawler is not None and coin:
            try:
                df = self._advanced_crawler.add_crawler_features(df, coin)
            except Exception as e:
                logger.warning(f"Advanced crawler falhou para {coin}: {e}")

        # 20. Regional Market features (Kimchi premium, sessoes, spreads)
        if self._regional_markets is not None:
            try:
                df = self._regional_markets.transform(df)
            except Exception as e:
                logger.warning(f"Regional markets falhou: {e}")

        # 21. Chart Vision features (padroes de candlestick, VISTA-inspired)
        if self._chart_vision is not None:
            try:
                df = self._chart_vision.transform(df)
            except Exception as e:
                logger.warning(f"Chart vision falhou: {e}")

        # 22. Blockchain NLP features (sentiment lexicon, buzz indicators)
        if self._blockchain_nlp is not None:
            try:
                df = self._blockchain_nlp.transform(df)
            except Exception as e:
                logger.warning(f"Blockchain NLP falhou: {e}")

        # 23. Multilingual Sentiment (CN, KR, JP, VN, ID, AR)
        if self._multilingual_sentiment is not None and coin:
            try:
                df = self._multilingual_sentiment.add_multilingual_features(df, coin)
            except Exception as e:
                logger.warning(f"Multilingual sentiment falhou para {coin}: {e}")

        # 24. Chart Pattern Detection (Phase 8 — head&shoulders, triangles, etc.)
        if self._chart_patterns is not None:
            try:
                df = self._chart_patterns.transform(df)
            except Exception as e:
                logger.warning(f"Chart patterns falhou: {e}")

        # 25. Regional Intelligence (Phase 10 — session analysis, premiums, calendar)
        if self._regional_intelligence is not None:
            try:
                df = self._regional_intelligence.transform(df)
            except Exception as e:
                logger.warning(f"Regional intelligence falhou: {e}")

        # 26. FinBERT + Twitter-RoBERTa Sentiment (Phase 6 NLP)
        if self._finbert_sentiment is not None:
            try:
                df = self._finbert_sentiment.transform(df)
            except Exception as e:
                logger.warning(f"FinBERT sentiment falhou: {e}")

        # 27. Timeframe Fusion (agrega 1h/4h em features diarias)
        if self._timeframe_fusion is not None and (df_1h is not None or df_4h is not None):
            try:
                df = self._timeframe_fusion.merge_timeframes(df, df_4h=df_4h, df_1h=df_1h)
            except Exception as e:
                logger.warning(f"Timeframe fusion falhou: {e}")

        # 29. Limpeza final — estrategia inteligente para preservar dados
        n_before = len(df)
        feature_cols = self.get_feature_columns(df)

        # 20a. Remover colunas com >50% NaN (features que falharam completamente)
        nan_ratio = df[feature_cols].isna().mean()
        cols_to_drop = nan_ratio[nan_ratio > 0.5].index.tolist()
        if cols_to_drop:
            logger.info(f"  Removendo {len(cols_to_drop)} colunas com >50% NaN: {cols_to_drop[:5]}...")
            df = df.drop(columns=cols_to_drop)

        # 20b. Forward-fill + backward-fill para NaN restantes (warm-up de indicadores)
        feature_cols = self.get_feature_columns(df)
        df[feature_cols] = df[feature_cols].ffill().bfill()

        # 20c. Preencher qualquer NaN restante com 0
        df[feature_cols] = df[feature_cols].fillna(0)

        # 20d. Remover primeiras linhas onde target pode ser NaN
        if "target" in df.columns:
            df = df.dropna(subset=["target"]).reset_index(drop=True)
        else:
            df = df.dropna().reset_index(drop=True)

        n_removed = n_before - len(df)
        logger.info(
            f"  Removidas {n_removed} linhas (warm-up/target). "
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

        # Alinhar colunas: garantir que todas as moedas tenham as mesmas features
        if results:
            all_feature_cols = set()
            for coin, df in results.items():
                all_feature_cols.update(self.get_feature_columns(df))

            for coin in results:
                missing = all_feature_cols - set(results[coin].columns)
                if missing:
                    logger.info(f"  {coin}: preenchendo {len(missing)} features ausentes com 0")
                    for col in missing:
                        results[coin][col] = 0.0

            # Atualizar feature_columns com a uniao
            self._feature_columns = sorted(all_feature_cols)
            logger.info(f"Feature columns finais: {len(self._feature_columns)} features (alinhadas)")

        return results
