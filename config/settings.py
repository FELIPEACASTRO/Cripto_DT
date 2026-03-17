"""Configuracao central do sistema de previsao de criptomoedas."""

from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    coins: list[str] = field(default_factory=lambda: [
        "BTC", "ETH", "BNB", "SOL", "XRP",
        "ADA", "DOGE", "AVAX", "DOT", "MATIC",
    ])
    quote_currency: str = "USDT"
    timeframes: list[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    history_days: int = 730  # 2 anos
    exchange: str = "binance"
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    rate_limit_sleep: float = 1.5  # segundos entre requests

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"


@dataclass
class FeatureConfig:
    lookback_window: int = 60
    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    ema_periods: list[int] = field(default_factory=lambda: [9, 21, 50, 200])
    sma_periods: list[int] = field(default_factory=lambda: [20, 50])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bbands_period: int = 20
    bbands_std: int = 2
    atr_period: int = 14
    stoch_period: int = 14
    williams_period: int = 14
    volume_sma_period: int = 20
    btc_correlation_window: int = 20
    # Wavelet denoising
    use_wavelet: bool = True
    wavelet_name: str = "db4"
    wavelet_threshold_mode: str = "soft"
    # Sentiment
    use_sentiment: bool = True
    # CEEMDAN decomposition
    use_decomposition: bool = True
    decomposition_method: str = "ceemdan"  # "ceemdan", "eemd", "emd"
    n_imfs: int = 6
    # Regime detection
    use_regime: bool = True
    n_regimes: int = 3
    # On-chain data
    use_onchain: bool = True
    # Multi-timeframe fusion
    use_multi_timeframe: bool = True
    # Volatility features (GARCH/HAR)
    use_volatility: bool = True
    use_har: bool = True
    # Bubble detection (GSADF)
    use_bubble: bool = True
    # Cross-crypto features
    use_cross_crypto: bool = True
    # Anomaly detection (Z-score)
    use_anomaly: bool = True
    # Macro variables (VIX, DXY, S&P500, etc.)
    use_macro: bool = True
    # NLP sentiment (news via Sentence-BERT)
    use_nlp_sentiment: bool = True
    # Whale monitoring
    use_whale: bool = True
    # Features da pesquisa Oriental (IA Leste Asiatico)
    use_regional_markets: bool = True
    use_chart_vision: bool = True
    use_blockchain_nlp: bool = True
    use_multilingual_sentiment: bool = True
    # Phase 6: NLP Sentiment (FinBERT + Twitter-RoBERTa)
    use_finbert_sentiment: bool = True
    # Phase 8: Chart Pattern Detection
    use_chart_patterns: bool = True
    # Phase 10: Regional Intelligence
    use_regional_intelligence: bool = True
    # Market Context Memory (aiagentstore.ai - Jina AI)
    embedding_backend: str = "auto"  # "auto", "jina", "sentence-transformers", "tfidf"
    embedding_model: str = "auto"  # "auto" = seleciona baseado no backend
    max_context_events: int = 10000
    # Entropy filter (noise reduction)
    use_entropy_filter: bool = True
    # Microstructure features (Hurst, fractal, VPIN, Amihud)
    use_microstructure: bool = True
    # Visual Pattern Analyzer (GAF, candlestick, S/R, volume profile)
    use_visual_patterns: bool = True
    # Funding Rate & Open Interest proxy
    use_funding_oi: bool = True
    # Feature selection
    feature_selection_method: str = "boruta"  # "boruta", "l1", "mutual_info", "none"


@dataclass
class LSTMConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64])
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 10
    grad_clip_max_norm: float = 1.0
    mc_dropout_samples: int = 30
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    loss_type: str = "madl"  # "madl", "directional_huber" ou "huber"
    madl_alpha: float = 2.0  # Peso da penalidade direcional no MADL


@dataclass
class XGBoostConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    use_direction: bool = True  # treinar tambem classificador direcional


@dataclass
class HelformerConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 80
    early_stop_patience: int = 10


@dataclass
class EMGNNConfig:
    hidden_size: int = 64
    n_scales: int = 3
    scale_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 60
    early_stop_patience: int = 8


@dataclass
class MDNConfig:
    hidden_size: int = 128
    n_mixtures: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 80
    early_stop_patience: int = 10
    dropout: float = 0.2


@dataclass
class ConformalConfig:
    alpha: float = 0.1  # 90% coverage
    adaptive_gamma: float = 0.005  # ACI learning rate


@dataclass
class TradingConfig:
    initial_balance: float = 10000.0
    commission: float = 0.001
    max_position: float = 1.0
    # PPO hyperparams
    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99
    ppo_clip_range: float = 0.2
    ppo_total_timesteps: int = 100_000


@dataclass
class HyperoptConfig:
    n_trials: int = 50
    timeout: int = 3600  # segundos
    use_hyperopt: bool = False  # Desabilitado por default (demorado)


@dataclass
class LightGBMConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    use_direction: bool = True


@dataclass
class TCNConfig:
    num_channels: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    kernel_size: int = 3
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 10


@dataclass
class BNNConfig:
    hidden_size: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    early_stop_patience: int = 20
    n_samples: int = 30
    kl_weight: float = 1e-3


@dataclass
class CNNLSTMConfig:
    cnn_channels: list[int] = field(default_factory=lambda: [64, 128])
    lstm_hidden: int = 64
    lstm_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 10


@dataclass
class MambaConfig:
    d_model: int = 64
    d_state: int = 16
    n_layers: int = 4
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 10
    mc_dropout_samples: int = 30
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    grad_clip_max_norm: float = 1.0
    loss_type: str = "madl"
    madl_alpha: float = 2.0
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: str = "auto"


@dataclass
class EvidentialConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 150
    early_stop_patience: int = 15
    evidence_coeff: float = 0.1


@dataclass
class DualPredictionConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 128])
    head_size: int = 64
    dropout: float = 0.2
    alpha: float = 0.6
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    early_stop_patience: int = 10
    classification_threshold: float = 0.6


@dataclass
class EvolutionaryEnsembleConfig:
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.1
    crossover_rate: float = 0.8
    elitism: int = 5
    tournament_k: int = 3


# ============================================================
# Phase 7: MoE Gating + Reasoning
# ============================================================

@dataclass
class MoEEnsembleConfig:
    """MoE Gating Network para roteamento inteligente de ensemble."""
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64])
    top_k: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    early_stop_patience: int = 20
    dropout: float = 0.2
    diversity_weight: float = 0.01  # Penalidade por concentracao em 1 modelo
    batch_size: int = 64


# ============================================================
# Phase 9: Synthetic Data & Augmentation
# ============================================================

@dataclass
class MarketGANConfig:
    """MarketGAN para geracao de dados sinteticos de mercado."""
    latent_dim: int = 32
    seq_len: int = 60
    n_features: int = 5  # OHLCV returns
    gen_channels: list[int] = field(default_factory=lambda: [64, 128, 64])
    disc_channels: list[int] = field(default_factory=lambda: [64, 128, 64])
    lr_generator: float = 1e-4
    lr_discriminator: float = 1e-4
    batch_size: int = 64
    n_critic: int = 5  # WGAN-GP: passos do critico por passo do gerador
    gp_weight: float = 10.0  # Gradient penalty weight
    max_epochs: int = 500
    n_synthetic_samples: int = 500


# ============================================================
# Phase 5: Time Series Foundation Models
# ============================================================

@dataclass
class ChronosConfig:
    """Amazon Chronos-Bolt: zero-shot/fine-tuned time series forecasting."""
    model_name: str = "amazon/chronos-bolt-small"
    context_length: int = 512
    prediction_length: int = 1
    num_samples: int = 20
    batch_size: int = 32
    fine_tune_epochs: int = 10
    fine_tune_lr: float = 1e-4
    device: str = "auto"


@dataclass
class TTMConfig:
    """IBM Tiny Time Mixers: lightweight MLP-Mixer for time series."""
    model_name: str = "ibm/TTM"
    context_length: int = 512
    prediction_length: int = 1
    patch_length: int = 64
    num_input_channels: int = 1
    fine_tune_epochs: int = 10
    fine_tune_lr: float = 1e-4
    batch_size: int = 32


@dataclass
class MOIRAIConfig:
    """Salesforce MOIRAI: multivariate foundation model."""
    model_name: str = "salesforce/moirai-1.0-R-small"
    context_length: int = 512
    prediction_length: int = 1
    num_samples: int = 20
    patch_size: str = "auto"
    fine_tune_epochs: int = 10
    fine_tune_lr: float = 1e-4
    batch_size: int = 32


# ============================================================
# Phase 6: Financial NLP & Sentiment
# ============================================================

@dataclass
class NLPSentimentConfig:
    """Configuracao para features de NLP/sentiment (FinBERT + Twitter-RoBERTa)."""
    finbert_model: str = "ProsusAI/finbert"
    twitter_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    batch_size: int = 32
    max_length: int = 128
    cache_results: bool = True
    use_proxy_features: bool = True  # Gerar proxies de OHLCV quando NLP indisponivel


@dataclass
class NewsScraperConfig:
    max_articles: int = 50
    cache_hours: int = 1
    use_cryptocompare: bool = True
    use_reddit: bool = True
    use_rss: bool = True
    rate_limit_sleep: float = 1.0


@dataclass
class SignalGeneratorConfig:
    risk_per_trade: float = 0.02
    max_position: float = 0.10
    min_confidence: float = 0.6
    min_model_agreement: float = 0.5
    atr_multiplier_sl: float = 2.0
    atr_multiplier_tp: float = 3.0
    max_portfolio_exposure: float = 0.5


@dataclass
class RealtimeConfig:
    update_interval_minutes: int = 60
    use_news_scraper: bool = True
    use_market_context: bool = True
    use_signal_generator: bool = True


@dataclass
class BacktestConfig:
    """Parametros de configuracao do backtesting."""
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001       # 0.1% por trade
    slippage_pct: float = 0.0005        # 0.05% slippage estimado
    max_position_pct: float = 0.10      # 10% do capital por posicao
    use_stop_loss: bool = True
    use_take_profit: bool = True
    risk_free_rate: float = 0.02        # Taxa livre de risco anualizada


@dataclass
class TrainingConfig:
    walk_forward_splits: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    prediction_horizon: int = 1  # 1 passo a frente
    target_type: str = "log_return"
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models")
    reports_dir: Path = field(default_factory=lambda: BASE_DIR / "reports")
    # Modelos a treinar
    use_helformer: bool = True
    use_emgnn: bool = True
    use_mdn: bool = True
    use_conformal: bool = True
    use_rl_trading: bool = True
    # Novos modelos (pesquisa 2024-2026)
    use_lightgbm: bool = True
    use_random_forest: bool = True
    use_svm: bool = True
    use_cnn_lstm: bool = True
    use_tcn: bool = True
    use_bnn: bool = True
    # Modelos da pesquisa Oriental (IA Leste Asiatico)
    use_mamba: bool = True
    use_evidential: bool = True
    use_dual_prediction: bool = True
    use_evolutionary_ensemble: bool = True
    # Phase 5: Foundation Models
    use_chronos: bool = True
    use_ttm: bool = True
    use_moirai: bool = True
    # Phase 6: NLP Sentiment
    use_nlp_finbert: bool = True
    # Phase 7: MoE Gating
    use_moe_ensemble: bool = True
    # Phase 9: Synthetic Data
    use_market_gan: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    helformer: HelformerConfig = field(default_factory=HelformerConfig)
    emgnn: EMGNNConfig = field(default_factory=EMGNNConfig)
    mdn: MDNConfig = field(default_factory=MDNConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    hyperopt: HyperoptConfig = field(default_factory=HyperoptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Novos modelos (pesquisa 2024-2026)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    tcn: TCNConfig = field(default_factory=TCNConfig)
    bnn: BNNConfig = field(default_factory=BNNConfig)
    cnn_lstm: CNNLSTMConfig = field(default_factory=CNNLSTMConfig)
    # Modelos da pesquisa Oriental (IA Leste Asiatico)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    evidential: EvidentialConfig = field(default_factory=EvidentialConfig)
    dual_prediction: DualPredictionConfig = field(default_factory=DualPredictionConfig)
    evolutionary_ensemble: EvolutionaryEnsembleConfig = field(default_factory=EvolutionaryEnsembleConfig)
    # Phase 5: Foundation Models
    chronos: ChronosConfig = field(default_factory=ChronosConfig)
    ttm: TTMConfig = field(default_factory=TTMConfig)
    moirai: MOIRAIConfig = field(default_factory=MOIRAIConfig)
    # Phase 6: NLP Sentiment
    nlp_sentiment: NLPSentimentConfig = field(default_factory=NLPSentimentConfig)
    # Phase 7: MoE Gating
    moe_ensemble: MoEEnsembleConfig = field(default_factory=MoEEnsembleConfig)
    # Phase 9: Synthetic Data
    market_gan: MarketGANConfig = field(default_factory=MarketGANConfig)
    # Melhorias (aiagentstore.ai)
    news_scraper: NewsScraperConfig = field(default_factory=NewsScraperConfig)
    signal_generator: SignalGeneratorConfig = field(default_factory=SignalGeneratorConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


# Instancia global de configuracao
config = Config()
