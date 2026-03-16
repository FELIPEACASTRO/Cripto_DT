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
    # Market Context Memory (aiagentstore.ai - Jina AI)
    embedding_backend: str = "auto"  # "auto", "jina", "sentence-transformers", "tfidf"
    embedding_model: str = "auto"  # "auto" = seleciona baseado no backend
    max_context_events: int = 10000
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
    # Melhorias (aiagentstore.ai)
    news_scraper: NewsScraperConfig = field(default_factory=NewsScraperConfig)
    signal_generator: SignalGeneratorConfig = field(default_factory=SignalGeneratorConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)


# Instancia global de configuracao
config = Config()
