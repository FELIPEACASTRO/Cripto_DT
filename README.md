# Cripto_DT — Sistema de Predicao de Criptomoedas com IA/ML

Sistema avancado de predicao de precos de criptomoedas usando 16 modelos de Machine Learning, pipeline de 18 features, arquitetura multi-agente e geracao de sinais de trading.

## Arquitetura

```
┌──────────────────────────────────────────────────────────────────┐
│                    AgentOrchestrator                             │
│  ┌──────────┐  ┌─────────────┐  ┌────────────┐  ┌───────────┐ │
│  │DataAgent │→│AnalystAgent │→│TraderAgent │→│ RiskAgent │ │
│  └──────────┘  └─────────────┘  └────────────┘  └───────────┘ │
└──────────────────────────────────────────────────────────────────┘
       │                │                │               │
   ┌───┴───┐      ┌─────┴─────┐    ┌────┴────┐    ┌─────┴─────┐
   │ OHLCV │      │ 18-Stage  │    │16 Models│    │ 6 Risk    │
   │ News  │      │ Feature   │    │Ensemble │    │ Layers    │
   │ Chain │      │ Pipeline  │    │Signals  │    │ Filters   │
   └───────┘      └───────────┘    └─────────┘    └───────────┘
```

## Criptomoedas Suportadas

BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, MATIC

## Modelos (16)

| Categoria | Modelos |
|-----------|---------|
| Tree-based | XGBoost, LightGBM, Random Forest |
| Linear | SVM (SVR) |
| Sequence | LSTM, GRU, CNN-LSTM, TCN |
| Advanced | Helformer (Transformer), EMGNN (Graph NN) |
| Probabilistico | MDN (Mixture Density), BNN (Bayesian NN) |
| Incerteza | Conformal Prediction, MC Dropout |
| Meta | Ensemble (Stacking Ridge) |
| RL | PPO (stable-baselines3 + Gymnasium) |

## Pipeline de Features (18 estagios)

1. Preprocessamento (cleaning, returns, target)
2. Wavelet denoising (db4)
3. Indicadores tecnicos (MACD, RSI, Bollinger, ADX, Ichimoku, ATR)
4. Lag features (5 lags)
5. Market features (correlacao BTC, dominancia)
6. Sentiment (Fear & Greed Index)
7. CEEMDAN decomposition (6 IMFs)
8. HMM regime detection (3 regimes: bull/bear/sideways)
9. On-chain features
10. GARCH volatility (3 modelos)
11. HAR realized variance (daily/weekly/monthly)
12. GSADF bubble detection
13. Z-score anomalies
14. Macro variables (VIX, DXY, SPX, gold, oil)
15. Whale monitoring (volume z-score)
16. NLP sentiment (Sentence-BERT)
17. Cross-crypto features (correlacoes cruzadas)
18. Feature selection (Boruta/L1/Mutual Info)

## Features Avancadas (Fase 3 - aiagentstore.ai)

- **Jina AI Embeddings**: Market context memory com janela de 8192 tokens
- **Advanced Crawler**: Async com retry, anti-bloqueio, 6 fontes de noticias
- **Smart Money Tracker**: Fluxo institucional, acumulacao/distribuicao, concentracao de baleias
- **Narrative Detection**: 8 narrativas cripto com momentum e sinal contrario
- **Multi-Agent Architecture**: DataAgent -> AnalystAgent -> TraderAgent -> RiskAgent

## Instalacao

### Requisitos

- Python 3.11+
- pip ou uv

### Setup Local

```bash
# Clonar repositorio
git clone https://github.com/FELIPEACASTRO/Cripto_DT.git
cd Cripto_DT

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Instalar extras (API + dev)
pip install -e ".[dev,api]"
```

### Docker

```bash
# Build e run com docker-compose
docker-compose up -d

# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## Uso

### Pipeline Completo

```bash
# Executar pipeline completo: coleta -> features -> treino -> predicao
python scripts/run_pipeline.py
```

### Etapas Individuais

```bash
# 1. Coletar dados
python scripts/collect_data.py

# 2. Treinar modelos
python scripts/train_models.py

# 3. Gerar predicoes
python scripts/predict.py

# 4. Pipeline real-time
python scripts/run_realtime.py
```

### API REST

```bash
# Iniciar servidor FastAPI
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Endpoints principais:
# GET  /health              - Health check
# GET  /predictions/{coin}  - Predicao para uma moeda
# GET  /predictions         - Predicoes para todas as moedas
# POST /predict/{coin}      - Disparar nova predicao
# GET  /signals/{coin}      - Sinais de trading
# GET  /models/{coin}/metrics - Metricas do modelo
# GET  /market/sentiment    - Sentimento de mercado
# GET  /market/narratives   - Narrativas dominantes
# POST /agents/run          - Executar pipeline de agentes
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

## Estrutura do Projeto

```
Cripto_DT/
├── config/
│   └── settings.py              # 18 dataclasses de configuracao
├── src/
│   ├── agents/                  # Arquitetura multi-agente
│   │   ├── base_agent.py        # Classe abstrata base
│   │   ├── data_agent.py        # Coleta de dados
│   │   ├── analyst_agent.py     # Analise e features
│   │   ├── trader_agent.py      # Predicoes e sinais
│   │   ├── risk_agent.py        # Gestao de risco (6 camadas)
│   │   └── orchestrator.py      # Coordenador central
│   ├── api/                     # REST API (FastAPI)
│   │   └── server.py            # Endpoints de producao
│   ├── data/                    # Coleta e processamento
│   │   ├── collector.py         # OHLCV via CCXT
│   │   ├── advanced_crawler.py  # Crawler async de noticias
│   │   ├── news_scraper.py      # Scraper multi-fonte
│   │   ├── narrative_detector.py # Detector de narrativas
│   │   ├── smart_money.py       # Rastreador de smart money
│   │   ├── market_context.py    # Memoria de contexto (Jina AI)
│   │   ├── nlp_sentiment.py     # Sentiment NLP (BERT)
│   │   ├── whale_monitor.py     # Monitor de baleias
│   │   ├── onchain.py           # Dados on-chain
│   │   ├── sentiment.py         # Fear & Greed Index
│   │   ├── macro.py             # Dados macroeconomicos
│   │   └── preprocessor.py      # Limpeza de dados
│   ├── features/                # Engenharia de features
│   │   ├── pipeline.py          # Orquestrador de 18 estagios
│   │   ├── technical.py         # Indicadores tecnicos
│   │   ├── wavelet.py           # Wavelet denoising
│   │   ├── decomposition.py     # CEEMDAN/EMD
│   │   ├── regime.py            # HMM regime detection
│   │   ├── volatility.py        # GARCH volatility
│   │   ├── har_volatility.py    # HAR realized variance
│   │   ├── bubble.py            # GSADF bubble detection
│   │   ├── anomaly.py           # Anomaly detection
│   │   ├── cross_crypto.py      # Correlacoes cruzadas
│   │   ├── timeframe_fusion.py  # Multi-timeframe fusion
│   │   └── feature_selection.py # Boruta/L1/MI selection
│   ├── models/                  # 16 modelos ML
│   │   ├── xgboost_model.py     # XGBoost
│   │   ├── lightgbm_model.py    # LightGBM
│   │   ├── random_forest.py     # Random Forest
│   │   ├── svm_model.py         # SVM/SVR
│   │   ├── lstm_gru.py          # LSTM e GRU
│   │   ├── cnn_lstm.py          # CNN-LSTM hibrido
│   │   ├── tcn.py               # Temporal Convolutional
│   │   ├── helformer.py         # Transformer + Hellinger
│   │   ├── graph_model.py       # EMGNN (Graph NN)
│   │   ├── mdn.py               # Mixture Density Network
│   │   ├── bnn.py               # Bayesian Neural Network
│   │   ├── conformal.py         # Conformal Prediction
│   │   ├── ensemble.py          # Stacking Ridge
│   │   └── losses.py            # Loss functions customizadas
│   ├── training/                # Infraestrutura de treino
│   │   ├── trainer.py           # Treinador principal
│   │   ├── walk_forward.py      # Walk-forward validation
│   │   ├── hyperopt.py          # Optuna hyperparameter tuning
│   │   ├── automl.py            # AutoML pipeline
│   │   └── backtester.py        # Motor de backtesting
│   ├── trading/                 # Trading e RL
│   │   ├── signal_generator.py  # Kelly Criterion + ATR SL/TP
│   │   ├── rl_agent.py          # PPO reinforcement learning
│   │   └── environment.py       # Gymnasium environment
│   ├── evaluation/              # Avaliacao
│   │   ├── metrics.py           # RMSE, MAE, Sharpe, Sortino
│   │   ├── explainability.py    # SHAP explainability
│   │   └── visualizer.py        # Visualizacoes
│   ├── monitoring/              # Monitoramento
│   │   └── tracker.py           # Data drift + performance
│   └── prediction/
│       └── predictor.py         # Motor de inferencia
├── dashboard/
│   └── app.py                   # Streamlit dashboard (5 tabs)
├── scripts/                     # Scripts de execucao
│   ├── run_pipeline.py          # Pipeline completo
│   ├── collect_data.py          # Coleta de dados
│   ├── train_models.py          # Treino de modelos
│   ├── predict.py               # Gerar predicoes
│   └── run_realtime.py          # Pipeline real-time
├── tests/                       # 132+ testes
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── ROADMAP.md
```

## Validacao e Walk-Forward

O sistema usa **walk-forward validation** com anti-leakage:
- Expanding window (janela crescente)
- 5 folds
- Split: 70% treino / 15% validacao / 15% teste
- Features calculadas separadamente por fold (sem data leakage)

## Gestao de Risco (RiskAgent - 6 Camadas)

1. **Filtro de confianca**: rejeita sinais com confianca < threshold
2. **Limites de posicao**: max 25% por posicao individual
3. **Exposicao do portfolio**: max 80% do capital total
4. **Filtro de correlacao**: evita posicoes em ativos correlacionados (r > 0.7)
5. **Escala por volatilidade**: reduz posicoes em alta volatilidade
6. **Protecao de drawdown**: corta posicoes se drawdown > 15%

## Testes

```bash
# Rodar todos os testes
python -m pytest tests/ -q

# Rodar com cobertura
python -m pytest tests/ --cov=src --cov-report=html
```

## Tech Stack

| Categoria | Tecnologias |
|-----------|-------------|
| ML/DL | PyTorch, scikit-learn, XGBoost, LightGBM |
| NLP | sentence-transformers, transformers (Jina AI) |
| Time Series | GARCH (arch), HMM (hmmlearn), wavelets (PyWavelets) |
| RL | stable-baselines3, Gymnasium |
| Data | pandas, numpy, ccxt, yfinance |
| API | FastAPI, uvicorn |
| Dashboard | Streamlit, Plotly |
| Infra | Docker, GitHub Actions |
| Testes | pytest |

## Roadmap

Consulte [ROADMAP.md](ROADMAP.md) para o plano de evolucao completo.

## Licenca

MIT
