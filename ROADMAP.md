# Cripto_DT — Roadmap de Evolucao

## Fase 1: Sistema Base (Concluido)
- [x] 16 modelos ML (XGBoost, LSTM, GRU, LightGBM, RF, SVM, CNN-LSTM, TCN, Helformer, EMGNN, MDN, BNN, Ensemble, Conformal)
- [x] Pipeline de 18 features (tecnicos, wavelets, CEEMDAN, HMM, GARCH/HAR, GSADF, anomalias, macro, whale, NLP, correlacoes)
- [x] Walk-forward validation com anti-leakage (expanding window, 5 folds, 70/15/15)
- [x] MC Dropout para estimativa de incerteza
- [x] Conformal Prediction para intervalos de confianca
- [x] PPO reinforcement learning (stable-baselines3 + Gymnasium)
- [x] SHAP explainability (TreeExplainer + KernelExplainer fallback)
- [x] Dashboard Streamlit

## Fase 2: Pipeline Real-Time e Sinais (Concluido)
- [x] News scraper multi-fonte (CryptoCompare, Reddit, RSS)
- [x] Market context memory com sentence-transformers embeddings
- [x] Signal generator (Kelly Criterion, ATR-based SL/TP)
- [x] Real-time pipeline orchestrator com health check
- [x] 71/71 testes passando (expandido para 132/132 na Fase 3)

## Fase 3: Melhorias aiagentstore.ai — Alta Prioridade (Concluido)

### 3.1 Jina AI Embeddings (Inspirado: Jina AI) ✓
- [x] Backend auto-detect com prioridade: jina -> sentence-transformers -> tfidf
- [x] Suporte a jina-embeddings-v3 com janela de 8192 tokens (vs 512 anterior)
- [x] Multilingual support para noticias em PT-BR e EN
- [x] Feature `ctx_narrative_similarity` no MarketContextMemory
- [x] Config: `embedding_backend`, `embedding_model`, `max_context_events`
- **Arquivo**: `src/data/market_context.py` (atualizado)

### 3.2 Advanced Web Crawler (Inspirado: Crawl4AI) ✓
- [x] Crawler async via aiohttp com fallback sincrono (requests)
- [x] Retry com backoff exponencial e rate limiting por dominio
- [x] Rotacao de 6 user agents e anti-bloqueio
- [x] Deduplicacao por MD5 de titulo normalizado
- [x] 6 fontes: CryptoCompare, CoinGecko, Reddit (4 subs), RSS (CoinDesk, CoinTelegraph, Decrypt, TheBlock)
- [x] Fear & Greed Index integration
- [x] 5 features: `crawler_news_count`, `crawler_sentiment_mean`, `crawler_fear_greed`, `crawler_buzz_score`, `crawler_source_diversity`
- **Arquivo**: `src/data/advanced_crawler.py` (1096 linhas)

### 3.3 Smart Money Tracker (Inspirado: SocialScan, aixbt) ✓
- [x] Smart Money Flow (-1 a 1): MFI ponderado por volume institucional
- [x] Acumulacao (0-1): AD Line z-score + divergencia volume-preco
- [x] Distribuicao (0-1): OBV-preco divergencia + volume declinante
- [x] Exchange Flow Ratio (0.1-10): posicao do close dentro do range H-L
- [x] Whale Concentration (0-1): ratio de volume anomalo (>2σ)
- [x] Enriquecimento opcional via APIs (Blockchain.com, Etherscan)
- **Arquivo**: `src/data/smart_money.py` (518 linhas)

### 3.4 Narrative Detection NLP (Inspirado: aixbt by Virtuals) ✓
- [x] 8 categorias de narrativas: DeFi, AI Crypto, RWA, Memecoins, L2 Scaling, Regulation, Institutional, Gaming/NFT
- [x] Recency weighting com decaimento exponencial (0.85/dia)
- [x] Coin-to-narrative alignment para 15 moedas
- [x] 5 features: `narrative_dominant_score`, `narrative_diversity`, `narrative_momentum_7d`, `narrative_alignment`, `narrative_contrarian`
- **Arquivo**: `src/data/narrative_detector.py` (589 linhas)

### 3.5 Multi-Agent Architecture (Inspirado: SigTech MAGIC, FinRobot) ✓
- [x] BaseAgent: classe abstrata com AgentResult, timing, error handling
- [x] DataAgent: orquestra coleta (OHLCV + news + whale data)
- [x] AnalystAgent: preprocessing + feature pipeline + NLP + narrativas + smart money
- [x] TraderAgent: ML predictions + signal generation + regime filtering
- [x] RiskAgent: 6 camadas (confidence, position limits, exposure, correlation, volatility, drawdown)
- [x] AgentOrchestrator: pipeline Data->Analyst->Trader->Risk com contexto acumulativo
- [x] 132/132 testes passando
- **Diretorio**: `src/agents/` (7 arquivos, 1571 linhas)

## Fase 4: Infraestrutura de Producao (Concluido)

### 4.1 FastAPI REST API ✓
- [x] 12 endpoints: health, predictions, signals, models, features, sentiment, narratives, agents, portfolio
- [x] Pydantic v2 schemas para request/response
- [x] Rate limiting in-memory (60 req/min por IP)
- [x] CORS middleware, background tasks, cache de predicoes (5min)
- [x] OpenAPI docs automaticos com tags por grupo
- **Arquivo**: `src/api/server.py`

### 4.2 Docker + Containerizacao ✓
- [x] Dockerfile multi-stage (builder + runtime) com Python 3.13-slim
- [x] docker-compose com 3 servicos: api (8000), dashboard (8501), worker
- [x] Volumes compartilhados para data/, models/, logs/
- [x] Health checks, resource limits, restart policies
- [x] .env.example com todas as variaveis de ambiente
- **Arquivos**: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`

### 4.3 CI/CD GitHub Actions ✓
- [x] Pipeline CI: lint (ruff), test (pytest), security (bandit)
- [x] Pipeline de treino agendado (domingos 00:00 UTC + manual dispatch)
- [x] Cache de dependencias pip
- **Diretorio**: `.github/workflows/`

### 4.4 pyproject.toml ✓
- [x] Configuracao moderna com grupos opcionais: [dev], [api], [all]
- [x] Ruff config (line-length=120, py311)
- [x] Pytest config (testpaths=tests)
- [x] Entry point: `cripto-dt` CLI
- **Arquivo**: `pyproject.toml`

### 4.5 Backtesting Engine ✓
- [x] Backtester com SL/TP, slippage, comissao, execucao T+1
- [x] BacktestResult com 15+ metricas (Sharpe, Sortino, Calmar, Omega, drawdown, win rate)
- [x] BenchmarkComparator: alpha, beta, information ratio vs buy-and-hold
- [x] Relatorio formatado em portugues
- **Arquivo**: `src/trading/backtester.py`

### 4.6 AutoML Pipeline (Inspirado: AutoML-Agent) ✓
- [x] Selecao automatica dos top-K modelos por performance recente
- [x] Tuning com Optuna (TPE sampler + MedianPruner) com search spaces por modelo
- [x] Pipeline completo: select top-5 -> tune top-3 -> weighted ensemble
- **Arquivo**: `src/training/automl.py`

### 4.7 Monitoring + Data Drift ✓
- [x] PerformanceTracker com log de predicoes em JSON
- [x] Deteccao de data drift via PSI (Population Stability Index)
- [x] Deteccao de model drift (degradacao de acuracia direcional)
- [x] Sistema de alertas com severidade (normal/moderado/severo)
- [x] Relatorios de performance com sumario 7d/30d
- **Arquivo**: `src/monitoring/tracker.py`

### 4.8 README + Documentacao ✓
- [x] README completo com arquitetura, instalacao, uso, estrutura
- [x] Diagrama ASCII da arquitetura multi-agente
- [x] Tabelas de modelos, features, tech stack
- [x] 159/159 testes passando

## Fase 5: Melhorias Futuras (Backlog)

### 5.1 FinRobot Chain-of-Thought (Inspirado: FinRobot)
- [ ] Padrao Data-CoT -> Concept-CoT -> Thesis-CoT para relatorios
- [ ] Geracao automatica de teses de investimento
- **GitHub**: https://github.com/AI4Finance-Foundation/FinRobot

### 5.2 TradingView Integration (Inspirado: 3Commas)
- [ ] Receber sinais de TradingView via webhook
- [ ] Enviar alertas de sinais para TradingView

### 5.3 Causal Reasoning (Inspirado: causaLens AI)
- [ ] Substituir correlacao por causalidade nas features
- [ ] Causal discovery automatica entre variaveis

### 5.4 Vector Database (Inspirado: Pinecone)
- [ ] Migrar market context memory para Pinecone/ChromaDB
- [ ] Busca semantica em escala com metadata filtering

### 5.5 StockAgent Simulation (Inspirado: StockAgent)
- [ ] Simulacao multi-agente para backtesting comportamental
- **GitHub**: https://github.com/MingyuJ666/Stockagent

---

## Referencias (aiagentstore.ai)

| Ferramenta | Tipo | Status | Relevancia |
|---|---|---|---|
| Jina AI | Open-source | Embeddings superiores | Alta |
| Crawl4AI | Open-source | Web crawler AI-ready | Alta |
| SocialScan | Freemium | On-chain intelligence | Alta |
| aixbt by Virtuals | Closed | Narrative detection (99% autonomia) | Alta |
| SigTech MAGIC | Closed | Multi-agent finance (89% autonomia) | Alta |
| FinRobot | Open-source | Financial AI agents | Media |
| AutoML-Agent | Open-source | ML pipeline automation | Media |
| 3Commas | Freemium | Trading bots + TradingView | Media |
| causaLens AI | Closed | Causal reasoning | Baixa |
| memU | Open-source | Agentic memory | Baixa |
| Pinecone | Freemium | Vector database | Baixa |
| StockAgent | Open-source | Multi-agent simulation | Baixa |
| APIx420 | Paid | 100+ crypto endpoints | Media |
| AgentHC API | Paid | Trading intelligence API | Media |
| Intellectia.AI | Freemium | AI investment research | Media |
| Mettalex | Open-source | AI-powered DEX | Baixa |
