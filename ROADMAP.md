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
- [x] 71/71 testes passando

## Fase 3: Melhorias aiagentstore.ai — Alta Prioridade (Em Andamento)

### 3.1 Jina AI Embeddings (Inspirado: Jina AI)
- [ ] Substituir all-MiniLM-L6-v2 por jina-embeddings-v3 no MarketContextMemory
- [ ] Janela de 8192 tokens (vs 512 atual) para contexto financeiro
- [ ] Multilingual support para noticias em PT-BR e EN
- **Impacto**: Melhoria direta na qualidade de busca semantica de eventos

### 3.2 Advanced Web Crawler (Inspirado: Crawl4AI)
- [ ] Crawler async com retry e anti-bloqueio
- [ ] Suporte a conteudo dinamico (JS rendering)
- [ ] Output otimizado para pipeline LLM
- [ ] Mais fontes: CoinDesk, CoinTelegraph, Decrypt, The Block
- **Impacto**: 5-10x mais dados de noticias com melhor qualidade

### 3.3 Smart Money Tracker (Inspirado: SocialScan, aixbt)
- [ ] Rastreamento de carteiras de baleias via APIs on-chain
- [ ] Deteccao de acumulacao/distribuicao por smart money
- [ ] Score de smart money flow como feature preditiva
- [ ] Alertas de movimentacao anomala
- **Impacto**: Indicador antecedente de alta qualidade

### 3.4 Narrative Detection NLP (Inspirado: aixbt by Virtuals)
- [ ] Detector de narrativas emergentes (DeFi, AI tokens, RWA, memecoins)
- [ ] Tracking de momentum de narrativas ao longo do tempo
- [ ] Score de forca de narrativa como feature preditiva
- [ ] Correlacao narrativa-preco para gerar alpha
- **Impacto**: Capturar movimentos de mercado antes que virem mainstream

### 3.5 Multi-Agent Architecture (Inspirado: SigTech MAGIC, FinRobot)
- [ ] DataAgent: orquestra coleta de dados (OHLCV, noticias, on-chain)
- [ ] AnalystAgent: executa analise tecnica e fundamentalista
- [ ] TraderAgent: gera sinais e gerencia ordens
- [ ] RiskAgent: monitora exposicao e aplica filtros de risco
- [ ] Orquestrador central coordenando todos os agentes
- **Impacto**: Modularidade, testabilidade e escalabilidade

## Fase 4: Melhorias aiagentstore.ai — Media Prioridade (Futuro)

### 4.1 FinRobot Chain-of-Thought (Inspirado: FinRobot)
- [ ] Padrão Data-CoT -> Concept-CoT -> Thesis-CoT para relatorios
- [ ] Geracao automatica de teses de investimento
- [ ] Relatorios explicaveis para cada previsao
- **GitHub**: https://github.com/AI4Finance-Foundation/FinRobot

### 4.2 AutoML Pipeline (Inspirado: AutoML-Agent)
- [ ] Selecao automatica de modelos baseada em performance recente
- [ ] Hyperparameter tuning autonomo por walk-forward window
- [ ] Multi-modality support (time-series + NLP + graph)
- **GitHub**: https://github.com/DeepAuto-AI/automl-agent

### 4.3 TradingView Integration (Inspirado: 3Commas, Coinrule)
- [ ] Receber sinais de TradingView via webhook
- [ ] Enviar alertas de sinais para TradingView
- [ ] Templates de estrategia pre-configurados
- **Impacto**: Ponte entre nossas previsoes e execucao real

## Fase 5: Melhorias Futuras (Backlog)

### 5.1 Causal Reasoning (Inspirado: causaLens AI)
- [ ] Substituir correlacao por causalidade nas features
- [ ] Causal discovery automatica entre variaveis
- [ ] Simulacao de cenarios contrafactuais

### 5.2 Persistent Memory (Inspirado: memU)
- [ ] Memoria hierarquica (file-based + RAG)
- [ ] Predicao de intencao do mercado
- [ ] Reducao de custo de tokens via caching

### 5.3 Vector Database (Inspirado: Pinecone)
- [ ] Migrar market context memory para Pinecone/ChromaDB
- [ ] Busca semantica em escala com metadata filtering
- [ ] Real-time indexing de novos eventos

### 5.4 StockAgent Simulation (Inspirado: StockAgent)
- [ ] Simulacao multi-agente para backtesting comportamental
- [ ] Testar como diferentes perfis reagiriam as previsoes
- **GitHub**: https://github.com/MingyuJ666/Stockagent

### 5.5 Mettalex DEX Integration (Inspirado: Mettalex)
- [ ] Trading descentralizado com AI agents
- [ ] Cross-chain interoperability
- [ ] Natural language trading commands

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
