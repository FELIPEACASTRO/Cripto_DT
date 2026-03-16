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

### 3.1 Jina AI Embeddings (Inspirado: Jina AI)
- [x] Backend auto-detect com prioridade: jina -> sentence-transformers -> tfidf
- [x] Suporte a jina-embeddings-v3 com janela de 8192 tokens (vs 512 anterior)
- [x] Multilingual support para noticias em PT-BR e EN
- [x] Feature `ctx_narrative_similarity` no MarketContextMemory
- **Arquivo**: `src/data/market_context.py`

### 3.2 Advanced Web Crawler (Inspirado: Crawl4AI)
- [x] Crawler async via aiohttp com fallback sincrono (requests)
- [x] Retry com backoff exponencial e rate limiting por dominio
- [x] Rotacao de 6 user agents e anti-bloqueio
- [x] Deduplicacao por MD5 de titulo normalizado
- [x] 6 fontes: CryptoCompare, CoinGecko, Reddit (4 subs), RSS (CoinDesk, CoinTelegraph, Decrypt, TheBlock)
- [x] Fear & Greed Index integration
- **Arquivo**: `src/data/advanced_crawler.py`

### 3.3 Smart Money Tracker (Inspirado: SocialScan, aixbt)
- [x] Smart Money Flow, Acumulacao, Distribuicao, Exchange Flow Ratio, Whale Concentration
- [x] Enriquecimento opcional via APIs (Blockchain.com, Etherscan)
- **Arquivo**: `src/data/smart_money.py`

### 3.4 Narrative Detection NLP (Inspirado: aixbt by Virtuals)
- [x] 8 categorias de narrativas com recency weighting
- [x] Coin-to-narrative alignment para 15 moedas
- **Arquivo**: `src/data/narrative_detector.py`

### 3.5 Multi-Agent Architecture (Inspirado: SigTech MAGIC, FinRobot)
- [x] BaseAgent, DataAgent, AnalystAgent, TraderAgent, RiskAgent, AgentOrchestrator
- **Diretorio**: `src/agents/`

## Fase 4: Infraestrutura de Producao (Concluido)
- [x] FastAPI REST API (12 endpoints)
- [x] Docker + docker-compose (3 servicos)
- [x] CI/CD GitHub Actions
- [x] pyproject.toml
- [x] Backtesting Engine (Sharpe, Sortino, Calmar, Omega)
- [x] AutoML Pipeline (Optuna)
- [x] Monitoring + Data Drift (PSI)
- [x] README + Documentacao
- [x] 159/159 testes passando

---

## Fase 5: Time Series Foundation Models (PRIORIDADE CRITICA)

> **Impacto esperado**: +10-20% acuracia direcional via zero-shot e fine-tuning
> **Fonte**: Pesquisa HuggingFace, Google, Amazon, Salesforce, CMU

### 5.1 Amazon Chronos-Bolt (Zero-Shot Forecasting)
- [ ] Integrar `amazon/chronos-bolt-base` (7.1M downloads, Apache-2.0)
- [ ] Wrapper ChronosModel que implementa interface BaseModel (fit/predict/predict_with_confidence)
- [ ] Zero-shot forecasting como baseline sem treino
- [ ] Fine-tuning em dados OHLCV crypto para cada moeda
- [ ] Benchmark: `mainmagic/chronos-t5-small-btc-m1` (BTC fine-tune existente)
- [ ] Adicionar ao ensemble como modelo #17
- **Ref**: T5-based, tokeniza series temporais em bins, previsoes probabilisticas nativas

### 5.2 Google TimesFM 2.0 (Foundation Model 500M)
- [ ] Integrar `google/timesfm-2.0-500m-pytorch` (Apache-2.0)
- [ ] Decoder-only com contexto variavel e horizonte flexivel
- [ ] Fine-tune em dados crypto multi-moeda
- [ ] Adicionar ao ensemble como modelo #18
- **Ref**: Pre-treinado em corpus massivo de series temporais reais + sinteticas

### 5.3 Salesforce MOIRAI (Multivariate + MoE)
- [ ] Integrar `Salesforce/moirai-1.1-R-large` (CC-BY-NC-4.0)
- [ ] Alimentar preco + volume + indicadores tecnicos simultaneamente (multivariate nativo)
- [ ] Testar MOIRAI-Agent para selecao automatica de especialista
- [ ] Testar `moirai-moe-1.0-R-base` (MoE para multi-dominio)
- **Ref**: Qualquer frequencia, variaveis e horizonte arbitrario

### 5.4 CMU MOMENT (Multi-Task: Forecast + Anomaly + Imputation)
- [ ] Integrar `AutonLab/MOMENT-1-large` (MIT license)
- [ ] Usar para 3 tarefas simultaneas:
  - Forecasting como membro do ensemble (#19)
  - Anomaly detection para regime de mercado (substituir/complementar z-score)
  - Imputation para dados faltantes no pipeline de features
- **Ref**: Pre-treinado no Timeseries-PILE, unico modelo multi-tarefa

### 5.5 Lag-Llama (Probabilistic Forecasting)
- [ ] Integrar `time-series-foundation-models/Lag-Llama` (Apache-2.0)
- [ ] Usar como prior probabilistico para BNN e MDN
- [ ] Alimentar output distribucional no Conformal Prediction
- [ ] Modelo leve (~1M params) — ideal para ensemble
- **Ref**: LLaMA-based, Student-t distribution head, lags como covariates

### 5.6 Time-MoE (ICLR 2025 Spotlight — MoE para Time Series)
- [ ] Integrar Time-MoE (2.4B params, pre-treinado no Time-300B)
- [ ] Sparse MoE: ativa subset de especialistas por previsao
- [ ] Benchmark contra ensemble atual
- **GitHub**: https://github.com/Time-MoE/Time-MoE
- **Paper**: arXiv:2409.16040

### 5.7 FinCast (Foundation Model para Financial Time Series)
- [ ] Integrar FinCast (1B MoE, 4 experts, top-k=2 routing)
- [ ] Primeiro foundation model PURPOSE-BUILT para series financeiras
- [ ] Point-Quantile loss: -23% MSE, -16% MAE em zero-shot
- [ ] 5x mais rapido que modelos densos equivalentes em GPU consumer
- [ ] Suporta crypto_1min, stock_1day, futures_1wk nativamente
- **Paper**: arXiv:2508.19609

### 5.8 Sundial (ICML 2025 ORAL — Top 1%, 1 TRILHAO de pontos)
- [ ] Integrar `thuml/sundial-base-128m` (Apache-2.0)
- [ ] TimeFlow Loss: flow-matching para valores continuos (sem tokenizacao discreta)
- [ ] Pre-treinado em TimeBench (1 TRILHAO de time points)
- [ ] SOTA em TSLib, GIFT-Eval e FEV benchmarks simultaneamente
- [ ] Zero-shot em milissegundos — ideal para real-time pipeline
- **GitHub**: https://github.com/thuml/Sundial
- **Paper**: arXiv:2502.00816

### 5.9 Chronos-2 (Amazon, Jan 2026 — Multivariate + Covariates)
- [ ] Atualizar de Chronos-Bolt para `amazon/chronos-2`
- [ ] Multivariate + covariate-informed: alimentar preco + on-chain + sentimento juntos
- [ ] 90%+ win rate sobre Chronos-Bolt
- [ ] 250x mais rapido e 20x menos memoria que Chronos original
- [ ] #1 entre modelos pre-treinados no GIFT-Eval

### 5.10 CryptoMamba (SSM Purpose-Built para Bitcoin)
- [ ] Integrar CryptoMamba: Mamba SSM com C-Blocks para crypto
- [ ] Complexidade NEAR-LINEAR (vs quadratica dos transformers)
- [ ] Captura regime shifts e long-range dependencies melhor que LSTM/GRU
- [ ] Publicado no IEEE ICBC 2025 e ICLR 2025 Workshop
- [ ] Substituir/complementar LSTM e GRU existentes
- **GitHub**: https://github.com/MShahabSepehri/CryptoMamba
- **Paper**: arXiv:2501.01010

### 5.11 IBM Tiny Time Mixers (TTM — Roda em CPU, 1M params)
- [ ] Integrar `ibm-granite/granite-timeseries-ttm-r1` (Apache-2.0)
- [ ] MLP-Mixer (nao Transformer): 1M-9.1M params
- [ ] #2 no GIFT-Eval — menor modelo no top 10, supera rivais 20x maiores
- [ ] 4-40% melhor que ML tradicional em zero/few-shot
- [ ] Roda em CPU puro — ideal como fallback de producao
- **Ref**: IBM Granite ecosystem

---

## Fase 6: Financial NLP & Sentiment Revolution (ALTA PRIORIDADE)

> **Impacto esperado**: +5-15% nos sinais de trading via sentimento preciso
> **Fonte**: HuggingFace, FinGPT, CardiffNLP

### 6.1 FinBERT — Padrao Ouro de Sentimento Financeiro
- [ ] Integrar `ProsusAI/finbert` (5.9M downloads) no pipeline NLP
- [ ] Substituir/complementar NLP atual com sentimento financeiro especializado
- [ ] 3 classes: positive/negative/neutral com dominio financeiro
- [ ] Processar noticias, analyst reports, social media sobre crypto
- **Arquivo alvo**: `src/data/nlp_sentiment.py`

### 6.2 Twitter-RoBERTa (Crypto Twitter/X)
- [ ] Integrar `cardiffnlp/twitter-roberta-base-sentiment-latest` (3.1M downloads, CC-BY-4.0)
- [ ] Treinado em 124M tweets — entende linguagem do Twitter (hashtags, emojis, slang)
- [ ] Usar para sentimento de crypto Twitter (principal canal de sinais)
- [ ] Integrar variante multilingual (`twitter-xlm-roberta-base-sentiment`) para KR/JP/CN

### 6.3 FinBERT Forward-Looking Statements
- [ ] Integrar `yiyanghkust/finbert-fls` para detectar declaracoes previsionais
- [ ] Identificar "forward-looking statements" em noticias (guidance, previsoes)
- [ ] Feature: `nlp_forward_looking_score` — sinal preditivo de movimento

### 6.4 FinBERT Tone (Chinese)
- [ ] Integrar `yiyanghkust/finbert-tone-chinese` (287K downloads, Apache-2.0)
- [ ] Analisar sentimento de comunidades crypto chinesas (WeChat, Weibo)
- [ ] China e mercado critico — sinais em chines sao alta prioridade

### 6.5 Ensemble de Sentimento (Multi-Model)
- [ ] Combinar ProsusAI/finbert + finbert-tone + Twitter-RoBERTa + FinancialBERT
- [ ] Voting/averaging para sentimento robusto
- [ ] Feature: `nlp_ensemble_sentiment` (media ponderada de 4 modelos)
- [ ] Sentimento rapido via `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` (2x mais rapido)

### 6.6 FinBERT-BiLSTM Hybrid (93.27% F1 em Crypto)
- [ ] Integrar FinBERT-BiLSTM: combina extracao de sentimento FinBERT + modelagem temporal BiLSTM
- [ ] F1-score de 93.27% especificamente em sentimento crypto
- [ ] Substituir pipeline de sentimento atual por abordagem hibrida
- [ ] Sentimento com lag temporal (nao apenas corrente) como feature
- **Paper**: arXiv:2411.12748

### 6.7 FinGPT Forecaster (LLM-Based Prediction)
- [ ] Integrar `FinGPT/fingpt-forecaster_dow30_llama2-7b_lora` (Apache-2.0)
- [ ] Alimentar news + dados de preco → previsao direcional com raciocinio
- [ ] Usar como meta-analisador no TraderAgent
- [ ] LoRA adapter sobre Llama-2 base
- **Ref**: https://github.com/AI4Finance-Foundation/FinGPT

### 6.7 Datasets de Treino para Fine-Tuning
- [ ] Baixar `takala/financial_phrasebank` (4840 frases anotadas por 16 especialistas)
- [ ] Baixar `zeroshot/twitter-financial-news-sentiment` (MIT, 10K-100K tweets)
- [ ] Baixar `zeroshot/twitter-financial-news-topic` (classificacao por topico)
- [ ] Fine-tune modelos de sentimento em dados crypto-especificos
- [ ] Criar dataset proprio: noticias crypto anotadas com impacto de preco

---

## Fase 7: MoE Gating Network + LLM Reasoning Agents (ALTA PRIORIDADE)

> **Impacto esperado**: +15-25% retorno via ensemble inteligente + raciocinio LLM
> **Fonte**: MIGA (arXiv), DeepSeek, Qwen, Alpha Arena

### 7.1 MoE Gating Network para Ensemble de 16 Modelos (Inspirado: MIGA)
- [ ] Implementar gating network (MLP pequeno) que roteia regimes de mercado para modelos especialistas
- [ ] Input do gating: features de mercado atuais (HMM regime, volatilidade, volume)
- [ ] Output: distribuicao de pesos sobre os 16 modelos
- [ ] Treinar gating para aprender quais modelos performam melhor em cada condicao:
  - LSTM/GRU para mercados trending
  - XGBoost/RF para mercados ranging
  - BNN/MDN para alta incerteza
  - Helformer/TCN para padroes complexos
- [ ] Top-K ativacao: ativar apenas 4-6 de 16 modelos por previsao (reduzir inferencia ~75%)
- [ ] Inner group attention (MIGA) para familias de modelos compartilharem informacao
- **Paper**: arXiv:2410.02241 (MIGA — 24% retorno excedente no CSI300)
- **Arquivo alvo**: `src/models/moe_ensemble.py` (novo)

### 7.2 DeepSeek-R1 como Reasoning Agent
- [ ] Integrar DeepSeek-R1 via API como meta-raciocinio do TraderAgent
- [ ] Chain-of-thought: analisar sinais conflitantes dos 16 modelos
- [ ] Input: outputs dos modelos + features + sentimento + narrativas
- [ ] Output: tese de investimento estruturada + direcao + confianca
- [ ] 126% retorno em competicao de crypto trading (Alpha Arena)
- **Ref**: MIT License, 671B MoE (37B ativos), arXiv:2501.12948

### 7.3 Qwen3 Max como Trading Strategy Agent
- [ ] Integrar Qwen3 Max via API (Alibaba Cloud)
- [ ] 22.32% retorno em 2 semanas no Alpha Arena (campea)
- [ ] Usar abordagem documentada: indicadores tecnicos + stop-loss/take-profit rigoroso
- [ ] Combinar com nosso signal_generator.py existente
- **Ref**: Tongyi Qianwen License (permissivo para pesquisa)

### 7.4 MiniCPM4.1 para Inferencia Local Eficiente
- [ ] Integrar `openbmb/MiniCPM4.1` (~4B params, Apache-2.0)
- [ ] Rodar em GPU consumer (RTX 3060/4060 suficiente)
- [ ] Usar como AnalystAgent local (sem depender de APIs externas)
- [ ] MiniCPM-SALA: 1M tokens de contexto para processar historicos longos
- **GitHub**: https://github.com/OpenBMB/MiniCPM

### 7.5 Kimi K2.5 API para Analise de Longo Contexto
- [ ] Integrar Kimi K2.5 via API ($0.15/M tokens — mais barato disponivel)
- [ ] 256K tokens de contexto: processar semanas de noticias em um prompt
- [ ] Usar para sumarizar e extrair sinais de grandes volumes de texto
- [ ] Agent swarm paradigm compativel com nossa arquitetura multi-agente
- **Ref**: MIT License (K2 open-source), 1T params MoE (32B ativos)

### 7.6 FinCoT — Structured Financial Prompting (+17.3% Acuracia)
- [ ] Implementar FinCoT: prompting estruturado com blueprints de raciocinio financeiro
- [ ] Melhorou Qwen3-8B de 63.2% para 80.5% em tarefas financeiras
- [ ] Reduz output em ate 8.9x vs CoT nao-estruturado
- [ ] Templates especificos: analise tecnica, sentimento, on-chain, macro
- **Paper**: arXiv:2506.16123 (FinNLP Workshop 2025)

### 7.7 TradingAgents Framework (UCLA + MIT, 7 Agent Roles)
- [ ] Avaliar `TauricResearch/TradingAgents` como orquestrador alternativo
- [ ] 7 papeis: Fundamentals, Sentiment, News, Technical, Researcher, Trader, Risk Manager
- [ ] Multi-provider LLM (GPT, Gemini, Claude, DeepSeek)
- [ ] Cada agente pode usar nossos modelos especializados como backend
- **GitHub**: https://github.com/TauricResearch/TradingAgents
- **Paper**: arXiv:2412.20138

### 7.8 MiniMax-M2.5 Financial Reasoning Agent (FinSearchComp 65.5)
- [ ] Integrar `MiniMaxAI/MiniMax-M2.5` (230B total / 10B ativo, Apache-2.0/Custom)
- [ ] Maior score FinSearchComp (65.5) entre todos os modelos pesquisados
- [ ] 1M tokens de contexto (M1) — processar meses de historico de mercado
- [ ] Variantes quantizadas disponiveis: GGUF, FP8, AWQ para deploy local
- [ ] Usar como core reasoning/prediction agent no multi-agent system
- **HuggingFace**: `MiniMaxAI/MiniMax-M2.5` (533K downloads)

### 7.9 Seed-OSS-36B / Seed1.8 para Long-Context Analysis
- [ ] Integrar `ByteDance-Seed/Seed-OSS-36B-Instruct` (512K context, Apache-2.0)
- [ ] Finance XpertBench: 62.0 — validado para financial reporting e market intelligence
- [ ] Ou usar Seed1.8 API (proprietario) para agentic decomposition
- [ ] UI-TARS para automacao de plataformas de trading
- **HuggingFace**: `ByteDance-Seed/Seed-OSS-36B-Instruct` (35K downloads)

### 7.10 FinRobot Chain-of-Thought (Data-CoT -> Concept-CoT -> Thesis-CoT)
- [ ] Implementar padrao: Data-CoT (facturar dados) -> Concept-CoT (identificar padroes) -> Thesis-CoT (gerar tese)
- [ ] Geracao automatica de teses de investimento por moeda
- [ ] Integrar com CCI4.0 CoT dataset (BAAI) para treino de raciocinio
- **GitHub**: https://github.com/AI4Finance-Foundation/FinRobot

---

## Fase 8: Vision & Chart Analysis (MEDIA PRIORIDADE)

> **Impacto esperado**: +5-10% via reconhecimento visual de padroes graficos
> **Fonte**: SKT A.X, MiniCPM-V, DEJIMA

### 8.1 SKT A.X 4.0-VL-Light (Chart Pattern Recognition)
- [ ] Integrar `skt/A.X-4.0-VL-Light` (8B params, Apache-2.0)
- [ ] 89.8% em analise de graficos e tabelas financeiras (KoBizDoc benchmark)
- [ ] Alimentar screenshots de graficos de candlestick para pattern recognition
- [ ] Detectar padroes visuais: head & shoulders, double top/bottom, flags, wedges
- [ ] Feature: `chart_pattern_signal` (-1 a 1)
- **Ref**: Melhor que Qwen2.5-VL-32B em graficos apesar de ser 4x menor

### 8.2 SenseNova-MARS/NEO para Chart Analysis (#1 Visual Reasoning China)
- [ ] Integrar `sensenova/SenseNova-MARS-8B` (MMSearch 74.3, supera GPT-5)
- [ ] #1 em visual reasoning na China (SuperCLUE 75.35)
- [ ] NEO architecture: requer apenas 1/10 dos dados tipicos para fine-tuning
- [ ] Analise de candlestick, support/resistance, volume profiles
- [ ] `sensenova/piccolo-large-zh` (MIT) para embeddings de documentos financeiros
- **HuggingFace**: `sensenova/SenseNova-MARS-8B`, `sensenova/NEO-9B`

### 8.3 MiniCPM-V para Chart Analysis Local
- [ ] Integrar `openbmb/MiniCPM-V-4.0` (4B params)
- [ ] Analise de candlestick em GPU consumer
- [ ] Supera GPT-4.1-mini em compreensao de imagens
- [ ] Processar graficos de TradingView automaticamente

### 8.3 MORFI — Multimodal Zero-Shot (Training-FREE, -88.9% Erro)
- [ ] Implementar abordagem MORFI: combinar representacao textual de precos + imagens de graficos via VLM
- [ ] TRAINING-FREE: sem treino adicional, apenas inferencia
- [ ] Reduz erro em 88.9% vs text-only (LLaVA MSE 0.0046 vs LLaMA MSE 0.0413)
- [ ] Gerar imagens de candlestick automaticamente a partir dos dados OHLCV
- [ ] Alimentar VLM (MiniCPM-V ou SKT VL-Light) com imagem + contexto textual
- **Paper**: ICCV 2025 Workshop (MORFI)

### 8.4 Pipeline de Chart Captioning (Inspirado: DEJIMA)
- [ ] Adaptar metodologia DEJIMA para crypto:
  1. Capturar screenshots de graficos automaticamente (Selenium/Playwright)
  2. Object detection para identificar padroes de candlestick
  3. LLM refinement para gerar descricoes naturais do estado do grafico
- [ ] Treinar VLM especifico para interpretacao de graficos crypto
- **Paper**: arXiv:2512.00773 (DEJIMA — 3.88M pares imagem-texto)

---

## Fase 9: Synthetic Data & Data Augmentation (MEDIA PRIORIDADE)

> **Impacto esperado**: +5-10% via aumento de dados para eventos raros
> **Fonte**: NVIDIA NeMo, DanQing, CCI4.0

### 9.1 NVIDIA NeMo Data Designer (Synthetic Market Scenarios)
- [ ] Usar NeMo Data Designer para gerar cenarios de mercado sinteticos
- [ ] PGM (Probabilistic Graphical Model): codificar correlacoes entre:
  - Faixa de preco BTC, regime de volatilidade, volume, funding rates
  - Sentimento de mercado, narrativa dominante, Fear & Greed Index
- [ ] Gerar cenarios "black swan" com fundamentacao estatistica real
- [ ] Data augmentation para eventos raros (crashes, flash crashes, pump & dumps)
- [ ] Gerar personas sinteticas de traders para treino de RL agent
- **GitHub**: https://github.com/NVIDIA-NeMo/DataDesigner (Apache-2.0)

### 9.2 MarketGANs — Synthetic Crash Data (TCN + GAN)
- [ ] Implementar MarketGANs: GAN com backbone TCN para gerar series financeiras sinteticas
- [ ] Preservar: caudas pesadas, volatility clustering, leverage effects, correlacoes cross-asset
- [ ] Gerar cenarios: flash crashes, pump & dumps, black swans, halving events
- [ ] MIT + JPMorgan: dados sinteticos identificam 18% mais fatores de risco
- [ ] Aumentar dados de treino para eventos raros no anomaly.py e bubble.py
- **Paper**: arXiv:2601.17773

### 9.3 Pipeline de Filtragem Multi-Estagio (Inspirado: DanQing)
- [ ] Adaptar metodologia DanQing para pipeline de noticias crypto:
  1. **Filtragem grossa**: classificador de spam/scam (modelo 1M params)
  2. **Blacklist de fontes**: filtrar fontes nao-confiaveis
  3. **Filtragem fina**: scoring de qualidade por relevancia
  4. **Cross-batch dedup**: deduplicacao temporal (mesma historia em periodos diferentes)
- [ ] Melhorar qualidade do input no news_scraper.py e advanced_crawler.py
- **Paper**: arXiv:2601.10305 (DanQing — 100M pares filtrados do Common Crawl)

### 9.3 CCI4.0 CoT Dataset para Fine-Tuning de Raciocinio
- [ ] Baixar subset CoT do `BAAI/CCI4.0-M2-CoT-v1` (gated access)
- [ ] 4.5 bilhoes de templates de raciocinio humano
- [ ] Fine-tune modelo de raciocinio para analise crypto multi-step
- [ ] Exemplo: "Dado RSI=72, MACD cruzando, whale acumulando → outlook probabilistico"

---

## Fase 10: Multilingual & Regional Intelligence (MEDIA-BAIXA PRIORIDADE)

> **Impacto esperado**: +3-8% via cobertura de mercados asiaticos
> **Fonte**: Swallow, SKT, Sarvam, Rakuten

### 10.1 Japanese Market Intelligence (Swallow LLM)
- [ ] Integrar `tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2` (Apache-2.0)
- [ ] Processar noticias de CoinPost (maior site crypto japones), Nikkei
- [ ] Japao: mercado crypto top-5 global, tax rate caindo para 20% em abr/2026
- [ ] Seguir abordagem NRI: fine-tune Swallow 8B em corpus crypto japones
- **Ref**: NRI financial fine-tune bateu GPT-4o em tarefas financeiras

### 10.2 Korean Market Intelligence (SKT A.X)
- [ ] Monitorar comunidades crypto coreanas (Upbit, Bithumb, DC Inside)
- [ ] Coreia: mercado crypto top-3 global, "Kimchi premium" como feature
- [ ] Usar `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` para sentimento KR
- [ ] Adaptar padroes de deteccao de manipulacao do FSC (Korea Financial Services Commission)
- **Ref**: FSC investindo 170M won em AI para detectar manipulacao crypto

### 10.3 Chinese Market Sentiment & LLM Agents
- [ ] Integrar `yiyanghkust/finbert-tone-chinese` (Apache-2.0)
- [ ] `baidu/ERNIE-4.5-21B-A3B` (Apache-2.0, $0.55/M tokens) como Chinese NLP agent
- [ ] `tencent/Hunyuan-A13B` (80B/13B MoE, dual-mode reasoning) como secondary reasoning agent
- [ ] `deepseek-ai/DeepSeek-OCR-2` (Apache-2.0, 1.3M downloads) para OCR de exchange screenshots e order books
- [ ] `Go4miii/DISC-FinLLM` (Fudan, Baichuan-13B + LoRA, 250K instruções financeiras chinesas) para NLP financeiro
- [ ] `zai-org/GLM-4.7-Flash` (MoE, MIT, 1.8M downloads, A-grade SuperCLUE-Fin) como reasoning financeiro
- [ ] `PaddlePaddle/PaddleOCR-VL` (0.9B, Apache-2.0, roda em CPU) como fallback OCR
- [ ] Monitorar WeChat, Weibo para sentimento crypto chines
- [ ] China drive significativo do mercado global de crypto
- **Nota**: HunyuanOCR não confirmado no HF; DeepSeek-OCR-2 é alternativa superior

### 10.4 Indian Market Data (AIKosh)
- [ ] Explorar datasets financeiros do AIKosh (aikosh.indiaai.gov.in, 3,000+ datasets)
- [ ] Indices macroeconomicos indianos como features complementares
- [ ] India: **119M usuarios crypto** (maior mercado global), dados de WazirX/CoinDCX APIs
- [ ] `ai4bharat/IndicSentiment` para sentimento multilíngue indiano

### 10.5 Southeast Asia & Gulf Markets (Novos — V2.1)
- [ ] `vinai/phobert-base-v2` (MIT) para sentimento crypto vietnamita (Vietnã top-3 adoção global)
- [ ] `indobenchmark/indobert-base-p2` (MIT, 242K downloads) para mercado indonésio
- [ ] `tiiuae/falcon-mamba-7b` (Apache-2.0) SSM para time series (complexidade near-linear)
- [ ] `inception-mbzuai/jais-2-8b` (Apache-2.0) para sentimento árabe (UAE top-5 crypto hub)
- [ ] `scb10x/typhoon-12b` para mercado tailandês
- [ ] Upbit API (310 moedas, 650 pares, REST+WebSocket) para dados coreanos
- [ ] Kimchi premium como feature: mean-reverting 1.24%, convergência 24 minutos

---

## Fase 11: Distributed Training & Advanced Infrastructure (BAIXA PRIORIDADE)

> **Impacto esperado**: Escalabilidade e eficiencia de treino
> **Fonte**: Flower AI, DeepSeek MoE

### 11.1 Flower AI Federated Learning
- [ ] Integrar framework Flower (Apache-2.0) para treino distribuido
- [ ] Treinar modelos em dados de multiplas exchanges sem centralizar dados
- [ ] Privacy-preserving: enviar apenas weight updates, nao dados brutos
- [ ] Escalar treino para multiplas GPUs/maquinas
- **GitHub**: https://github.com/flwrlabs/flower

### 11.2 DeepSeek MoE Architecture para Ensemble Customizado
- [ ] Estudar arquitetura DeepSeek-V3 (256 routed experts + 1 shared)
- [ ] Aplicar padrao ao nosso ensemble: cada modelo como "expert"
- [ ] MuonClip optimizer para treino sem loss spikes
- **Ref**: DeepSeek-V3 treinado por apenas $5.6M (extremamente eficiente)

### 11.3 TradingView Integration
- [ ] Receber sinais de TradingView via webhook
- [ ] Enviar alertas de sinais para TradingView
- [ ] Pine Script para indicadores customizados

### 11.4 Vector Database (ChromaDB/Pinecone)
- [ ] Migrar market context memory para ChromaDB (open-source) ou Pinecone
- [ ] Busca semantica em escala com metadata filtering
- [ ] Armazenar embeddings de noticias historicas para retrieval

### 11.5 Causal Reasoning
- [ ] Substituir correlacao por causalidade nas features
- [ ] Causal discovery automatica entre variaveis (Granger causality, PCMCI)

### 11.6 Evolutionary Model Merging (Sakana AI — Nature Machine Intelligence)
- [ ] Implementar evolutionary model merging para otimizar ensemble de 16 modelos
- [ ] Usar algoritmo genetico para descobrir pesos e combinacoes otimas automaticamente
- [ ] Sakana AI + MUFG Bank: validado em producao no maior banco do Japao
- [ ] Substitui/complementa performance_weighted com otimizacao automatica
- **Paper**: Sakana AI, Nature Machine Intelligence
- **GitHub**: github.com/SakanaAI/evolutionary-model-merge

### 11.7 VISTA Framework — Chart Analysis Training-Free
- [ ] Implementar VISTA: VLM + imagens de graficos + texto de precos + CoT prompts
- [ ] Training-free: sem fine-tuning adicional, apenas inferencia
- [ ] Reduz erro em 89.83% vs text-only (paper)
- [ ] Gerar candlestick images automaticamente dos dados OHLCV
- [ ] Testar com InternVL3.5-14B, Qwen3-VL-8B ou MiniCPM-V-4_5
- **Paper**: arXiv:2505.18570

### 11.8 ProbFM — Evidential Regression para Crypto
- [ ] Integrar ProbFM: evidential regression validada especificamente em crypto
- [ ] Intervalos de confianca mais calibrados que MC Dropout
- [ ] Complementa/substitui BNN existente
- [ ] Urgente: nosso conformal prediction mostra 0% confianca — ProbFM pode resolver

---

## Metricas de Benchmark (Pipeline v2 — Performance Weighted Ensemble)

> BTC, 730 dias, 5 walk-forward folds, 158 features, 13 modelos (inclui LightGBM)

### Ensemble v1 vs v2 (media 5 folds)

| Metrica | v1 (Stacking) | **v2 (Perf. Weighted)** | Melhoria |
|---------|---------------|-------------------------|----------|
| **Dir. Accuracy** | 56.1% | **69.0%** | **+12.9pp** |
| RMSE | 0.0376 | **0.0337** | -10.4% |
| Correlacao | 0.5394 | **0.6054** | +12.2% |
| **Sharpe Ratio** | 1.56 | **6.92** | **+343%** |
| **Profit Factor** | 1.34 | **3.98** | **+197%** |

### Ensemble Dir. Accuracy por Fold (v2)

| Fold | Dir. Acc | Modelos Excluidos | Top Peso |
|------|----------|-------------------|----------|
| 1 | **71.4%** | TCN | SVM 22.5% |
| 2 | **69.4%** | - | SVM 23.7% |
| 3 | **66.7%** | - | SVM+MDN 19.5% |
| 4 | **74.5%** | TCN, BNN | GRU 26.7% |
| 5 | **63.0%** | CNN-LSTM, TCN | MDN 26.9% |

### Previsao BTC (16/03/2026)
- **Direcao**: BAIXA | **Preco**: $72,815 -> $65,987 (-9.4%)

### Observacoes v2
- Ensemble agora rivaliza com melhores modelos individuais (69.0% vs 70.4% SVM)
- Filtragem automatica de modelos fracos (TCN excluido em 3/5 folds)
- SVM + XGBoost recebem ~40% do peso total
- LightGBM corrigido e contribuindo 12.3%
- Sharpe 6.92 e Profit Factor 3.98 excedem metas

### Metricas Consolidadas do Ensemble v1 (referencia)

| Metrica | Valor |
|---------|-------|
| Acuracia Direcional | 56.1% |
| RMSE | 0.0376 |
| Correlacao | 0.5394 |
| **Sharpe Ratio** | **1.5608** |
| **Profit Factor** | **1.3394** |

> **Meta pos-Fase 5-7**: Acuracia direcional do ensemble >75% com Sharpe >2.0 (**Sharpe ja superado em 3.5x!**)

---

## Matriz de Prioridade — Todas as Descobertas

### Tier 1 — Game Changers (Implementar Primeiro)

| Recurso | Tipo | Impacto | Licenca | Esforco |
|---------|------|---------|---------|---------|
| Chronos-Bolt (Amazon) | Time Series Foundation Model | Muito Alto | Apache-2.0 | Medio |
| TimesFM 2.0 (Google) | Time Series Foundation Model | Muito Alto | Apache-2.0 | Medio |
| FinBERT (ProsusAI) | Financial Sentiment | Alto | Aberta | Baixo |
| Twitter-RoBERTa (CardiffNLP) | Social Sentiment | Alto | CC-BY-4.0 | Baixo |
| MoE Gating Network (MIGA) | Ensemble Intelligence | Muito Alto | Paper aberto | Alto |
| DeepSeek-R1 API | Reasoning Agent | Muito Alto | MIT | Medio |
| MiniMax-M2.5 | Financial Reasoning (65.5 FinSearch) | Muito Alto | Custom | Medio |
| **VISTA Framework** | **VLM Chart Analysis Training-Free (+89.83%)** | **Muito Alto** | **Paper** | **Medio** |
| **Sakana Evo Merge** | **Ensemble Optimization (Nature MI)** | **Muito Alto** | **Apache-2.0** | **Medio** |
| **ProbFM** | **Evidential Regression Crypto** | **Muito Alto** | **Paper** | **Medio** |
| **GLM-4.7-Flash** | **Financial LLM A-grade (MIT, 1.8M dl)** | **Alto** | **MIT** | **Medio** |

### Tier 2 — Forte Impacto

| Recurso | Tipo | Impacto | Licenca | Esforco |
|---------|------|---------|---------|---------|
| MOIRAI (Salesforce) | Multivariate Forecasting | Alto | CC-BY-NC-4.0 | Medio |
| MOMENT (CMU) | Multi-Task TS | Alto | MIT | Medio |
| Qwen3 Max API | Trading Strategy | Alto | Tongyi License | Baixo |
| FinGPT Forecaster | LLM Prediction | Alto | Apache-2.0 | Medio |
| Seed-OSS-36B (ByteDance) | Long Context Agent (512K) | Alto | Apache-2.0 | Medio |
| SenseNova-MARS (SenseTime) | Chart Visual Reasoning (#1 CN) | Alto | Open | Medio |
| ERNIE 4.5-21B-A3B (Baidu) | Chinese NLP Agent | Alto | Apache-2.0 | Baixo |
| finbert-fls | Forward-Looking Detection | Medio | Aberta | Baixo |
| MiniCPM4.1 | Local LLM | Medio | Apache-2.0 | Medio |
| **DeepSeek-OCR-2** | **OCR (substitui HunyuanOCR, 1.3M dl)** | **Alto** | **Apache-2.0** | **Baixo** |
| **InternVL3.5-14B** | **VLM Chart Analysis** | **Alto** | **Apache-2.0** | **Medio** |
| **KorFinMTEB** | **26 Datasets Financeiros KR** | **Alto** | **Open** | **Baixo** |
| **DISC-FinLLM (Fudan)** | **Chinese Financial NLP** | **Alto** | **Open** | **Baixo** |
| **Falcon-Mamba-7B** | **SSM Near-Linear TS** | **Alto** | **Apache-2.0** | **Medio** |
| **CryptoPulse** | **Dual-Prediction Crypto** | **Alto** | **Paper** | **Medio** |
| **Baichuan4-Finance API** | **93.62% Financial (API)** | **Alto** | **API** | **Baixo** |

### Tier 3 — Enhancements

| Recurso | Tipo | Impacto | Licenca | Esforco |
|---------|------|---------|---------|---------|
| SKT A.X VL-Light | Chart Analysis | Medio | Apache-2.0 | Alto |
| NeMo Data Designer | Synthetic Data | Medio | Apache-2.0 | Alto |
| Lag-Llama | Probabilistic TS | Medio | Apache-2.0 | Medio |
| Time-MoE | TS Foundation MoE | Medio | Aberta | Alto |
| Kimi K2.5 API | Long Context | Medio | MIT | Baixo |
| Hunyuan-A13B (Tencent) | Dual-Mode Reasoning | Medio | Tencent License | Medio |
| DanQing Pipeline | News Filtering (DeepGlint AI) | Medio | CC-BY-4.0 | Medio |
| **DLT-Corpus** | **2.98B tokens blockchain** | **Medio** | **Open** | **Baixo** |
| **Economy Watchers Survey** | **Sentimento JP mensal** | **Medio** | **Open** | **Baixo** |
| **Won/FINKRX** | **80K instruções fin. KR** | **Medio** | **ACL 2025** | **Medio** |
| **PaddleOCR-VL** | **OCR CPU (0.9B)** | **Medio** | **Apache-2.0** | **Baixo** |
| **FinTSB benchmark** | **TS financeiro 15 anos** | **Medio** | **Open** | **Baixo** |
| **MME-Finance** | **Benchmark candlestick** | **Medio** | **Open** | **Baixo** |

### Tier 4 — Regional/Experimental

| Recurso | Tipo | Impacto | Licenca | Esforco |
|---------|------|---------|---------|---------|
| Swallow 120B (JP) | Japanese NLP (expandido) | Medio | Apache-2.0 | Medio |
| finbert-tone-chinese | Chinese Sentiment | Medio-Baixo | Apache-2.0 | Baixo |
| Flower AI | Federated Learning | Baixo | Apache-2.0 | Alto |
| SKT A.X K1 (519B/33B MoE) | Korean LLM | Baixo | Apache-2.0 | Muito Alto |
| Sarvam 105B (India) | Indian LLM | Baixo | Apache-2.0 | Alto |
| AIKosh Datasets | Indian Macro Data (3,000+) | Baixo | Variada | Baixo |
| **PhoBERT (Vietnã)** | **Sentimento crypto VN (top-3 adoção)** | **Medio** | **MIT** | **Baixo** |
| **IndoBERT (Indonésia)** | **Maior mercado crypto SEA** | **Medio-Baixo** | **MIT** | **Baixo** |
| **Jais-2 (UAE)** | **Sentimento árabe (top-5 crypto)** | **Medio-Baixo** | **Apache-2.0** | **Baixo** |
| **PLaMo-fin-base (JP)** | **Financial LLM (PFN)** | **Medio** | **Incerta** | **Medio** |
| **K-EXAONE 236B (LG)** | **#7 global MoE** | **Medio** | **Incerta** | **Alto** |
| **Kimchi premium feature** | **Mean-revert 1.24%, 24min** | **Medio** | **N/A** | **Baixo** |

---

## Referencias — Pesquisa Internacional de IA (Marco 2026)

### Modelos de Time Series Foundation
| Modelo | Organizacao | Params | Downloads | Licenca |
|--------|-------------|--------|-----------|---------|
| Chronos-Bolt | Amazon | 8M-200M | 7.1M | Apache-2.0 |
| TimesFM 2.0 | Google | 500M | 17K | Apache-2.0 |
| MOIRAI 1.1 | Salesforce | Large | 1.5M | CC-BY-NC-4.0 |
| MOMENT-1 | CMU AutonLab | Large | 235K | MIT |
| Lag-Llama | TSFL | ~1M | Gated | Apache-2.0 |
| Time-MoE | ICLR 2025 | 2.4B | Paper | Aberta |
| Timer-S1 | Mar 2026 | 8.3B MoE | Paper | Aberta |

### Modelos de NLP Financeiro
| Modelo | Organizacao | Downloads | Foco |
|--------|-------------|-----------|------|
| ProsusAI/finbert | Prosus | 5.9M | Sentimento financeiro |
| twitter-roberta | CardiffNLP | 3.1M | Social media sentiment |
| finbert-tone | HKUST | 832K | Tom financeiro |
| finbert-tone-chinese | HKUST | 287K | Sentimento chines |
| finbert-fls | HKUST | 2.5K | Forward-looking statements |
| FinancialBERT | Ahmed Rachid | 44K | Sentimento alternativo |
| FinGPT Forecaster | AI4Finance | 1.3K | LLM + previsao |
| distilroberta-fin | mrm8488 | 371K | Sentimento rapido |

### LLMs para Crypto Trading (Alpha Arena Results)
| Modelo | Retorno | Estilo | Acesso |
|--------|---------|--------|--------|
| Qwen3 Max | +22.32% (2 sem) | Tecnico + stop-loss rigoroso | Alibaba Cloud API |
| DeepSeek V3.1 | +126% (season) | Quant diversificado | DeepSeek API (MIT) |
| DeepSeek-R1 | +4.89% (season) | Reasoning bull/bear | DeepSeek API (MIT) |
| GPT-5 | -60% | Wiped out | OpenAI API |
| Gemini 2.5 Pro | Negativo | Wiped out | Google API |

### Plataformas e Datasets Regionais
| Recurso | Pais | Tipo | URL |
|---------|------|------|-----|
| ModelScope | China | Hub de modelos (70K+) | modelscope.cn |
| CSTCloud | China | Infraestrutura (sem modelos financeiros) | cstcloud.cn |
| AIKosh | India | Datasets nacionais (3,000+) | aikosh.indiaai.gov.in |
| Swallow LLM | Japao | LLMs JP/EN (agora 120B) | swallow-llm.github.io |
| CCI4.0 | China (BAAI) | 35TB dataset + 430M CoT | huggingface.co/BAAI |
| KorFinMTEB | Coreia | 26 datasets financeiros | HuggingFace |
| DISC-Fin-SFT | China (Fudan) | 250K instruções financeiras | HuggingFace |
| DLT-Corpus | Multi | 2.98B tokens blockchain | HuggingFace |
| MME-Finance | China | Benchmark multimodal + candlestick | HuggingFace |
| Economy Watchers | Japao | Sentimento mensal desde 2000 | HuggingFace |
| FinTSB | China (Tongji) | Benchmark TS 15 anos | GitHub |
| NeMo DataDesigner | NVIDIA | Dados sinteticos | github.com/NVIDIA-NeMo |
| Flower AI | Global | Federated learning | flower.ai |

### Modelos Novos — Descoberta V2.1 (Marco 2026)
| Modelo | Pais | Tipo | Params | License | Relevancia |
|--------|------|------|--------|---------|-----------|
| GLM-4.7-Flash | China | Financial LLM (A-grade) | MoE | MIT | 8/10 |
| DeepSeek-OCR-2 | China | OCR (1.3M downloads) | - | Apache-2.0 | 7/10 |
| DeepSeek-V3.2 | China | LLM (dez 2025) | MoE | MIT | 9/10 |
| Qwen3.5-35B-A3B | China | MoE multimodal (fev 2026) | 35B/3B | Apache-2.0 | 8/10 |
| InternVL3.5-14B | China | VLM chart analysis | 14B | Apache-2.0 | 8/10 |
| DISC-FinLLM | China (Fudan) | Financial NLP chinês | 13B LoRA | Open | 7/10 |
| Baichuan4-Finance | China | 93.62% FLAME (API only) | - | API | 9/10 |
| PaddleOCR-VL | China | OCR CPU-friendly | 0.9B | Apache-2.0 | 7/10 |
| K-EXAONE 236B | Coreia | #7 global MoE | 236B | - | 8/10 |
| HyperCLOVA X Omni | Coreia | NAVER multimodal | 8B | - | 7/10 |
| PLaMo-fin-base | Japao | Financial LLM (PFN) | - | - | 8/10 |
| GPT-OSS-Swallow-120B | Japao | Maior LLM JP open | 120B | Apache-2.0 | 7/10 |
| Falcon-Mamba-7B | UAE | SSM near-linear | 7B | Apache-2.0 | 8/10 |
| Jais-2-8B | UAE | Arabic-English | 8B | Apache-2.0 | 7/10 |
| PhoBERT | Vietnã | Vietnamese NLP | 135M | MIT | 7/10 |

### Papers Fundamentais
| Paper | Ano | Contribuicao |
|-------|-----|--------------|
| MIGA (MoE for Stock Market) | 2024 | 24% retorno excedente CSI300 |
| Time-MoE (ICLR Spotlight) | 2025 | MoE para time series, 2.4B params |
| DanQing (Cross-Modal Filtering) | 2026 | Pipeline de filtragem multi-estagio (DeepGlint AI) |
| DEJIMA (JP Vision-Language) | 2025 | 3.88M pares imagem-texto (U. Tokyo) |
| Re(Visiting) TSFMs in Finance | 2025 | **CRITICO: Fine-tune >> zero-shot em finance** |
| DeXposure-FM | 2026 | Graph NN para DeFi credit |
| PreBit | 2022 | FinBERT + CNN multimodal BTC |
| Timer-S1 | 2026 | 8.3B MoE SOTA em GIFT-Eval |
| **VISTA** | **2025** | **VLM + charts training-free, +89.83%** |
| **ProbFM** | **2025** | **Evidential regression validada em crypto** |
| **CryptoPulse** | **2025** | **Dual-prediction crypto-specific** |
| **Meta-RL-Crypto** | **2025** | **RL auto-melhorante com on-chain** |
| **QuantAgent** | **2025** | **Multi-agent HFT Bitcoin** |
| **FinMamba** | **2025** | **Graph + Mamba para correlacoes crypto** |
| **Won/FINKRX** | **2025** | **ACL: NLP financeiro coreano (80K instruções)** |
| **Baichuan4-Finance** | **2024** | **93.62% FLAME, domain self-constraint** |
| **DISC-FinLLM** | **2023** | **Fudan: LLM financeiro chinês completo** |
| **Evolutionary Model Merging** | **2024** | **Sakana AI: otimização de ensemble (Nature MI)** |
| **Kimchi Premium Dynamics** | **2024** | **Mean-reversion 1.24%, conv. 24min** |
| **EDINET-Bench** | **2026** | **Sakana AI: benchmark financeiro JP (ICLR)** |
| **DLT-Corpus** | **2026** | **2.98B tokens blockchain/DLT** |
| **FinTSB** | **2025** | **Tongji: benchmark time series financeiro** |
| **MME-Finance** | **2024** | **Benchmark multimodal + candlestick charts** |

### Relatório Completo de Pesquisa
- **Arquivo**: `reports/RELATORIO_IA_ORIENTAL_V2.1.md`
- **Cobertura**: 200+ modelos, 100+ papers, 50+ datasets, 30+ benchmarks, 40+ universidades
- **Países**: China, Coreia, Japão, Índia, Singapura, Taiwan, Vietnã, UAE, Indonésia, Tailândia
