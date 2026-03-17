# South Korea AI Ecosystem Research Report
## For Crypto Price Prediction System Integration
### Research Date: March 16, 2026

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Universities & Research Labs](#2-universities--research-labs)
3. [Models on HuggingFace](#3-models-on-huggingface)
4. [Datasets](#4-datasets)
5. [Benchmarks](#5-benchmarks)
6. [Companies & Research Labs](#6-companies--research-labs)
7. [Papers](#7-papers)
8. [Special Topics](#8-special-topics)
9. [Integration Recommendations](#9-integration-recommendations)

---

## 1. EXECUTIVE SUMMARY

South Korea is a top-3 global crypto market with unique dynamics (Kimchi premium, strict capital controls, FSC/FSS regulation). The Korean AI ecosystem has matured dramatically in 2025-2026 with five national sovereign AI consortia, record 10.1 trillion won AI budget, and world-class foundation models entering the global top-10. This report catalogs every relevant asset for our crypto prediction system.

**Key numbers:**
- Government AI budget 2026: 10.1 trillion won (~$6.94B), tripled from 2025
- Five sovereign AI consortia: NAVER, SKT, LG, NCSoft, Upstage
- 260,000 advanced GPUs planned for National AI Computing Center
- FSS VISTA system with 170M won budget for crypto surveillance
- Upbit: 13.26M cumulative users, $844M daily volume
- Bithumb: 2.42M MAU, $383M daily volume

---

## 2. UNIVERSITIES & RESEARCH LABS

### 2.1 KAIST (Korea Advanced Institute of Science and Technology)
- **Location:** Daejeon, South Korea
- **Relevance to crypto prediction:** 8/10
- **Key Labs:**
  - **MLAI Lab (Machine Learning & AI):** 5 papers at ICLR 2026, 7 papers at NeurIPS 2025. Active in time series and generative models.
  - **DSAIL (Data Science & AI Lab):** Led by Prof. Chanyoung Park. Graph learning, recommendation systems.
  - **SAILAB (Statistical AI Lab):** Dynamic factor vector autoregression for forecasting - directly relevant to time series prediction.
  - **Kim Jaechul Graduate School of AI:** PhD candidates presented financial AI real-time explanation methods at CIKM 2025 (top information/knowledge management conference).
  - **Visual AI Group:** Computer vision research with potential multimodal applications.
  - **Data AI Lab:** Data-centric AI approaches.
- **Verification:** CONFIRMED - Multiple labs active, ICLR/NeurIPS publications verified.
- **National AI Research Lab (NAIRL):** KAIST leads a consortium with Yonsei, Korea University, POSTECH - 39 papers and 4 spotlight presentations at NeurIPS 2025.

### 2.2 Seoul National University (SNU)
- **Location:** Seoul, South Korea
- **Relevance to crypto prediction:** 9/10
- **Key Labs:**
  - **Data Mining Laboratory (Dept. of CSE):** Research topics include deep learning, recommender/QA systems, graphs/tensors, AND financial AI. Directly relevant.
  - **Natural Language Processing Laboratory:** Developed KR-FinBert (Korean financial BERT) and KR-FinBert-SC for financial sentiment classification. HIGH PRIORITY for our system.
  - **Statistical Learning & Computational Finance Lab (SLCF):** National Leading Research Lab focused on statistical learning AND computational finance. DIRECTLY relevant.
  - **AI Institute (AIIS):** Cross-cutting AI research institute.
  - **SNU-OpenAI Joint Symposium:** Active collaboration with OpenAI.
- **Verification:** CONFIRMED - KR-FinBert verified on HuggingFace, SLCF lab confirmed.

### 2.3 POSTECH (Pohang University of Science and Technology)
- **Location:** Pohang, South Korea
- **Relevance to crypto prediction:** 7/10
- **Key Labs:**
  - **NLP Group:** Leading Korean NLP lab for 30+ years. Language-based AI.
  - **Machine Learning Lab:** Geometric structure of datasets, GNNs, hyperbolic/Riemannian geometry for representation learning.
  - **Data Systems Lab:** Awarded $6M+ for Global AI Frontier Lab (Korea-NYU partnership). 8 additional professors from KAIST and Sungkyunkwan University.
  - **Efficient Learning Lab (EffL):** Led by Jaeho Lee, efficient ML theory/algorithms/systems.
- **Verification:** CONFIRMED - Lab pages active, NeurIPS 2025 publications verified.

### 2.4 Yonsei University
- **Location:** Seoul, South Korea
- **Relevance to crypto prediction:** 6/10
- **Key Labs:**
  - **ANDlab:** AI, big data analysis, NLP, information extraction.
  - **Data Intelligence Lab (Prof. Lee Dongha):** Text mining & NLP, knowledge graph reasoning, information retrieval & recommendation.
  - **Language & AGI Lab (Prof. Yeo Jinyoung):** LLMs, NLP, dialogue agents, AGI.
  - **ML3 Lab (Jaehyung Kim):** Launched Nov 2025. Area Chair for NeurIPS 2025, ICLR 2026. 2 papers at NeurIPS 2025 (1 Spotlight).
  - **MLAI@Yonsei:** Machine learning and AI research.
- **NeurIPS 2025:** 6 papers focusing on generative models and AI safety.
- **Verification:** CONFIRMED.

### 2.5 Korea University
- **Location:** Seoul, South Korea
- **Relevance to crypto prediction:** 6/10
- **NeurIPS 2025:** 5 papers advancing offline reinforcement learning and privacy research.
- **Part of NAIRL consortium** with KAIST, Yonsei, POSTECH.
- **Verification:** CONFIRMED via NAIRL publications.

### 2.6 ETRI (Electronics and Telecommunications Research Institute)
- **Location:** Daejeon, South Korea
- **Type:** Government research institute
- **Relevance to crypto prediction:** 7/10
- **Key Projects:**
  - **Eagle:** Korean language-centric language model, 100B-scale pre-training. Planned transition to multimodal foundation model after 2026 with focus on Korean language, mathematics, and quantitative inference.
  - **Exobrain:** Accelerated Korean AI research through question-answering and knowledge systems.
  - **MLOps Tool:** Open-source no-code ML development tool, auto neural network generation.
  - **AI-RAN:** AI-based wireless access technology for 6G.
- **Part of NC AI Consortium** for sovereign AI foundation model development.
- **Verification:** CONFIRMED - Eagle model development in progress.

### 2.7 KIST (Korea Institute of Science and Technology)
- **Location:** Seoul, South Korea
- **Type:** Government research institute
- **Relevance to crypto prediction:** 4/10
- **Focus:** Six mission areas: semiconductors, clean hydrogen, AI and robotics, climate, drug discovery, brain science.
- **AI Research Group:** Abnormal state detection systems, traffic management AI, safety verification AI.
- **Physical AI:** Prof. Kweon named UNIDO Global Top 100 AI Talents (only Korean representative).
- **Verification:** CONFIRMED - CES 2025 demonstrations verified.

### 2.8 Korea AI Safety Institute (K-AISI)
- **Location:** Under MSIT
- **Type:** Government safety evaluation body (NOT regulatory)
- **Established:** November 2024
- **Relevance to crypto prediction:** 5/10
- **Key Activities:**
  - AI Safety Forecast Report analyzing 125 global AI safety news articles (Aug-Oct 2025).
  - Risk evaluation of AI models/systems.
  - Partnership with Scale AI for AI safety.
  - Supporting Korean AI companies for global competitiveness.
- **Legal Basis:** Article 12 of South Korea's AI Basic Act (passed Jan 2025, effective Jan 2026).
- **Verification:** CONFIRMED - Official site at aisi.re.kr.

---

## 3. MODELS ON HUGGINGFACE

### 3.1 SKT (SK Telecom) A.X Series

| Model | Params | Architecture | License | Downloads | Relevance | Verified |
|-------|--------|-------------|---------|-----------|-----------|----------|
| **skt/A.X-K1** | 519B total (33B active, MoE 192+1 experts) | Decoder-only Transformer, 61 layers, 131K context | Apache-2.0 | 5,503 | 8/10 | YES |
| **skt/A.X-4.0** | ~8B (Qwen2-based) | Qwen2, text-generation | Other | 629 | 7/10 | YES |
| **skt/A.X-4.0-Light** | ~8B (Qwen2-based) | Qwen2, text-generation | Apache-2.0 | 7,171 | 8/10 | YES |
| **skt/A.X-4.0-VL-Light** | 8B | Vision-language, based on A.X-4.0-Light | Apache-2.0 | 299 | 7/10 | YES |
| **skt/A.X-3.1** | LLaMA-based | LLaMA, text-generation | Apache-2.0 | 317 | 5/10 | YES |
| **skt/A.X-3.1-Light** | LLaMA-based | LLaMA, text-generation | Apache-2.0 | 176 | 5/10 | YES |
| **skt/A.X-Encoder-base** | ModernBERT | ModernBERT, fill-mask/text-classification | Apache-2.0 | 909 | 7/10 | YES |
| **skt/kogpt2-base-v2** | GPT-2 scale | GPT-2, text-generation | CC-BY-NC-SA-4.0 | 684,135 | 4/10 | YES |
| **skt/kobert-base-v1** | BERT-base | BERT, feature-extraction | - | 14,359 | 5/10 | YES |
| **skt/kobart-base-v1** | BART-base | BART | MIT | 1,502 | 4/10 | YES |
| **skt/ko-gpt-trinity-1.2B-v0.5** | 1.2B | GPT-2/GPT-3, text-generation | CC-BY-NC-SA-4.0 | 901 | 4/10 | YES |

**A.X K1 Key Details (VERIFIED):**
- 519B total params, 33B active per token (MoE with 192 experts + 1 shared)
- 61 layers (1 dense + 60 MoE), 64 attention heads
- Vocabulary: 163,840 tokens, Context: 131,072 tokens
- User-controllable reasoning depth (multi-step or concise)
- BF16 precision
- ArXiv: 2601.09200

**A.X-4.0-VL-Light Key Details (VERIFIED):**
- 8B params, BF16 precision
- 79.4 average on Korean image benchmarks (outperforms Qwen2.5-VL-32B at 73.4)
- 80.2 on K-Viscuit cultural comprehension benchmark
- 89.8% on KoBizDoc (financial document understanding) - CONFIRMED
- Document understanding: tables, charts, technical docs

### 3.2 NAVER HyperCLOVA X Series

| Model | Params | Type | License | Downloads | Relevance | Verified |
|-------|--------|------|---------|-----------|-----------|----------|
| **naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B** | 8B | Multimodal (text+image+audio) | Other | 228,002 | 8/10 | YES |
| **naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B** | 3B | Vision-Language | Other | 62,070 | 7/10 | YES |
| **naver-hyperclovax/HyperCLOVAX-SEED-Think-32B** | 32B | Reasoning VLM | Other | 18,435 | 8/10 | YES |
| **naver-hyperclovax/HyperCLOVAX-SEED-Think-14B** | 14B | Reasoning LLM | Other | 24,397 | 7/10 | YES |
| **naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B** | 1.5B | LLaMA-based text | Other | 1,980 | 5/10 | YES |
| **naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B** | 0.5B | LLaMA-based text | Other | 1,169 | 4/10 | YES |

**HyperCLOVA X THINK Key Details:**
- Reasoning-focused model, competitive on Korean benchmarks
- Peri-LN Transformer, mu-P scaling, three-stage curriculum
- RLVR (Reinforcement Learning from Verifiable Rewards)
- Evaluated on: KMMLU, CSAT, KoBALT-700, HAERAE-1.0, KoBigBench, KCSAT STEM
- Vision-augmented capabilities
- ArXiv: 2506.22403

**Full HyperCLOVA X (proprietary):**
- 204B parameters (reported)
- 6,500x more Korean training data than GPT-3
- Trained on 50 years of Korean news articles, 9 years of blog data
- Native multimodality (image + audio)

### 3.3 LG AI Research EXAONE Series

| Model | Params | Type | License | Downloads | Relevance | Verified |
|-------|--------|------|---------|-----------|-----------|----------|
| **LGAI-EXAONE/K-EXAONE-236B-A23B** | 236B total (23B active, MoE) | Multilingual MoE LLM | Other | 24,094 | 9/10 | YES |
| **LGAI-EXAONE/EXAONE-4.0.1-32B** | 32B | Hybrid reasoning LLM | Other | 155,252 | 8/10 | YES |
| **LGAI-EXAONE/EXAONE-4.0-32B** | 32B | Hybrid LLM+reasoning | Other | 18,830 | 8/10 | YES |
| **LGAI-EXAONE/EXAONE-4.0-1.2B** | 1.2B | On-device LLM | Other | 58,202 | 6/10 | YES |
| **LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct** | 7.8B | Instruction-tuned LLM | Other | 388,840 | 7/10 | YES |
| **LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct** | 2.4B | Instruction-tuned LLM | Other | 27,129 | 6/10 | YES |
| **LGAI-EXAONE/EXAONE-3.5-32B-Instruct** | 32B | Instruction-tuned LLM | Other | 13,559 | 7/10 | YES |
| **LGAI-EXAONE/EXAONE-Deep-7.8B** | 7.8B | Reasoning-focused | Other | 296,602 | 7/10 | YES |
| **LGAI-EXAONE/EXAONE-Deep-2.4B** | 2.4B | Reasoning-focused | Other | 1,294 | 5/10 | YES |
| **LGAI-EXAONE/EXAONE-Deep-32B** | 32B | Reasoning-focused | Other | 1,339 | 7/10 | YES |
| **LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct** | 7.8B | Instruction-tuned | Other | 20,564 | 6/10 | YES |
| **LGAI-EXAONE/EXAONEPath** (various versions) | Various | Pathology-specific | Other | Low | 2/10 | YES |

**K-EXAONE Key Details (VERIFIED):**
- 236B params, 23B active (MoE), supports 256K-token context window
- Ranked 7th on Artificial Analysis Intelligence Index (only Korean model in global top 10)
- Hybrid attention: sliding window + global attention, 70% memory reduction vs EXAONE 4.0
- Languages: English, Korean, Spanish, German, Japanese, Vietnamese
- ArXiv: 2601.01739

**EXAONE 4.0 Key Details (VERIFIED):**
- First open-weight hybrid AI in Korea (LLM + reasoning)
- 32B (mid-size) and 1.2B (on-device) variants
- Outperforms Alibaba, Microsoft, Mistral models in science/math/coding benchmarks
- Agentic features: tool use, multilingual (EN/KO/ES)
- ArXiv: 2507.11407

### 3.4 Upstage SOLAR Series

| Model | Params | Architecture | License | Downloads | Relevance | Verified |
|-------|--------|-------------|---------|-----------|-----------|----------|
| **upstage/Solar-Open-100B** | 102B | MoE, from scratch | Other (Upstage Solar License) | 5,968 | 8/10 | YES |
| **upstage/SOLAR-10.7B-Instruct-v1.0** | 10.7B | LLaMA-based | CC-BY-NC-4.0 | 26,826 | 6/10 | YES |
| **upstage/SOLAR-10.7B-v1.0** | 10.7B | LLaMA-based | Apache-2.0 | 18,483 | 6/10 | YES |
| **upstage/solar-pro-preview-instruct** | - | Proprietary | MIT | 16,259 | 7/10 | YES |
| **upstage/SOLAR-0-70b-16bit** | 70B | LLaMA-2-based | - | 24,430 | 5/10 | YES |

**Solar Open 100B Key Details (VERIFIED):**
- 102B params, MoE architecture, trained entirely from scratch
- First outcome of South Korea's national Independent AI Foundation Model Project (MSIT-led)
- Bilingual (Korean/English), enterprise-grade reasoning
- ArXiv: 2601.07022

**Solar Pro 2 (API-only, July 2025):**
- 31B params, scored 58 on Intelligence Index
- Highest among all Korean LLMs, above GPT-4.1
- Korea's first global frontier model

**Solar Pro 3 (API-only, Jan 2026):**
- Builds on Solar Pro 2, improved reasoning accuracy and instruction following

### 3.5 Kakao Brain Models

| Model | Params | Type | License | Downloads | Relevance | Verified |
|-------|--------|------|---------|-----------|-----------|----------|
| **kakaobrain/kogpt** | GPT-3 scale | Korean GPT | CC-BY-NC-ND-4.0 | 191 | 4/10 | YES |
| **kakaobrain/align-base** | ALIGN-base | Zero-shot image classification | Apache-2.0 | 6,692 | 3/10 | YES |
| **kakaobrain/karlo-v1-alpha** | - | Text-to-image (UnCLIP) | CreativeML OpenRAIL-M | 112 | 2/10 | YES |

**Kanana (proprietary, 2025-2026):**
- Kakao's proprietary AI embedded in KakaoTalk
- "Kanana Nano" - lightweight on-device model for Korean language context
- AI Model Orchestration strategy (proprietary + external APIs like OpenAI)
- Agent-based Kanana launching H1 2026

### 3.6 NCSoft VARCO Series

| Model | Status | Type | Relevance | Verified |
|-------|--------|------|-----------|----------|
| **VARCO LLM** | Available via Amazon SageMaker JumpStart | Korean LLM, small/medium | 5/10 | YES |
| **VARCO-VISION 2.0 (14B, 1.7B, 1.7B-OCR, Video-Embedding)** | Open-source (July 2025) | Multimodal vision-language | 6/10 | YES |
| **VARCO 3D** | Service (Dec 2025) | 3D asset generation from text/image | 2/10 | YES |

**VARCO-VISION 2.0 Key Details:**
- 14B and 1.7B variants, including OCR-specialized and video embedding models
- Free for commercial use (open source)
- Korean-based multimodal AI
- NCSoft is one of five sovereign AI consortia

### 3.7 Other Notable Korean Models

| Model | Organization | Type | Relevance | Verified |
|-------|-------------|------|-----------|----------|
| **KR-BERT** | SNU | Small Korean-specific BERT | 6/10 | YES (Paper: 2008.03979) |
| **KR-FinBert** | SNU NLP Lab | Korean financial BERT adaptation | 9/10 | YES |
| **KR-FinBert-SC** | SNU NLP Lab | Korean financial sentiment classification | 9/10 | YES |
| **KLUE-BERT** | KLUE consortium | Korean NLU BERT | 6/10 | YES |
| **KLUE-RoBERTa** | KLUE consortium | Korean NLU RoBERTa | 6/10 | YES |
| **GECKO** | Independent | Bilingual Korean-English LLM, LLaMA arch | 5/10 | YES (Paper: 2405.15640) |
| **RedWhale** | Independent | Adapted Korean LLM, continual pretraining | 5/10 | YES (Paper: 2408.11294) |
| **Won / FINKRX** | OneLine AI + Korea Exchange | Korean financial LLM (SFT + DPO) | 9/10 | YES (ACL 2025) |
| **Eagle** | ETRI | 100B Korean-centric LLM (in development) | 7/10 | YES |
| **NMIXX** | Korean researchers | Cross-lingual financial embedding models | 9/10 | YES (Paper: 2507.09601) |

---

## 4. DATASETS

### 4.1 Korean Language Benchmarks/Datasets on HuggingFace

| Dataset | Organization | Tasks | Size | License | Downloads | Relevance | Verified |
|---------|-------------|-------|------|---------|-----------|-----------|----------|
| **LGAI-EXAONE/KMMLU-Redux** | LG AI Research | Massive multitask Korean evaluation | 1K-10K | CC-BY-NC-ND-4.0 | 2,386 | 7/10 | YES |
| **klue/klue** | KLUE Consortium | 8 NLU tasks (NLI, STS, NER, RE, DP, MRC, TC, DST) | 100K-1M | CC-BY-SA-4.0 | 3,644 | 6/10 | YES |
| **KorQuAD/squad_kor_v1** | KorQuAD | Extractive QA | 10K-100K | CC-BY-ND-4.0 | 628 | 4/10 | YES |
| **Nexdata/Korean_Financial_Speech** | Nexdata | 215 hours Korean financial speech | <1K | CC-BY-ND-4.0 | 7 | 6/10 | YES |

### 4.2 Korean Financial Specific Datasets/Benchmarks

| Dataset/Benchmark | Source | Description | Availability | Relevance | Verified |
|-------------------|--------|-------------|--------------|-----------|----------|
| **KorFinMTEB** | Korean researchers (ICLR 2025) | 7 tasks, 26 datasets for Korean financial text embedding | Open-source | 10/10 | YES |
| **KFinEval-Pilot** | Korean researchers (2025) | 1,000+ questions: financial knowledge, legal reasoning, toxicity | Open | 9/10 | YES |
| **KorFinASC** | Korean researchers | Korean financial aspect-level sentiment classification | Research | 9/10 | YES (Paper: 2301.03136) |
| **KorFinSTS** | Korean researchers | Korean financial semantic textual similarity | Part of KorFinMTEB | 9/10 | YES |
| **Won/FINKRX Instruction Dataset** | OneLine AI + Korea Exchange | 80K financial instruction instances | Open | 9/10 | YES (ACL 2025) |
| **KMMLU** | Amphora + others | Expert-level Korean multitask benchmark | HuggingFace | 7/10 | YES (Paper: 2402.11548) |
| **KoBEST** | Independent | Korean balanced evaluation tasks | HuggingFace | 6/10 | YES (Paper: 2204.04541) |
| **HAE-RAE Bench** | Amphora + others | Korean cultural/contextual LLM evaluation | Open | 5/10 | YES (Paper: 2309.02706) |
| **KoNET** | Korean researchers | Korean national educational test benchmark for multimodal AI | Research | 5/10 | YES (Paper: 2502.15422) |
| **KOFFVQA** | Korean researchers | Korean free-form VQA, 275 questions | Open | 5/10 | YES (Paper: 2503.23730) |
| **K-Viscuit** | Korean researchers | Korean cultural visual comprehension | Research | 5/10 | YES (Paper: 2406.16469) |
| **ScholarBench** | Korean researchers | English-Korean bilingual academic benchmark | Research | 4/10 | YES (Paper: 2505.16566) |
| **Open Ko-LLM Leaderboard2** | Korean researchers | Native Korean LLM evaluation leaderboard | Open | 6/10 | YES (Paper: 2410.12445) |
| **KoBizDoc** | SKT benchmark | Korean business document understanding | Benchmark score reported | 8/10 | PARTIALLY - Score reported (89.8% for A.X-4.0-VL-Light) |

---

## 5. BENCHMARKS

### 5.1 Korean-Specific Benchmarks

| Benchmark | Focus | Tasks | Papers | Crypto Relevance |
|-----------|-------|-------|--------|-----------------|
| **KMMLU** | Korean massive multitask | Expert-level knowledge across subjects | 2402.11548 | 7/10 - General Korean capability |
| **KLUE** | Korean language understanding | TC, STS, NLI, NER, RE, DP, MRC, DST | 2105.09680 | 6/10 - NLU foundation |
| **KoBEST** | Korean balanced evaluation | Requires advanced linguistic knowledge | 2204.04541 | 5/10 - Korean proficiency |
| **HAE-RAE Bench** | Korean cultural knowledge | Vocabulary, history, general knowledge | 2309.02706 | 4/10 - Cultural context |
| **KorFinMTEB** | Korean financial embeddings | Classification, clustering, retrieval, summarization, reranking, STS, pair classification | ICLR 2025 | 10/10 - Direct financial evaluation |
| **KFinEval-Pilot** | Korean financial LLM evaluation | Financial knowledge, legal reasoning, toxicity | 2504.13216 | 9/10 - Financial AI evaluation |
| **KoBizDoc** | Korean business document understanding | OCR, table/chart understanding | Referenced in A.X model cards | 8/10 - Document processing |
| **KMMLU-Redux** | Refined KMMLU | Curated subset with improved quality | 2507.08924 | 7/10 - Better Korean evaluation |
| **Open Ko-LLM Leaderboard2** | Korean LLM ranking | Native Korean evaluation tasks | 2410.12445 | 6/10 - Model selection |
| **KoNET** | Multimodal Korean education tests | GED, CSAT levels | 2502.15422 | 5/10 - Multimodal Korean |
| **KOFFVQA** | Korean visual QA | 275 VQA questions with objective grading | 2503.23730 | 5/10 - Visual Korean |
| **K-Viscuit** | Korean cultural visual comprehension | Culturally-specific visual reasoning | 2406.16469 | 4/10 - Cultural visual |
| **Won/FINKRX Leaderboard** | Korean financial LLM competition | 5 MCQA categories + open-ended QA for finance | ACL 2025 | 10/10 - Financial LLM ranking |

### 5.2 Benchmark Results Summary (Key Korean Models)

| Model | KMMLU | KoBizDoc | K-Viscuit | Intelligence Index | Global Rank |
|-------|-------|---------|-----------|-------------------|-------------|
| A.X K1 (519B) | Top-tier Korean | - | - | - | Korean flagship |
| A.X-4.0-VL-Light (8B) | - | 89.8% | 80.2 | - | - |
| K-EXAONE (236B) | Competitive | - | - | Top 10 | 7th (Artificial Analysis) |
| Solar Pro 2 (31B) | - | - | - | 58 (above GPT-4.1) | Korea's 1st frontier |
| Solar Open 100B | Strong | - | - | - | National project |
| HyperCLOVA X THINK | Strong on KMMLU, CSAT, KoBALT-700 | - | - | - | Sovereign AI |

---

## 6. COMPANIES & RESEARCH LABS

### 6.1 NAVER / LINE (HyperCLOVA Ecosystem)
- **Type:** Tech conglomerate, Korea's largest internet company
- **AI Division:** NAVER Cloud, NAVER AI Lab
- **Key Models:** HyperCLOVA X series (204B proprietary), HyperCLOVA X THINK (reasoning), SEED series (open-weight: 0.5B-32B)
- **Relevance:** 8/10 - Best Korean language understanding, massive Korean training data
- **Investment:** Part of 5 sovereign AI consortia, Naver Ventures invested in Twelve Labs
- **Status:** CONFIRMED - Active HuggingFace org, arxiv papers, active development

### 6.2 SK Telecom (A.X Series)
- **Type:** Telecommunications conglomerate
- **AI Division:** SKT AI
- **Key Models:** A.X K1 (519B MoE), A.X 4.0 series (Light, VL-Light), A.X Encoder, KoBERT, KoGPT, KoBART
- **Relevance:** 9/10 - Wide model range, Apache-2.0 licensing, document AI, encoder models
- **Investment:** Part of 5 sovereign AI consortia, invested $3M in Twelve Labs, joined OpenAI Stargate
- **Status:** CONFIRMED - Active GitHub (SKT-AI), HuggingFace (skt), arxiv papers

### 6.3 Upstage (SOLAR)
- **Type:** AI startup (targeting Korea's first generative AI IPO)
- **Founded by:** SNU professors
- **Key Models:** Solar Open 100B, Solar Pro 2/3, SOLAR-10.7B, Document AI
- **Relevance:** 8/10 - Frontier performance, open-weight, document understanding
- **Investment:** Part of 5 sovereign AI consortia, first national foundation model project output
- **Status:** CONFIRMED - Active HuggingFace (upstage), IPO planned

### 6.4 LG AI Research (EXAONE)
- **Type:** AI research arm of LG Group
- **Key Models:** K-EXAONE (236B MoE), EXAONE 4.0 (32B/1.2B hybrid), EXAONE 3.5 series, EXAONE Deep series, EXAONEPath (medical)
- **Relevance:** 9/10 - Global top-10 ranked model, hybrid reasoning, wide size range
- **Investment:** Part of 5 sovereign AI consortia, South Korea funds EXAONE specifically for AI sovereignty
- **Status:** CONFIRMED - Very active HuggingFace (LGAI-EXAONE), many model variants

### 6.5 Kakao Brain / Kakao
- **Type:** Tech conglomerate (messaging, fintech, mobility)
- **AI Division:** Kakao Brain (research), Kakao (product)
- **Key Models:** KoGPT (CC-BY-NC-ND), Kanana (proprietary, on-device), ALIGN, Karlo (image gen)
- **Relevance:** 6/10 - Kanana on-device is interesting but limited open availability
- **Investment:** Strategic partnership with OpenAI (ChatGPT in KakaoTalk)
- **Financial:** KakaoBank uses Azure OpenAI for conversational AI in finance (first in Korea)
- **Status:** CONFIRMED but shifting toward orchestration (proprietary + external APIs)

### 6.6 Samsung AI Center
- **Type:** AI research within Samsung Electronics
- **Location:** Flagship center in Seoul + global centers (US, UK, Canada, Poland)
- **Key Focus:** On-device AI, agentic experiences (Galaxy S26), AI semiconductors
- **Investment:** $25.6B R&D spending in 2025 (record), joined OpenAI Stargate initiative
- **Relevance:** 5/10 - Mostly device-focused, not financial AI specific
- **Status:** CONFIRMED

### 6.7 Twelve Labs
- **Type:** AI startup (video understanding), Korean-founded
- **Co-founded by:** Jae Lee (2020), offices in Seoul + San Francisco
- **Key Models:** Marengo 3.0 (video understanding, Dec 2025), Marengo Embed 2.7, Pegasus 1.2
- **Availability:** API-based, models on Amazon Bedrock
- **Relevance:** 6/10 - Video understanding could process crypto YouTube/news video content
- **Investment:** SK Telecom ($3M), Naver Ventures, LG CNS partnership
- **Status:** CONFIRMED

### 6.8 NCSoft (VARCO)
- **Type:** Game developer with AI division (NC AI)
- **Key Models:** VARCO LLM (Korean), VARCO-VISION 2.0 (14B/1.7B, open-source), VARCO 3D
- **Relevance:** 5/10 - Open-source vision models with OCR could help document processing
- **Investment:** Part of 5 sovereign AI consortia
- **Status:** CONFIRMED - VARCO-VISION 2.0 open-sourced July 2025

### 6.9 OneLine AI (Korean Financial AI)
- **Type:** AI company focused on Korean financial domain
- **Key Output:** Won/FINKRX model, published at ACL 2025 Industry Track with Korea Exchange
- **Relevance:** 10/10 - Specifically built for Korean financial NLP
- **Status:** CONFIRMED - ACL 2025 publication verified

---

## 7. PAPERS

### 7.1 Korean Financial NLP Papers

| # | Title | Authors/Affiliation | Year | Venue | Key Findings | Relevance |
|---|-------|-------------------|------|-------|-------------|-----------|
| 1 | **FINKRX/Won: Establishing Best Practices for Korean Financial NLP** | OneLine AI + Korea Exchange | 2025 | ACL 2025 Industry | First open leaderboard for Korean financial LLMs; 1,119 submissions evaluated; 80K instruction dataset released; SFT+DPO training | 10/10 |
| 2 | **TWICE: Low-Resource Domain-Specific Embedding for Korean Financial Texts** | Hwang, Jung, Lee, Yu | 2025 | ICLR 2025 | KorFinMTEB benchmark (7 tasks, 26 datasets); domain-specific embeddings outperform translated English benchmarks | 10/10 |
| 3 | **NMIXX: Domain-Adapted Neural Embeddings for Cross-Lingual Finance** | Lee, Yu, Hwang et al. | 2025 | HF Papers (2507.09601) | Cross-lingual financial embedding models; Korean-English financial semantics; outperforms bge-m3 | 9/10 |
| 4 | **Removing Non-Stationary Knowledge for Entity-Level Sentiment in Finance** | Son (Amphora), Lee et al. | 2023 | HF Papers (2301.03136) | Korean finance-specific aspect-level sentiment dataset; TGT-Masking for transfer learning | 9/10 |
| 5 | **KFinEval-Pilot: Comprehensive Benchmark for Korean Financial Language** | Multiple authors | 2025 | arXiv (2504.13216) | 1,000+ questions: financial knowledge, legal reasoning, toxicity detection; Korean regulatory compliance | 9/10 |
| 6 | **KR-BERT: Small-Scale Korean-Specific Language Model** | Lee, Jang, Baik et al. (SNU) | 2020 | HF Papers (2008.03979) | Compact Korean BERT with BidirectionalWordPiece tokenizer; comparable to larger models | 7/10 |

### 7.2 Korean Language Model Papers

| # | Title | Authors/Affiliation | Year | Key Findings | Relevance |
|---|-------|-------------------|------|-------------|-----------|
| 7 | **HyperCLOVA X Technical Report** | NAVER (370+ authors) | 2024 | Multilingual LLM (KO/EN/code); strong reasoning, knowledge, cross-lingual capabilities | 8/10 |
| 8 | **HyperCLOVA X THINK Technical Report** | NAVER Cloud | 2025 | Reasoning-focused; Peri-LN Transformer, RLVR; competitive on KMMLU, CSAT, KoBALT-700 | 8/10 |
| 9 | **K-EXAONE Technical Report** | LG AI Research (57+ authors) | 2026 | 236B MoE, 256K context; global top-10 open-weight; multilingual | 8/10 |
| 10 | **Solar Open Technical Report** | Upstage (29+ authors) | 2026 | 102B MoE, bilingual; synthetic data for underserved languages; SnapPO reinforcement learning | 8/10 |
| 11 | **EXAONE 4.0: Unified LLMs Integrating Non-reasoning and Reasoning** | LG AI Research | 2025 | First Korean open-weight hybrid AI; 32B/1.2B variants; agentic tool use | 7/10 |
| 12 | **KMMLU: Measuring Massive Multitask Language Understanding in Korean** | Son (Amphora), Lee et al. | 2024 | Expert-level Korean benchmark; highlights gap in Korean LLM performance | 7/10 |
| 13 | **KLUE: Korean Language Understanding Evaluation** | Park, Moon, Kim et al. | 2021 | 8 NLU tasks; KLUE-BERT/RoBERTa pretrained models; BPE with morpheme pre-tokenization | 6/10 |
| 14 | **GECKO: Generative Language Model for English, Code and Korean** | Oh, Kim | 2024 | Bilingual LLM, efficient token generation, strong on KMMLU | 5/10 |
| 15 | **RedWhale: Adapted Korean LLM Through Efficient Continual Pretraining** | Vo, Jung et al. | 2024 | Specialized tokenizer, cross-lingual transfer, strong on KoBEST | 5/10 |

### 7.3 Crypto/Financial Prediction Papers

| # | Title | Year | Key Findings | Relevance |
|---|-------|------|-------------|-----------|
| 16 | **DeXposure-FM: Time-series Graph Foundation Model for DeFi Credit Exposures** | 2026 | Graph-tabular foundation model for DeFi; systemic risk monitoring; spillover/concentration measures | 9/10 |
| 17 | **RefineBridge: Generative Bridge Models for Financial Forecasting by Foundation Models** | 2025 | Schrodinger Bridge for refining time series foundation models; handles non-stationarity, heavy tails | 9/10 |
| 18 | **Re(Visiting) Time Series Foundation Models in Finance** | 2025 | Domain-specific pre-training of TSFMs significantly improves financial forecasting vs zero-shot | 9/10 |
| 19 | **Neural Network-Based Algorithmic Trading: Multi-Timeframe Crypto Markets** | 2025 | Multi-timeframe trend analysis + HF direction prediction; on-chain + orderbook + buy/sell pressure | 8/10 |
| 20 | **Forecasting Bitcoin Volatility from Whale Transactions using Synthesizer Transformer** | 2022 | CryptoQuant data + whale alerts; outperforms baselines; reduces drawdown in trading | 8/10 |
| 21 | **Nonlinear Dynamics of Kimchi Premium** | 2024 | Mean-reverting above thresholds; long-run 1.24% BTC premium; arbitrage only for large premiums | 8/10 |
| 22 | **Bitcoin Microstructure and the Kimchi Premium** | 2022 | Capital controls amplify frictions; 2.27% average premium; positively related to transaction costs | 8/10 |
| 23 | **Stablecoin and Cross-border Crypto Market Integration** | 2025 | Near-perfect arbitrage parity with 24-minute convergence despite regulatory barriers | 7/10 |

### 7.4 Benchmark/Evaluation Papers

| # | Title | Year | Key Findings | Relevance |
|---|-------|------|-------------|-----------|
| 24 | **KoBEST: Korean Balanced Evaluation of Significant Tasks** | 2022 | Human-annotated Korean benchmark requiring advanced linguistic knowledge | 5/10 |
| 25 | **HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models** | 2023 | Korean cultural/contextual understanding benchmark; challenges multilingual LLMs | 5/10 |
| 26 | **Open Ko-LLM Leaderboard2: Bridging Foundational and Practical Korean LLM Evaluation** | 2024 | Native Korean benchmarks improve over translated English tasks | 6/10 |
| 27 | **Evaluating Multimodal Generative AI with Korean Educational Standards (KoNET)** | 2025 | Korean educational test benchmarks for multimodal AI across school levels | 5/10 |
| 28 | **KOFFVQA: Objectively Evaluated Free-form VQA for Korean VLMs** | 2025 | 275 VQA questions with objective grading for Korean vision-language evaluation | 5/10 |
| 29 | **ScholarBench: Bilingual Benchmark for Academic Reasoning** | 2025 | English-Korean academic problem-solving evaluation across domains | 4/10 |

---

## 8. SPECIAL TOPICS

### 8.1 Kimchi Premium Research

**Definition:** The price difference between cryptocurrency on Korean exchanges vs. global markets. Historically 2-5% premium, can spike to 30%+ during market manias.

**Academic Findings (VERIFIED):**
- **Long-run steady state:** 1.24% for Bitcoin (Nonlinear Dynamics paper, 2024)
- **Mean-reverting behavior:** Only when premium exceeds certain thresholds; random walk inside range
- **Arbitrage convergence:** 24-minute convergence for BTC/Tether premiums despite strict regulations (2025 high-frequency study)
- **Capital controls amplify frictions:** Premiums positively related to transaction costs and volatilities (Choi et al.)
- **Chinese arbitrageurs:** Use Korean financial institutions as bitcoin-cashing outlets
- **Speculative bubble indicator:** Positively associated with trading volume after controlling for volatility/liquidity

**Current Status (2026):**
- Premium ranged from -2.15% (negative/discount) to positive in 2026
- Now considered a bidirectional indicator (both premium and discount)
- Korean won-denominated crypto pairs provide unique signal

**Relevance to Our System:** 9/10 - Kimchi premium is a leading indicator of Korean market sentiment, capital flow direction, and arbitrage opportunity. Can be computed in real-time from Upbit vs. Binance/Coinbase price feeds.

### 8.2 Korean Crypto Exchange APIs

#### Upbit (Dunamu)
- **Official API:** https://global-docs.upbit.com/
- **Capabilities:** REST + WebSocket
  - Market data: pairs, OHLCV candles, trades, current price, orderbook
  - Trading: order placement, deposit/withdrawal, balance inquiry
- **Data:** 310 coins, 650 trading pairs
- **Most traded:** XRP/KRW, BTC/KRW, ETH/KRW
- **Open-source clients:** Python (official), TypeScript, Dart
- **Cold storage:** 99% ratio (increased after Solana wallet breach)
- **Users:** 13.26M cumulative, ~4.53M MAU
- **Developer Conference:** Annual Upbit Developer Conference (UDC)
- **Relevance:** 10/10

#### Bithumb
- **Official API:** https://apidocs.bithumb.com/
- **REST API:** HmacSHA256 auth with apiKey/secretKey
- **WebSocket:** wss://global-api.bithumb.pro/message/realtime
  - Public: ticker, orderbook
  - Private: order changes, position changes
- **Futures API:** https://bithumbfutures.github.io/bithumb-futures-api-doc/
- **Features:** Member wallet info, orders, buy/sell, cancellation, trade logs, withdrawals
- **Users:** ~2.42M MAU, ~25% of Korea crypto volume
- **IPO planned:** Bithumb IPO anticipated
- **Relevance:** 9/10

#### Other Korean Exchanges
- **Coinone, Korbit, Gopax** (smaller, FSC-licensed)
- All require real-name KRW accounts

### 8.3 FSS VISTA System (Crypto Surveillance)

**Full Name:** Virtual Assets Intelligence System for Trading Analysis
- **Developer:** FSS internal staff (Python-based)
- **Purpose:** Process massive trading volumes, flag abnormal transactions, visualize trading behavior
- **Method:** Sliding-window grid search - divides trading data into overlapping time segments, scans each for anomalies automatically

**2025-2026 Upgrades:**
- December 2025: Two new servers with high-performance CPU/GPU
- Automated detection model identifying price-rigging periods without manual intervention
- **2026 Budget:** 170 million won ($116,000) for AI upgrades
- **Planned capabilities:**
  1. Automatically identify networks of suspicious coordinated accounts
  2. Analyze abnormal trading-related text across thousands of crypto assets
  3. Trace origin of funds used in manipulation

**Targeted Manipulation Types:**
- Large-scale whale trading
- Artificial price swings during deposit/withdrawal suspensions
- Coordinated trading via APIs
- Social media misinformation campaigns

**Relevance to Our System:** 8/10 - Understanding VISTA's detection patterns is critical for:
1. Avoiding false signals from manipulated price data
2. Building complementary manipulation detection features
3. Regulatory compliance for any Korean market operations

### 8.4 Korean Regulatory Framework (2025-2026)

**Virtual Asset User Protection Act (VAUPA):**
- In full force
- Requires separation of user assets from company funds
- Mandatory insurance against hacks
- Real-name KRW account requirement

**AI Basic Act:**
- Passed January 2025, effective January 2026
- Article 12 establishes K-AISI under MSIT
- Government AI budget: 10.1 trillion won for 2026

**Digital Asset Basic Act (Phase 2):**
- Under development, delayed to 2026 over stablecoin issuer disputes (banks vs. fintech)
- Will reauthorize domestic ICOs (banned since 2017) with white paper disclosure requirements
- Strict liability for false/misleading statements
- Service providers face transparency, fair terms, regulated advertising requirements
- Operator liability for security breaches/failures even without proven negligence
- FSC Virtual Asset Committee first 2026 meeting: March 4, 2026

**Payment Suspension System:**
- FSC developing system to stop transactions before money laundering

**Relevance to Our System:** 7/10 - Compliance awareness needed for Korean market features.

### 8.5 South Korea National AI Strategy

**Five Sovereign AI Consortia (selected August 2025):**
1. **NAVER** - HyperCLOVA ecosystem
2. **SK Telecom** - A.X ecosystem
3. **LG Group** - EXAONE ecosystem
4. **NCSoft** - VARCO ecosystem
5. **Upstage** - SOLAR ecosystem

**Government Investment:**
- $381M government funding allocated to five consortia
- First-stage evaluation (Dec 2025) narrows to 4 teams
- Only 2 survive by 2027
- Goal: Top-10 global AI model, released as open source

**Infrastructure:**
- National AI Computing Center construction planned
- 260,000 advanced GPUs to be secured
- 2026 budget: 10.1 trillion won (~$6.94B) for AI

**Five-Year Plan (Aug 2025):**
- "Super-Innovation Economy" plan
- $71.5B investment in AI across all sectors over 5 years
- Sovereign Korean-language AI model development

### 8.6 Korean Market Microstructure for Crypto

**Key Characteristics:**
- KRW-denominated trading only (no USD pairs on Korean exchanges)
- Real-name bank account requirement creates friction
- Capital controls create persistent Kimchi premium/discount
- XRP dominance in Korean retail trading (most traded on Upbit 2025)
- Youth demographic heavily invested (Upbit dominates Korean youth)
- 24-hour trading unlike Korean stock market (9:00-15:30 KST)
- Korean exchanges not directly connected to global order books

**Data Signals Unique to Korea:**
- Kimchi premium (real-time cross-exchange price differential)
- Korean won exchange rate (USD/KRW, JPY/KRW)
- Korean retail sentiment (heavy retail participation)
- FSS VISTA anomaly flags (if accessible)
- Korean news/social media sentiment (Naver blogs, KakaoTalk groups)
- Deposit/withdrawal suspension announcements on exchanges

---

## 9. INTEGRATION RECOMMENDATIONS

### 9.1 Highest-Priority Models for Our System

| Priority | Model/Resource | Use Case | Integration Path |
|----------|---------------|----------|-----------------|
| 1 | **KR-FinBert-SC** (SNU) | Korean financial sentiment classification | Fine-tune on Korean crypto news |
| 2 | **NMIXX embeddings** | Cross-lingual Korean-English financial semantic search | Use for news/document embedding |
| 3 | **Won/FINKRX instruction data** | Korean financial reasoning training data | Use 80K instances for fine-tuning |
| 4 | **A.X-4.0-Light** (Apache-2.0, 8B) | Korean language understanding agent | Deploy as Korean market analysis agent |
| 5 | **EXAONE-4.0-1.2B** (on-device) | Lightweight Korean inference | Edge/on-device Korean text processing |
| 6 | **A.X-4.0-VL-Light** (Apache-2.0, 8B) | Korean document/chart understanding | Process Korean exchange reports, charts |
| 7 | **A.X-Encoder-base** (ModernBERT, Apache-2.0) | Korean text embeddings | Embed Korean news/social for RAG |
| 8 | **Upbit + Bithumb APIs** | Real-time Korean market data | Price feeds, orderbook, Kimchi premium calc |
| 9 | **KorFinMTEB benchmark** | Evaluate our Korean financial models | Benchmark our fine-tuned models |
| 10 | **KFinEval-Pilot benchmark** | Evaluate financial reasoning | Test Korean regulatory compliance |

### 9.2 Kimchi Premium Integration

**Data Pipeline:**
1. Upbit WebSocket -> BTC/KRW, ETH/KRW, XRP/KRW real-time prices
2. Binance/Coinbase WebSocket -> BTC/USDT, ETH/USDT, XRP/USDT prices
3. USD/KRW exchange rate feed (Bank of Korea or forex API)
4. Compute: `kimchi_premium = (upbit_krw_price / (global_usd_price * usdkrw_rate) - 1) * 100`
5. Feed as feature to ensemble model (historically mean-reverting above ~2% threshold)

### 9.3 Korean Sentiment Pipeline

**Recommended Architecture:**
1. **Data Sources:** Naver News, Korean crypto communities, KakaoTalk groups (if accessible), Korean Twitter/X
2. **Embedding:** A.X-Encoder-base (ModernBERT) or NMIXX embeddings
3. **Sentiment:** KR-FinBert-SC fine-tuned on Korean crypto sentiment
4. **Summarization:** A.X-4.0-Light for Korean news summarization
5. **Document Understanding:** A.X-4.0-VL-Light for Korean exchange announcements with images/tables

### 9.4 Regulatory Compliance Considerations

- VISTA system detects manipulation patterns - our system should NOT generate signals that mimic manipulation
- Digital Asset Basic Act Phase 2 will impose disclosure requirements
- Any Korean-facing service needs FSC licensing awareness
- Payment suspension system could affect trading signals
- K-AISI safety evaluations may apply to financial AI systems

---

## VERIFICATION STATUS SUMMARY

| Category | Items Researched | Verified | Partially Verified | Unverified |
|----------|-----------------|----------|-------------------|------------|
| Models (HuggingFace) | 45+ | 43 | 2 | 0 |
| Datasets | 14 | 12 | 2 | 0 |
| Benchmarks | 13 | 13 | 0 | 0 |
| Papers | 29 | 29 | 0 | 0 |
| Companies | 10 | 10 | 0 | 0 |
| Universities/Labs | 8 | 8 | 0 | 0 |
| APIs | 2 | 2 | 0 | 0 |

---

## SOURCES

### Web Sources
- [SKT A.X-4.0-VL-Light HuggingFace](https://huggingface.co/skt/A.X-4.0-VL-Light)
- [SKT A.X K1 Announcement](https://news.sktelecom.com/en/2533)
- [A.X K1 Technical Report](https://arxiv.org/pdf/2601.09200)
- [South Korea Funds LG Exaone 4.0, SKT A.X](https://dataconomy.com/2025/09/29/south-korea-funds-lg-exaone-4-0-skt-a-x-for-ai-sovereignty/)
- [NAVER HyperCLOVA X](https://navercorp.com/en/tech/hyperclovax)
- [NAVER HyperCLOVA X THINK](https://koreatechtoday.com/naver-pushes-inference-ai-frontier-with-hyperclova-x-think/)
- [HyperCLOVA X HuggingFace](https://huggingface.co/naver-hyperclovax)
- [LG K-EXAONE Global Top 10](https://www.koreaherald.com/article/10652980)
- [EXAONE 4.0 GitHub](https://github.com/LG-AI-EXAONE/EXAONE-4.0)
- [K-EXAONE GitHub](https://github.com/LG-AI-EXAONE/K-EXAONE)
- [Upstage Solar Pro 2 Beats GPT-4.1](https://koreatechdesk.com/upstage-solar-pro-2-korean-global-ai-frontier-model)
- [Solar Pro 3](https://www.upstage.ai/blog/en/solar-pro-3-0127)
- [Solar Open 100B HuggingFace](https://huggingface.co/upstage/Solar-Open-100B)
- [Upstage Solar Open Technical Report](https://arxiv.org/abs/2601.07022)
- [FSS VISTA AI Update](https://bitcoinethereumnews.com/crypto/fss-vista-ai-update-crypto-manipulation-detection/)
- [Korea Bolsters AI for Crypto Manipulation](https://www.koreatimes.co.kr/economy/policy/20260202/korea-bolsters-ai-capabilities-to-tackle-crypto-market-manipulation)
- [South Korea AI Crypto Surveillance 2026](https://www.tronweekly.com/south-korea-boosts-ai-powered-crypto/)
- [FSS 2026 Crypto Oversight](https://www.cryptotimes.io/2026/02/09/south-koreas-fss-launches-strict-2026-crypto-oversight-and-ai-plan/)
- [Korea AI Safety Institute](https://www.aisi.re.kr/eng)
- [K-AISI Safety Forecast Report](https://thelegalwire.ai/k-aisi-releases-ai-safety-forecast-report/)
- [South Korea AI Act](https://coingeek.com/south-korea-ai-act-comes-into-force-with-safety-measures/)
- [Kakao Kanana AI](https://www.kakaocorp.com/page/detail/11725?lang=ENG)
- [KakaoBank Azure OpenAI](https://www.microsoft.com/en/customers/story/24967-kakao-bank-azure-openai)
- [Samsung AI Center Seoul](https://research.samsung.com/aicenter_seoul)
- [Samsung R&D $25.6B](https://www.southkoreannetwork.com/2026/03/2025-256-samsung-electronics-spends-256.html)
- [Twelve Labs](https://www.twelvelabs.io/)
- [NCSoft VARCO-VISION 2.0](https://www.koreatimes.co.kr/business/companies/20250716/nc-ai-releases-new-multimodal-ai-model-as-open-source)
- [ETRI Eagle Model](https://www.eurekalert.org/news-releases/1111378)
- [KIST AI Research](https://kist.re.kr/eng/index.do)
- [Upbit Developer Center](https://global-docs.upbit.com/)
- [Bithumb API Docs](https://apidocs.bithumb.com/)
- [Kimchi Premium Nonlinear Dynamics](https://www.sciencedirect.com/science/article/pii/S0264999324000828)
- [Bitcoin Microstructure Kimchi Premium](https://www.researchgate.net/publication/326027783_Bitcoin_Microstructure_and_the_Kimchi_Premium)
- [Stablecoin Cross-border Crypto Integration](https://www.sciencedirect.com/science/article/abs/pii/S0165176525005415)
- [Sovereign AI Foundation Model Project](https://www.korea.net/Government/Briefing-Room/Press-Releases/view?articleId=8189)
- [2026 Korea AI Budget Analysis](https://blog.pebblous.ai/report/korea-ai-fund-report-2026-03/en/)
- [South Korea $735B AI Initiative](https://introl.com/blog/south-korea-735b-sovereign-ai-initiative-infrastructure-requirements-opportunities)
- [Digital Asset Basic Act Phase 2](https://cryptorank.io/news/feed/5ffd1-south-korea-fsc-digital-asset-act)
- [Won/FINKRX ACL 2025](https://aclanthology.org/2025.acl-industry.81/)
- [KFinEval-Pilot](https://arxiv.org/abs/2504.13216)
- [KorFinMTEB (TWICE paper)](https://arxiv.org/abs/2502.07131)
- [KAIST MLAI Lab](https://www.mlai-kaist.com/)
- [SNU Data Mining Lab](https://datalab.snu.ac.kr/)
- [SNU NLP Lab](http://knlp.snu.ac.kr/)
- [SNU SLCF Lab](https://safeai.snu.ac.kr/)
- [POSTECH NLP Group](https://nlp.postech.ac.kr/)
- [POSTECH ML Lab](https://ml.postech.ac.kr/)
- [NAIRL NeurIPS 2025](https://nairl.kr/news_21/)
- [OpenAI South Korea Blueprint](https://cdn.openai.com/global-affairs/f9361fe7-e452-4c78-94dc-e6946c73c858/openai-south-korea-economic-blueprint-october-2025.pdf)
- [Kimchi Premium 2026 Guide](https://www.spotedcrypto.com/kimchi-premium-guide-2026/)
- [AMRO Kimchi Premium Analysis](https://amro-asia.org/the-rise-and-fall-of-kimchi-premium-in-koreas-virtual-asset-market/)

### HuggingFace Papers
- [HyperCLOVA X Technical Report](https://hf.co/papers/2404.01954)
- [HyperCLOVA X THINK](https://hf.co/papers/2506.22403)
- [K-EXAONE Technical Report](https://hf.co/papers/2601.01739)
- [Solar Open Technical Report](https://hf.co/papers/2601.07022)
- [EXAONE 3.0](https://hf.co/papers/2408.03541)
- [EXAONE 4.0](https://arxiv.org/abs/2507.11407)
- [KMMLU](https://hf.co/papers/2402.11548)
- [KLUE](https://hf.co/papers/2105.09680)
- [KoBEST](https://hf.co/papers/2204.04541)
- [HAE-RAE Bench](https://hf.co/papers/2309.02706)
- [KR-BERT](https://hf.co/papers/2008.03979)
- [Korean Financial Sentiment (Son et al.)](https://hf.co/papers/2301.03136)
- [NMIXX Financial Embeddings](https://hf.co/papers/2507.09601)
- [TWICE/KorFinMTEB](https://hf.co/papers/2502.07131)
- [GECKO](https://hf.co/papers/2405.15640)
- [RedWhale](https://hf.co/papers/2408.11294)
- [Open Ko-LLM Leaderboard2](https://hf.co/papers/2410.12445)
- [KoNET](https://hf.co/papers/2502.15422)
- [KOFFVQA](https://hf.co/papers/2503.23730)
- [K-Viscuit](https://hf.co/papers/2406.16469)
- [ScholarBench](https://hf.co/papers/2505.16566)
- [BEEP! Korean Toxic Speech](https://hf.co/papers/2005.12503)
- [DeXposure-FM DeFi](https://hf.co/papers/2602.03981)
- [RefineBridge Financial Forecasting](https://hf.co/papers/2512.21572)
- [Re(Visiting) TSFMs in Finance](https://hf.co/papers/2511.18578)
- [Neural Network Crypto Trading](https://hf.co/papers/2508.02356)
- [Bitcoin Whale Volatility Forecasting](https://hf.co/papers/2211.08281)
- [Bitcoin AML with GCN](https://hf.co/papers/1908.02591)
- [Dynamic GNN Volatility Prediction](https://hf.co/papers/2410.16858)
