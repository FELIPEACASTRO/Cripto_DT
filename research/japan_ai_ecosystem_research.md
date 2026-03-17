# Japan AI Ecosystem Research for Crypto Price Prediction
## Exhaustive Research Report - March 2026

---

# TABLE OF CONTENTS
1. [Executive Summary](#1-executive-summary)
2. [Universities & Research Labs](#2-universities--research-labs)
3. [Models on HuggingFace](#3-models-on-huggingface)
4. [Datasets](#4-datasets)
5. [Benchmarks](#5-benchmarks)
6. [Companies & Labs](#6-companies--labs)
7. [Papers](#7-papers)
8. [Special Topics](#8-special-topics)
9. [Integration Recommendations](#9-integration-recommendations)

---

# 1. EXECUTIVE SUMMARY

Japan is undergoing a massive AI transformation backed by >$135B in combined government/private investment through 2030. The crypto regulatory environment is shifting dramatically with a flat 20% tax rate effective April 2026 (down from 55%), reclassifying 105 crypto assets as financial products under the FIEA. This creates a unique opportunity window for our crypto prediction platform.

**Key findings for our system:**
- The Swallow model family (now up to 120B params) is VERIFIED as the strongest open Japanese LLM series
- NRI's financial fine-tune of Swallow beating GPT-4o by 9.6 points is VERIFIED (insurance compliance task)
- PLaMo-fin-base (Feb 2026) is a NEW dedicated Japanese financial LLM from Preferred Networks
- Sakana AI + MUFG partnership creates the most advanced Japanese financial AI in production
- Three Japanese financial benchmarks now exist: japanese-lm-fin-harness, EDINET-Bench, Ebisu
- CoinPost VERIFIED as Japan's #1 crypto media (global top-3 crypto site for 22+ months)
- Economy Watchers Survey dataset provides monthly-updated Japanese economic sentiment data since 2000

---

# 2. UNIVERSITIES & RESEARCH LABS

## 2.1 Institute of Science Tokyo (formerly Tokyo Tech) - Swallow Team
- **Lab:** Okazaki Laboratory (Prof. Naoaki Okazaki)
- **Key Project:** Swallow LLM series - Japan's most important open LLM family
- **Latest Models:** GPT-OSS-Swallow-120B-RL-v0.1 (Feb 2026), Qwen3-Swallow-8B/32B (Jan 2026)
- **Infrastructure:** Uses ABCI 3.0 and GENIAC compute resources
- **Key Papers:** arXiv:2404.17733 (Swallow corpus), arXiv:2404.17790 (continual pre-training), arXiv:2505.09388, arXiv:2412.02595
- **Collaborators:** AIST, NRI, Tohoku University, RIKEN
- **Relevance to crypto prediction:** 10/10 - Primary source of Japanese-capable LLMs for our NLP pipeline
- **Verification:** CONFIRMED via HuggingFace repos and publications

## 2.2 University of Tokyo (Todai)
- **Harada Lab (RCAST):** Machine Intelligence - computer vision, reinforcement learning, robotics
  - Created DEJIMA dataset (3.88M Japanese image-text pairs)
  - Key for multimodal document understanding of Japanese financial reports
- **NLP Group (Miyao Lab):** Syntactic parsing, semantic analysis, dialog systems
- **Relevance to crypto prediction:** 7/10 - DEJIMA dataset useful for Japanese document understanding
- **Verification:** CONFIRMED via lab websites and DEJIMA paper (arXiv:2512.00773)

## 2.3 RIKEN Center for Advanced Intelligence Project (AIP)
- **Director:** Masashi Sugiyama (leading ML researcher)
- **Established:** April 2016
- **Focus:** Fundamental AI, scientific acceleration, societal problem-solving
- **Key Labs:** Machine Learning, Optimization, Statistics, NLP, Image Processing
- **Connection to Osaka U:** Yoshinobu Kawahara (Osaka U professor) is RIKEN AIP team director for Dynamical Systems Learning
- **FugakuNEXT:** RIKEN partnering with Fujitsu + NVIDIA for next-gen zettascale supercomputer (~2030)
- **Relevance to crypto prediction:** 6/10 - Foundational ML research, time series methods
- **Verification:** CONFIRMED via riken.jp

## 2.4 AIST (National Institute of Advanced Industrial Science and Technology)
- **Key Infrastructure:** ABCI 3.0 supercomputer - 6128 NVIDIA H200 GPUs, 6.22 exaflops (half precision)
- **Operational:** January 2025
- **Role:** Provides compute for Swallow training, GENIAC projects, and multimodal AI development
- **Collaboration:** Joint development of Swallow models with Institute of Science Tokyo
- **Relevance to crypto prediction:** 7/10 - Infrastructure partner, co-developer of Swallow
- **Verification:** CONFIRMED via NVIDIA blog and AIST website

## 2.5 Tohoku University - NLP Lab
- **Lab:** Prof. Kentaro Inui's Natural Language Processing group
- **Key Models:** Japanese BERT v2/v3 (tohoku-nlp/bert-base-japanese-v3)
  - 12 layers, 768 hidden dimensions, 12 attention heads
  - Trained on CC-100 Japanese + Japanese Wikipedia
- **Center:** Center for Language AI Research (CLaiR)
- **Relevance to crypto prediction:** 8/10 - Japanese BERT models are foundational for sentiment analysis
- **Verification:** CONFIRMED via HuggingFace (tohoku-nlp org) and GitHub (cl-tohoku)

## 2.6 NII (National Institute of Informatics) - LLM-jp Project
- **Project:** LLM-jp - collaborative open Japanese LLM initiative
- **Participants:** 1,000+ researchers/engineers from universities and corporations
- **Center:** Research and Development Center for Large Language Models (LLMC), established April 2024
- **Latest Model:** llm-jp-3-172b-instruct3 (172B params, GPT-3 scale, trained on 2.1T tokens)
- **Funded by:** Ministry of Education (MEXT) - transparency and reliability of generative AI
- **Open Leaderboard:** Open Japanese LLM Leaderboard on HuggingFace (llm-jp space)
- **Relevance to crypto prediction:** 7/10 - Fully open Japanese LLM, can be fine-tuned for financial tasks
- **Verification:** CONFIRMED via nii.ac.jp and HuggingFace

## 2.7 Kyoto University - Language Media Processing Lab
- **Key Tools:** Juman++ (morphological analyzer), KWJA (integrated Japanese analyzer using foundation models)
- **Corpora:** KyotoCorpus, Kyoto University Web Document Leads Corpus
- **Research Focus:** Predicate-argument structures, text understanding, machine translation
- **HuggingFace:** ku-nlp organization
- **Relevance to crypto prediction:** 6/10 - Japanese text analysis tools for preprocessing
- **Verification:** CONFIRMED via lab website and GitHub (ku-nlp)

## 2.8 Osaka University
- **AI Research Center (SANKEN):** Artificial Intelligence Research Center
- **Key Researcher:** Yoshinobu Kawahara - dual appointment as RIKEN AIP team director
- **Focus:** Dynamical systems learning (relevant to time series), medical AI
- **Relevance to crypto prediction:** 5/10 - Dynamical systems research applicable to market modeling
- **Verification:** CONFIRMED via osaka-u.ac.jp

## 2.9 NAIST (Nara Institute of Science and Technology)
- **Lab:** NLP Laboratory (Prof. Taro Watanabe)
- **2025 Research:** Vision-language models, machine translation, grammatical error correction
- **Conference Papers:** COLING 2025, ACL 2025 (student workshop)
- **Relevance to crypto prediction:** 4/10 - General NLP research, less finance-specific
- **Verification:** CONFIRMED via nlp.naist.jp

---

# 3. MODELS ON HUGGINGFACE

## 3.1 Swallow Family (tokyotech-llm) - HIGHEST PRIORITY

### Generation 1: Llama 2 Base (Nov-Dec 2023)
| Model | Params | License | Downloads |
|-------|--------|---------|-----------|
| Swallow-7b-hf | 7B | Llama 2 | 294 |
| Swallow-7b-instruct-hf | 7B | Llama 2 | 309 |
| Swallow-13b-hf | 13B | Llama 2 | 18 |
| Swallow-13b-instruct-hf | 13B | Llama 2 | 21 |
| Swallow-70b-hf | 70B | Llama 2 | 19 |
| Swallow-70b-instruct-hf | 70B | Llama 2 | 902 |
- NVE (No Vocabulary Expansion) variants also available
- Relevance: 8/10 | Verification: CONFIRMED

### Generation 2: Mistral/Mixtral Base (Feb 2024)
| Model | Params | License |
|-------|--------|---------|
| Swallow-MS-7b-v0.1 | 7B | Apache-2.0 |
| Swallow-MX-8x7b-NVE-v0.1 | 8x7B MoE | Apache-2.0 |
- Relevance: 7/10 | Verification: CONFIRMED

### Generation 3: Qwen3 Base (Jan 2026) - LATEST
| Model | Params | License | Downloads |
|-------|--------|---------|-----------|
| Qwen3-Swallow-8B-RL-v0.2 | 8B | Apache-2.0 | 4,200 |
| Qwen3-Swallow-32B-RL-v0.2-AWQ-INT4 | 32B (4-bit) | Apache-2.0 | 1,100 |
- Trained on: swallow-math-v2, swallow-code-v2, Swallow-Nemotron-Post-Training-Dataset-v1
- Relevance: 10/10 | Verification: CONFIRMED

### Generation 4: GPT-OSS Base (Feb 2026) - NEWEST
| Model | Params | License | Downloads |
|-------|--------|---------|-----------|
| GPT-OSS-Swallow-20B-SFT-v0.1 | 20B | Apache-2.0 | 1,400 |
| GPT-OSS-Swallow-20B-RL-v0.1 | 20B | Apache-2.0 | 7,700 |
| GPT-OSS-Swallow-120B-SFT-v0.1 | 120B | Apache-2.0 | 1,900 |
| GPT-OSS-Swallow-120B-RL-v0.1 | 120B | Apache-2.0 | 3,800 |
- Highest Japanese task score (0.642) among open models <=120B
- Based on OpenAI's GPT-OSS open-weight models (arXiv:2508.10925)
- Relevance: 10/10 | Verification: CONFIRMED

## 3.2 PFN Financial Models (pfnet)

| Model | Params | Type | Relevance |
|-------|--------|------|-----------|
| pfnet/nekomata-14b-pfn-qfin | 14B | Financial continual pretrained | 9/10 |
| pfnet/nekomata-14b-pfn-qfin-inst-merge | 14B | Financial instruction-tuned (merged) | 9/10 |
- Method: Continual pretraining on financial data + instruction vector merging
- Benchmark: State-of-the-art on japanese-lm-fin-harness
- Paper: arXiv:2404.10555, arXiv:2409.19854
- **PLaMo-fin-base** (Feb 2026): New derivative for financial institutions (not yet on HF, enterprise-only)
- Verification: CONFIRMED

## 3.3 CyberAgent Models

| Model | Params | License | Downloads |
|-------|--------|---------|-----------|
| OpenCALM series (160M-6.8B) | Various | CC BY-SA 4.0 | Various |
| CyberAgentLM3-22B-Chat | 22.5B | Apache-2.0 | - |
| Llama-3.1-70B-Japanese-Instruct-2407 | 70B | Llama 3.1 | 177 |
| Mistral-Nemo-Japanese-Instruct-2408 | 12B | Apache-2.0 | 550 |
| DeepSeek-R1-Distill-Qwen-14B-Japanese | 14B | MIT | 490 |
| DeepSeek-R1-Distill-Qwen-32B-Japanese | 32B | MIT | 353 |
- Latest: DeepSeek R1 distillations (Jan 2025) with strong reasoning
- Relevance: 7/10 | Verification: CONFIRMED

## 3.4 Rinna Models

| Model | Params | License | Downloads |
|-------|--------|---------|-----------|
| japanese-gpt2-small/medium | 0.1-0.3B | MIT | 2.9K-10.8K |
| japanese-gpt-neox-small | 0.2B | MIT | 507K |
| japanese-gpt-neox-3.6b | 3.6B | MIT | Various |
| japanese-gpt-neox-3.6b-instruction-ppo | 3.6B | MIT | 733 |
| japanese-gpt-1b | 1B | MIT | 1.6K |
| japanese-roberta-base | 125M | MIT | 8.7K |
| japanese-clip-vit-b-16 | ViT-B/16 | Apache-2.0 | 45.8K |
| japanese-cloob-vit-b-16 | ViT-B/16 | Apache-2.0 | 7.2K |
| japanese-stable-diffusion | - | CreativeML | - |
- Relevance: 6/10 (CLIP models useful for multimodal) | Verification: CONFIRMED

## 3.5 Stability AI Japan Models

| Model | Params | Architecture |
|-------|--------|-------------|
| japanese-stablelm-instruct-alpha-7b | 7B | StableLM |
| japanese-stablelm-base-beta-7b | 7B | StableLM |
| japanese-stablelm-base-gamma-7b | 7B | StableLM |
| japanese-stablelm-2-base-1_6b | 1.6B | StableLM 2 |
| japanese-stablelm-2-instruct-1_6b | 1.6B | StableLM 2 |
- Note: Stability AI had financial difficulties; model development may be slower
- Relevance: 5/10 | Verification: CONFIRMED

## 3.6 Tohoku NLP BERT Models

| Model | Type | Downloads |
|-------|------|-----------|
| tohoku-nlp/bert-base-japanese-v3 | Encoder (fill-mask) | High |
| tohoku-nlp/bert-base-japanese-v2 | Encoder (fill-mask) | High |
- Trained on CC-100 Japanese + Wikipedia
- Critical for sentiment classification pipelines
- Relevance: 8/10 | Verification: CONFIRMED

## 3.7 SB Intuitions (SoftBank) - Sarashina

| Model | Params | Status |
|-------|--------|--------|
| sbintuitions/sarashina2.1-1b | 1B | Available |
| Sarashina (full) | 460B | Internal/enterprise |
| Sarashina Mini | Lightweight | Commercialized Nov 2025 |
- Relevance: 5/10 (enterprise-focused) | Verification: CONFIRMED

## 3.8 LLM-jp (NII)

| Model | Params | License |
|-------|--------|---------|
| llm-jp-3-172b-instruct3 | 172B | Open |
| Dense-btx-japanese-expert-1.5B | 1.5B | Apache-2.0 |
- Fully open, trained from scratch on Japanese data
- Relevance: 6/10 | Verification: CONFIRMED

## 3.9 ABEJA Models

| Model | Base | Purpose |
|-------|------|---------|
| ABEJA-V2 (32B Qwen 2.5) | Qwen 2.5 | Japanese fine-tune |
| ABEJA Qwen2.5-7B-Japanese | Qwen 2.5 7B | Distillation learning |
- Relevance: 5/10 | Verification: CONFIRMED via model registries

---

# 4. DATASETS

## 4.1 Japanese Financial Datasets - HIGH PRIORITY

### Economy Watchers Survey (retarfi/economy-watchers-survey)
- **Source:** Cabinet Office of Japan, monthly since January 2000
- **Content:** Regional economic assessments from workers in retail, services, manufacturing, construction, real estate
- **Tasks:** 3-class classification, 12-class classification, 5-class sentiment analysis
- **Availability:** HuggingFace (retarfi/economy-watchers-survey), auto-updated monthly
- **License:** Original government data
- **Relevance to crypto:** 9/10 - Direct Japanese economic sentiment indicator, monthly cadence
- **Verification:** CONFIRMED

### EDINET-Bench (SakanaAI/EDINET-Bench)
- **Source:** FSA's EDINET system - 10 years of annual reports from Japanese listed companies
- **Tasks:** Accounting fraud detection, earnings forecasting, industry prediction
- **Accepted:** ICLR 2026
- **Tools:** edinet2dataset (GitHub) for custom dataset construction
- **Relevance to crypto:** 8/10 - Japanese corporate financial analysis
- **Verification:** CONFIRMED

### Ebisu Benchmark Dataset
- **Source:** Expert-annotated Japanese financial language understanding
- **Tasks:** JF-ICR (implicit commitment recognition in investor Q&A), JF-TE (hierarchical financial terminology extraction)
- **Published:** February 2026
- **Relevance to crypto:** 7/10 - Financial language understanding evaluation
- **Verification:** CONFIRMED (arXiv:2602.01479)

### japanese-lm-fin-harness Tasks
- **Source:** pfnet-research (Preferred Networks)
- **Tasks:** Securities analyst exam questions, CPA audit theory, binary sentiment analysis
- **License:** CC-BY 4.0 (CPA data)
- **Relevance to crypto:** 8/10 - Standard Japanese financial evaluation
- **Verification:** CONFIRMED

## 4.2 Japanese Vision-Language Datasets

### DEJIMA (arXiv:2512.00773)
- **Institution:** University of Tokyo (Harada Lab, Machine Intelligence Lab)
- **Size:** 3.88M image-text pairs (DEJIMA-Cap + DEJIMA-VQA)
- **Scale:** 23.7x STAIR Captions, 39.1x Japanese Visual Genome
- **Method:** Web collection + object detection + LLM refinement
- **Quality:** Higher Japaneseness and naturalness than translation-based datasets
- **Website:** mil-tokyo.github.io/DEJIMA-dataset/
- **Relevance to crypto:** 6/10 - Japanese multimodal understanding for chart/report analysis
- **Verification:** CONFIRMED

### WAON (arXiv:2510.22276)
- **Content:** Large-scale Japanese image-text pairs from Common Crawl
- **Benchmark:** WAON-Bench for Japanese cultural image classification
- **Relevance to crypto:** 5/10
- **Verification:** CONFIRMED

## 4.3 General Japanese NLP Datasets

### JGLUE (Japanese General Language Understanding Evaluation)
- **Tasks:** MARC-ja (sentiment), JCommonsenseQA, JSTS (textual similarity), JNLI (inference), JSQuAD (QA)
- **Status:** Standard Japanese LLM evaluation suite
- **Relevance to crypto:** 6/10 (general capability benchmark)
- **Verification:** CONFIRMED

### Swallow Corpus v3.2
- **Source:** Large-scale Japanese web text from Common Crawl
- **Usage:** Training data for all Swallow models
- **Paper:** arXiv:2404.17733 ("Building a Large Japanese Web Corpus for LLMs")
- **Relevance to crypto:** 7/10 (foundation for Japanese LLM training)
- **Verification:** CONFIRMED

### livedoor News Corpus
- **Content:** Japanese news articles across multiple categories
- **Usage:** Standard Japanese text classification benchmark
- **Relevance to crypto:** 5/10
- **Verification:** CONFIRMED (well-known dataset)

### LLM-jp Japanese Image-Text Pairs
- **Source:** NII GitLab (gitlab.llm-jp.nii.ac.jp)
- **Relevance to crypto:** 4/10
- **Verification:** CONFIRMED

---

# 5. BENCHMARKS

## 5.1 Japanese Financial Benchmarks

### japanese-lm-fin-harness (PFN)
- **Maintainer:** Preferred Networks (pfnet-research)
- **Tasks:** Securities analyst exam, CPA audit theory, financial sentiment analysis
- **Format:** 0-shot evaluation, standardized prompts
- **GitHub:** github.com/pfnet-research/japanese-lm-fin-harness
- **Relevance:** 10/10 - Primary Japanese financial LLM benchmark
- **Verification:** CONFIRMED

### EDINET-Bench (Sakana AI) - ICLR 2026
- **Tasks:** Fraud detection, earnings forecasting, industry prediction
- **Data:** 10 years of EDINET filings
- **Finding:** SOTA models only slightly better than logistic regression on binary classification
- **HuggingFace:** SakanaAI/EDINET-Bench
- **GitHub:** SakanaAI/EDINET-Bench, SakanaAI/edinet2dataset
- **Relevance:** 9/10 - Expert-level Japanese financial evaluation
- **Verification:** CONFIRMED

### Ebisu (Feb 2026)
- **Tasks:** JF-ICR (implicit commitment recognition), JF-TE (financial terminology extraction)
- **Models Evaluated:** 22 LLMs (open-source and proprietary)
- **Finding:** Even SOTA systems struggle; scaling and adaptation do not reliably help
- **Relevance:** 8/10 - Tests nuanced Japanese financial language understanding
- **Verification:** CONFIRMED (arXiv:2602.01479)

### pfmt-bench-fin-ja (PFN)
- **Type:** Japanese financial MT-Bench variant
- **GitHub:** pfnet-research/pfmt-bench-fin-ja
- **Relevance:** 8/10
- **Verification:** CONFIRMED

## 5.2 General Japanese LLM Benchmarks

### Nejumi LLM Leaderboard 4 (Weights & Biases Japan)
- **Scope:** 45+ models evaluated
- **Categories:** Reasoning (ARC-AGI, ARC-AGI-2), knowledge (JMMLU-Pro, Humanity's Last Exam), coding (SWE-Bench, JHumanEval, MT-Bench Coding), tool use (BFCL)
- **URL:** wandb.ai/wandb-japan/llm-leaderboard
- **Relevance:** 7/10
- **Verification:** CONFIRMED

### Swallow LLM Leaderboard
- **Maintainer:** Institute of Science Tokyo
- **URL:** swallow-llm.github.io/leaderboard
- **Focus:** Japanese language task evaluation across all major models
- **Relevance:** 8/10
- **Verification:** CONFIRMED

### Open Japanese LLM Leaderboard (LLM-jp / NII)
- **Platform:** HuggingFace Space (llm-jp/open-japanese-llm-leaderboard)
- **Relevance:** 7/10
- **Verification:** CONFIRMED

### JGLUE
- **Tasks:** MARC-ja, JCommonsenseQA, JSTS, JNLI, JSQuAD
- **Status:** Standard suite, used by all Japanese LLM developers
- **Relevance:** 6/10
- **Verification:** CONFIRMED

### JMMMU-Pro (Dec 2025)
- **Type:** Image-based Japanese multi-discipline multimodal understanding
- **Method:** Vibe Benchmark Construction with image generation
- **Relevance:** 5/10
- **Verification:** CONFIRMED (arXiv:2512.14620)

### Heron-Bench
- **Type:** Japanese Vision Language Model benchmark
- **Relevance:** 5/10
- **Verification:** CONFIRMED (arXiv:2404.07824)

---

# 6. COMPANIES & LABS

## 6.1 Preferred Networks (PFN) - HIGHEST RELEVANCE

- **Founded:** 2014, Tokyo
- **Key Models:**
  - PLaMo-100B (ground-up Japanese LLM, paper: arXiv:2410.07563)
  - PLaMo 2.1 Prime (Oct 2025) - tool calling, multi-database integration
  - PLaMo Translate (May 2025) - adopted by Japan Digital Agency "Gennai" project
  - **PLaMo-fin-base (Feb 2026)** - Financial-specific derivative for institutions
  - nekomata-14b-pfn-qfin - Financial continual pretrained model
- **Hardware:** MN-Core / MN-Core 2 custom AI processors
- **Financial Benchmark:** japanese-lm-fin-harness, pfmt-bench-fin-ja
- **Key Researcher:** Masanori Hirano (financial LLM specialist)
- **Relevance to crypto:** 10/10 - Only company with dedicated Japanese financial LLM + benchmark
- **Verification:** CONFIRMED

## 6.2 Sakana AI - CRITICAL FOR FINANCE

- **Founded:** 2023 by David Ha (ex-Google Brain) and Llion Jones (Transformer co-inventor)
- **Valuation:** $2.635B (Nov 2025 Series B, $135M raised)
- **Core Technology:** Evolutionary model merging (Nature Machine Intelligence 2025)
- **Key Models:**
  - EvoLLM-JP, EvoVLM-JP (Japanese foundation models via evolutionary merging)
  - CycleQD (accepted ICLR 2025)
  - M2N2 (Aug 2025) - dynamic model merging
- **MUFG Partnership (May 2025):**
  - 3-year deal, largest AI transformation in Japanese banking
  - "AI Loan Expert" system in real-world testing
  - Ren Ito (COO) as MUFG AI advisor
- **EDINET-Bench:** Created Japanese financial benchmark (ICLR 2026)
- **NEDO/GENIAC Grant:** Received supercomputing grant
- **Relevance to crypto:** 9/10 - Financial AI production deployment, model merging applicable to our ensemble
- **Verification:** CONFIRMED

## 6.3 CyberAgent

- **Models:** OpenCALM series, CyberAgentLM3-22B, DeepSeek-R1 Japanese distillations
- **VLM:** Japanese 7.5B vision-language model (Apache-2.0)
- **Benchmark:** LCTG-Bench (controlled text generation)
- **Relevance to crypto:** 6/10
- **Verification:** CONFIRMED

## 6.4 Rinna Co.

- **Models:** Japanese GPT-NeoX (3.6B), GPT-2, RoBERTa, CLIP, CLOOB
- **Significance:** Pioneer in Japanese LLMs, first RLHF Japanese model
- **License:** Most models MIT
- **Relevance to crypto:** 6/10 (CLIP models useful for multimodal sentiment)
- **Verification:** CONFIRMED

## 6.5 NTT

- **Model:** tsuzumi 2 (Oct 2025)
  - Full-scratch development (no open-source base)
  - Runs on single GPU, performance comparable to GPT-5 on most tasks
  - Reinforced knowledge in financial, medical, and public sectors
  - Target: 500B yen orders by FY2027
- **Relevance to crypto:** 7/10 - Strong financial capability, but enterprise/proprietary
- **Verification:** CONFIRMED

## 6.6 SoftBank / SB Intuitions

- **Models:** Sarashina (460B), Sarashina Mini (commercialized Nov 2025)
- **Large Telecom Model (LTM):** Evolved to domestic AI model using Sarashina
- **OpenAI JV:** SB OAI Japan (50-50 with OpenAI), launching enterprise AI services 2026
- **Infrastructure:** World's largest NVIDIA DGX SuperPOD with DGX B200 (Dec 2025)
- **Relevance to crypto:** 5/10
- **Verification:** CONFIRMED

## 6.7 Rakuten

- **Model:** Rakuten AI 3.0 (Dec 2025)
  - ~700B params MoE (40B active per token)
  - Developed under GENIAC project
  - 90% cost reduction vs third-party frontier models
  - Open-weight release planned Spring 2026
- **Relevance to crypto:** 6/10 (general purpose, but massive scale)
- **Verification:** CONFIRMED

## 6.8 NRI (Nomura Research Institute)

- **Method:** Fine-tuning Llama 3.1 Swallow 8B with synthetic data
- **Result:** +9.6 percentage points over GPT-4o (2024-11-20) on insurance compliance
- **Approach:** Generate synthetic scenarios for domains with limited real data, then fine-tune
- **Standardized:** Method replicable across industries by modifying input data
- **Private LLM:** Generative AI solution minimizing data leak risks (Jan 2024)
- **Relevance to crypto:** 9/10 - Proven financial fine-tuning methodology we can replicate
- **Verification:** CONFIRMED (nri.com press release April 2025)

## 6.9 ABEJA

- **Focus:** Enterprise AI SaaS since 2012
- **Models:** ABEJA-V2 (32B Qwen 2.5), various fine-tunes
- **Scale:** 15M+ queries monthly (May 2025) for logistics/healthcare clients
- **Relevance to crypto:** 4/10
- **Verification:** CONFIRMED

## 6.10 Fujitsu

- **Role:** Lead designer of FugakuNEXT (with RIKEN and NVIDIA)
- **Platform:** Kozuchi AI platform
- **NEC:** Also developing Cotomi platform (Japanese language accuracy leader)
- **Relevance to crypto:** 4/10
- **Verification:** CONFIRMED

## 6.11 LINE Yahoo (LY Corporation)

- **AI Mandate:** All 11,000 employees required to use generative AI
- **Tools:** SeekAI (internal), Yahoo Search generative summaries
- **Infrastructure:** Building combined private cloud (Feb 2026)
- **LLM:** Historical Japanese large language models (line-corporation org on HF)
- **Relevance to crypto:** 4/10
- **Verification:** CONFIRMED

---

# 7. PAPERS (25+ Relevant Papers)

## 7.1 Japanese Financial NLP

| # | Title | Authors/Institution | Year | Key Finding | Relevance |
|---|-------|-------------------|------|-------------|-----------|
| 1 | Construction of Domain-specified Japanese LLM for Finance through Continual Pre-training | Hirano, Imajo (PFN) | 2024 | SOTA on japanese-lm-fin-harness with 10B-class models | 10/10 |
| 2 | Construction of Instruction-tuned LLMs for Finance without Instruction Data | Hirano, Imajo (PFN) | 2024 | Continual pretrain + instruction vector merging | 10/10 |
| 3 | Ebisu: Benchmarking LLMs in Japanese Finance | Peng et al. | 2026 | Even SOTA models struggle on implicit Japanese financial language | 9/10 |
| 4 | EDINET-Bench: Evaluating LLMs on Complex Financial Tasks | Sugiura et al. (Sakana AI) | 2025 | LLMs barely beat logistic regression on fraud detection | 9/10 |
| 5 | Economy Watchers Survey Provides Datasets and Tasks for Japanese Financial Domain | Suzuki, Sakaji | 2024 | Monthly economic sentiment dataset since 2000 | 9/10 |
| 6 | Enhancing Financial Domain Adaptation via Model Augmentation (CALM) | Tanabe, Hirano et al. | 2024 | Cross-attention between two LLMs for financial adaptation | 8/10 |
| 7 | Construction of a Japanese Financial Benchmark for LLMs | PFN | 2024 | Foundation of japanese-lm-fin-harness | 8/10 |
| 8 | NRI Task-Specific LLM Method | NRI | 2025 | Swallow 8B + synthetic data beats GPT-4o by 9.6pp | 9/10 |

## 7.2 Japanese LLM Training

| # | Title | Authors/Institution | Year | Key Finding | Relevance |
|---|-------|-------------------|------|-------------|-----------|
| 9 | Building a Large Japanese Web Corpus for LLMs | Okazaki et al. (IST) | 2024 | Swallow corpus construction methodology | 8/10 |
| 10 | Continual Pre-Training for Cross-Lingual LLM Adaptation | Fujii, Nakamura et al. | 2024 | Swallow training methodology: CPT > training from scratch | 8/10 |
| 11 | PLaMo-100B: Ground-Up Japanese Language Model | PFN | 2024 | QK Normalization, Z-Loss, SFT+DPO for 100B model | 7/10 |
| 12 | Stabilizing Reasoning in Medical LLMs (Preferred-MedLLM-Qwen-72B) | Kawakami et al. | 2025 | CPT + Reasoning Preference Optimization beats GPT-4o on IgakuQA | 7/10 |
| 13 | Why We Build Local LLMs: Analysis from 35 Japanese and Multilingual LLMs | Various | 2024 | Empirical study of Japanese LLM landscape | 7/10 |
| 14 | GPT-OSS-Swallow Model Card | IST | 2026 | 20B/120B models, highest Japanese performance at scale | 8/10 |

## 7.3 Crypto/Financial Prediction

| # | Title | Authors | Year | Key Finding | Relevance |
|---|-------|---------|------|-------------|-----------|
| 15 | Review of Deep Learning Models for Crypto Price Prediction | Wu et al. | 2024 | LSTM, CNN, Transformer comparison; multivariate outperforms univariate | 9/10 |
| 16 | Forecasting Bitcoin Volatility from Whale Transactions using Synthesizer Transformer | Herremans, Low | 2022 | CryptoQuant data + whale tweets for volatility spikes | 9/10 |
| 17 | Enhancing Price Prediction in Cryptocurrency using Transformer + Technical Indicators | Khaniki, Manthouri | 2024 | Performer neural network + FAVOR+ + BiLSTM | 8/10 |
| 18 | Neural Network-Based Algorithmic Trading: Multi-Timeframe in Crypto | Zhang | 2025 | Multi-timeframe + on-chain metrics + orderbook dynamics | 8/10 |
| 19 | Financial Time Series Forecasting using CNN and Transformer | Zeng et al. | 2023 | CNN for short-term + Transformer for long-term dependencies | 7/10 |
| 20 | Timer: Transformers for Time Series at Scale | Liu et al. | 2024 | GPT-style large time series model for forecasting/anomaly detection | 8/10 |
| 21 | Moirai: Unified Training of Universal Time Series Forecasting Transformers | Woo et al. | 2024 | Zero-shot forecasting with masked encoder architecture | 8/10 |
| 22 | PreBit: Multimodal with FinBERT for Bitcoin Price Movement | Zou, Herremans | 2022 | FinBERT + CNN + technical indicators + social media | 8/10 |
| 23 | FinDPO: Financial Sentiment for Algorithmic Trading via Preference Optimization | Iacovides et al. | 2025 | DPO for finance-specific sentiment, strong Sharpe ratio | 8/10 |

## 7.4 Model Merging & Efficiency

| # | Title | Authors | Year | Key Finding | Relevance |
|---|-------|---------|------|-------------|-----------|
| 24 | Evolutionary Optimization of Model Merging Recipes | Akiba et al. (Sakana AI) | 2024 | Evolutionary algorithm automates model merging without training | 9/10 |
| 25 | Competition and Attraction Improve Model Fusion (M2N2) | Abrantes et al. (Sakana AI) | 2025 | Dynamic evolutionary merging with diversity preservation | 7/10 |

## 7.5 Japanese Multimodal

| # | Title | Authors | Year | Key Finding | Relevance |
|---|-------|---------|------|-------------|-----------|
| 26 | DEJIMA: Novel Large-scale Japanese Dataset for VQA | Katsube et al. (UTokyo) | 2025 | 3.88M pairs, highest Japaneseness quality | 6/10 |
| 27 | WAON: Large-Scale Japanese Image-Text Dataset | Sugiura et al. | 2025 | SOTA on Japanese cultural benchmarks with SigLIP2 | 5/10 |
| 28 | Heron-Bench: Evaluating VLMs in Japanese | Inoue et al. | 2024 | Standard Japanese VLM evaluation | 5/10 |

## 7.6 Additional Financial/Time Series

| # | Title | Authors | Year | Relevance |
|---|-------|---------|------|-----------|
| 29 | FinMultiTime: Four-Modal Bilingual Dataset for Financial Time-Series | Xu et al. | 2025 | 8/10 |
| 30 | Instruct-FinGPT: Financial Sentiment by Instruction Tuning | Zhang et al. | 2023 | 7/10 |
| 31 | MADL Loss Function for Algorithmic Investment Strategies | Michankov et al. | 2023 | 7/10 |

---

# 8. SPECIAL TOPICS

## 8.1 Japan Crypto Regulation & Tax Reform (April 2026)

### Tax Reform - VERIFIED
- **Current:** Progressive taxation up to 55% (as "miscellaneous income")
- **New (April 2026):** Flat 20% separated taxation (aligned with stocks/capital gains)
- **Scope:** 105 approved cryptocurrencies including BTC, ETH
- **Reclassification:** Crypto assets become "financial products" under FIEA
- **Disclosure:** Exchanges must publish detailed token information (tech, volatility, risks)
- **AI Monitoring:** FSA deploying AI-powered tools to detect manipulation and abnormal trading
- **Impact:** Expected to drive significant institutional entry and retail adoption

### Market Size
- Revenue: $368.5M (2025) projected to $1.17B (2033)
- Top exchanges: bitFlyer (38% share), Coincheck (27.2%)
- bitFlyer saw ~200% volume surge in March 2026

## 8.2 Japan National AI Strategy

### AI Promotion Act (May 2025)
- Framework law promoting AI development and utilization
- Light-touch regulation (unlike EU AI Act)
- Effective June 4, 2025

### AI Basic Plan (December 2025)
- First national AI plan approved
- Four pillars: accelerate AI use, strengthen development, improve reliability, transform society
- 1 trillion yen ($6.34B) five-year support from FY2026
- New public-private AI company for domestic foundation models

### Government AI Platform "Gennai"
- Planned deployment to 100,000+ public officials from May 2026
- Uses PLaMo Translate for Japanese translation

## 8.3 GENIAC Project (Generative AI Accelerator Challenge)

- **Organizer:** METI + NEDO
- **Purpose:** Compute resources for domestic foundation model development
- **Participants Include:**
  - Rakuten (produced Rakuten AI 3.0 - 700B MoE)
  - Sakana AI (received NEDO grant)
  - Ubitus (405B model for East Asian languages)
  - Institute of Science Tokyo (Swallow models)
- **Infrastructure:** Ubitus building new AI data center in Maizuru City, Kyoto (2026-2027)

## 8.4 Fugaku & FugakuNEXT

### Current Fugaku
- Used for Swallow LLM training (IST, RIKEN, Fujitsu, Tohoku U collaboration)
- 442 petaflops (Top500)

### FugakuNEXT
- **Partners:** RIKEN + Fujitsu + NVIDIA
- **Timeline:** Basic design complete FY2025, detailed design FY2026, operational ~2030
- **Performance:** 100x application performance over Fugaku
- **Innovation:** First Japanese flagship system with GPU accelerators (NVIDIA)
- **Budget:** ~$750M
- **Applications:** AI for Science, hypothesis generation, experiment simulation

## 8.5 CoinPost - Japanese Crypto Media

- **Founded:** August 2017
- **Status:** Japan's #1 crypto media, global top-3 for 22+ consecutive months (SimilarWeb)
- **App:** CoinPost App (iOS) with proprietary "CoinPost Crypto Indicator"
- **Data:** Fundamentals-based indicators affecting Bitcoin and crypto markets
- **Relevance:** 8/10 - Primary Japanese-language crypto sentiment source
- **Verification:** CONFIRMED

## 8.6 Japan Market Microstructure

- **Exchanges:** bitFlyer (38% market share), Coincheck (27.2%), GMO Coin, bitbank
- **Regulation:** All exchanges registered with FSA, strict KYC/AML
- **Hours:** 24/7 crypto trading, but Japanese market hours (9:00-15:00 JST) show distinct patterns
- **Unique Feature:** High retail participation, LINE integration with crypto services

---

# 9. INTEGRATION RECOMMENDATIONS FOR OUR PLATFORM

## 9.1 Immediate Actions (High Impact, Available Now)

### A. Japanese Sentiment Pipeline
1. **Base Model:** Qwen3-Swallow-8B-RL-v0.2 (Apache-2.0, 8B params, latest)
2. **Sentiment Backbone:** tohoku-nlp/bert-base-japanese-v3 for classification
3. **Financial Fine-tune:** Follow NRI methodology - generate synthetic Japanese financial scenarios, fine-tune Swallow
4. **Data Source:** Economy Watchers Survey (monthly auto-updated), CoinPost scraping
5. **Benchmark:** Evaluate on japanese-lm-fin-harness + EDINET-Bench

### B. Financial Analysis Agent
1. **Model:** pfnet/nekomata-14b-pfn-qfin-inst-merge (financial instruction-tuned)
2. **Alternative:** GPT-OSS-Swallow-20B-RL-v0.1 (stronger reasoning, Apache-2.0)
3. **Task:** Analyze EDINET filings, detect market signals from Japanese corporate disclosures

### C. Evolutionary Model Merging
1. **Tool:** Sakana AI's evolutionary-model-merge (GitHub: SakanaAI/evolutionary-model-merge)
2. **Application:** Automatically find optimal merge of our 16 ML models
3. **Paper:** arXiv:2403.13187

## 9.2 Medium-Term (1-3 months)

### D. Japanese Market Context Memory
1. Integrate Economy Watchers Survey as monthly macro signal
2. Track FSA regulatory announcements (105 approved tokens list)
3. Monitor bitFlyer/Coincheck volume as Japan-specific indicator
4. April 2026 tax reform as regime-change signal

### E. Multimodal Document Understanding
1. **Dataset:** DEJIMA for training Japanese document understanding
2. **Models:** rinna/japanese-clip-vit-b-16 for visual features
3. **Application:** Parse Japanese financial charts, reports, regulatory documents

### F. Dedicated Japanese Financial LLM
1. Follow Hirano/PFN methodology (arXiv:2404.10555):
   - Continual pretrain Swallow on Japanese financial corpus
   - Merge instruction-tuned vector (arXiv:2409.19854)
2. Evaluate on Ebisu + EDINET-Bench + japanese-lm-fin-harness

## 9.3 Long-Term (3-6 months)

### G. Multi-Agent Japanese Market Intelligence
1. Agent 1: CoinPost scraper + Japanese crypto sentiment (Swallow-based)
2. Agent 2: EDINET filing analyzer (nekomata financial model)
3. Agent 3: Economy Watchers macro sentiment tracker
4. Agent 4: bitFlyer/Coincheck order flow analyzer
5. Orchestrator: Use evolutionary merging to combine agent signals

### H. Infrastructure
1. Apply for ABCI 3.0 access (AIST) for model training
2. Monitor Rakuten AI 3.0 open-weight release (Spring 2026)
3. Track PLaMo-fin-base availability for financial institutions

---

# APPENDIX A: KEY URLs AND RESOURCES

## Model Repositories
- Swallow: https://hf.co/tokyotech-llm
- PFN Financial: https://hf.co/pfnet
- CyberAgent: https://hf.co/cyberagent
- Rinna: https://hf.co/rinna
- Stability AI Japan: https://hf.co/stabilityai (japanese-stablelm-*)
- Tohoku NLP: https://hf.co/tohoku-nlp
- LLM-jp: https://hf.co/llm-jp
- SB Intuitions: https://hf.co/sbintuitions
- Sakana AI: https://hf.co/SakanaAI

## Benchmarks & Leaderboards
- Swallow Leaderboard: https://swallow-llm.github.io/leaderboard
- Nejumi Leaderboard: https://wandb.ai/wandb-japan/llm-leaderboard
- Open Japanese LLM Leaderboard: https://huggingface.co/spaces/llm-jp/open-japanese-llm-leaderboard
- japanese-lm-fin-harness: https://github.com/pfnet-research/japanese-lm-fin-harness
- EDINET-Bench: https://github.com/SakanaAI/EDINET-Bench

## Key Datasets
- Economy Watchers Survey: https://hf.co/datasets/retarfi/economy-watchers-survey
- EDINET-Bench: https://hf.co/datasets/SakanaAI/EDINET-Bench
- JGLUE: Standard HuggingFace datasets

## Tools
- Evolutionary Model Merge: https://github.com/SakanaAI/evolutionary-model-merge
- KWJA (Japanese Analyzer): https://github.com/ku-nlp/kwja
- Juman++: https://github.com/ku-nlp/jumanpp
- edinet2dataset: https://github.com/SakanaAI/edinet2dataset

## Government & Regulatory
- FSA: https://www.fsa.go.jp/en/
- GENIAC/METI: https://www.meti.go.jp/english/
- ABCI 3.0: https://abci.ai/en/

## Media
- CoinPost: https://coinpost.jp/
- CoinPost Corporate: https://coinpost.jp/corporate_global/

---

# APPENDIX B: VERIFICATION STATUS SUMMARY

| Item | Status | Source |
|------|--------|--------|
| Swallow LLM (tokyotech-llm) | VERIFIED | HuggingFace, papers |
| DEJIMA dataset (arXiv:2512.00773) | VERIFIED | arXiv, project website |
| NRI Swallow fine-tune beating GPT-4o | VERIFIED (+9.6pp on insurance compliance) | NRI press release |
| CoinPost as largest Japanese crypto site | VERIFIED (global top-3) | CoinPost corporate |
| Japan 20% crypto tax (April 2026) | VERIFIED | Multiple news sources, FSA proposal |
| GENIAC project | VERIFIED | METI official, multiple participants |
| Sakana AI + MUFG partnership | VERIFIED | sakana.ai, Bloomberg |
| PLaMo-fin-base (Feb 2026) | VERIFIED | itbusinesstoday.com |
| NTT tsuzumi 2 | VERIFIED | NTT press release |
| Rakuten AI 3.0 (700B MoE) | VERIFIED | Rakuten press release |
| LLM-jp 172B model | VERIFIED | NII press release |
| ABCI 3.0 (6128 H200 GPUs) | VERIFIED | NVIDIA blog, AIST |
| FugakuNEXT partnership | VERIFIED | RIKEN, Fujitsu, NVIDIA |
| Economy Watchers Survey dataset | VERIFIED | HuggingFace, arXiv |
| EDINET-Bench (ICLR 2026) | VERIFIED | OpenReview, GitHub |
| Ebisu benchmark | VERIFIED | arXiv:2602.01479 |
| GPT-OSS-Swallow-120B | VERIFIED | HuggingFace, gigazine.net |
| Sarashina (SoftBank) | VERIFIED | SoftBank press release |
| AI Promotion Act (May 2025) | VERIFIED | Multiple legal sources |
| bitFlyer 200% volume surge (Mar 2026) | VERIFIED | bitcoinke.io |
