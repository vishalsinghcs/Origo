<div align="center">
  <br />
  <h1>Origo</h1>
  <h3>Enterprise AI Security Gateway</h3>
  <p>A multi-layered, offline security proxy that protects confidential information before it reaches any LLM — powered by fine-tuned Small Language Models running on a single GPU.</p>
  <br />

  [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](#tech-stack)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](#tech-stack)
  [![Qwen2.5](https://img.shields.io/badge/Qwen2.5-3B%20%7C%208B-7C3AED.svg)](#architecture)
  [![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-EF4444.svg)](#fine-tuning)
  [![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](#deployment)
  [![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20SageMaker%20%7C%20RDS-FF9900.svg?logo=amazonaws&logoColor=white)](#deployment)
  [![Status](https://img.shields.io/badge/Status-Active%20Development-22C55E.svg)](#)
</div>

<br/>

---

## About

Origo is an end-to-end machine learning engineering project that builds a fully functional AI security gateway from scratch — covering synthetic data generation, deep data preprocessing, QLoRA fine-tuning, model evaluation, API development, and cloud deployment.

The system sits as an invisible proxy between enterprise users and external LLM APIs (OpenAI, Anthropic, etc.). Every prompt is intercepted, analyzed through multiple security layers, and either allowed through cleanly, rewritten with sensitive data redacted, or blocked entirely — all in real time with minimal latency overhead.

The core innovation is replacing expensive, fragile zero-shot prompting with **purpose-built fine-tuned models** that are immune to prompt injection attacks, require no lengthy system prompts, and run entirely offline on commodity hardware.

> 🔧 *This project is in active development. The data engineering and preprocessing pipelines are complete, with model fine-tuning and deployment phases currently underway.*

---

## Unique Selling Points

- **Offline & Private**: All security analysis happens locally. No prompts are sent to third-party guardrail services. The models run on a single consumer GPU.
- **Fine-Tuned Over Zero-Shot**: Instead of relying on massive system prompts (1,500+ tokens of instructions per request), Origo bakes security knowledge directly into model weights via QLoRA fine-tuning — slashing token overhead and eliminating prompt injection vulnerabilities.
- **Multi-Layered Defense**: Four distinct security layers (Regex → 3B Classifier → 8B Rewriter → Policy Engine) ensure no single point of failure. Each layer is optimized for its specific task.
- **Bespoke Synthetic Data Pipeline**: Since real enterprise security data is highly sensitive and scarce, the entire training corpus is synthetically generated using teacher LLMs, then rigorously cleaned through N-gram deduplication, hallucination filtering, and deep Pandas-based preprocessing.
- **Production-Ready Architecture**: Not a notebook demo. The system is designed with Docker Compose orchestration, PostgreSQL audit logging, a React admin dashboard, and full AWS deployment with vLLM inference serving.

---

## Architecture

Origo distributes the security workload across four specialized layers, each optimized for speed and accuracy at its specific task:

```
                     User Prompt
                          │
                          ▼
            ┌─────────────────────────────┐
            │  Layer 0: Deterministic     │
            │  Scanner (Python + Regex)   │
            │                             │
            │  • Credit Cards, SSNs       │
            │  • API Keys, JWTs           │
            │  • High-Entropy Secrets     │
            │  • Emails, Phone Numbers    │
            └─────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │  Layer 1: Semantic Router   │
            │  Qwen2.5-3B (Fine-tuned)    │
            │                             │
            │  SAFE │ SUSPICIOUS │ UNSAFE │
            │  + Confidence Score         │
            └─────────────────────────────┘
                 │                 │
                 ▼                 ▼
           Safe Path        Suspicious / Unsafe
                 │                 │
                 │                 ▼
                 │    ┌──────────────────────────┐
                 │    │  Layer 2: Specialist     │
                 │    │  Rewriter (Qwen2.5-8B)   │
                 │    │                          │
                 │    │  • Entity Extraction     │
                 │    │  • Threat Classification │
                 │    │  • Prompt Sanitization   │
                 │    │  • Action Decision       │
                 │    │    (REDACT/REWRITE/BLOCK)│
                 │    └──────────────────────────┘
                 │                 │
                 └────────┬────────┘
                          ▼
            ┌─────────────────────────────┐
            │  Layer 3: Policy Engine     │
            │  (Deterministic Python)     │
            │                             │
            │  • Enterprise Rule Matching │
            │  • Token Replacement        │
            │  • Block / Allow Decision   │
            │  • PostgreSQL Audit Logging │
            └─────────────────────────────┘
                          │
                          ▼
                    Final Safe Prompt
                   → External LLM API
```

**Post-Rewrite Verification**: After Layer 2 rewrites a prompt, it passes through Layer 0's scanners again. If any secrets remain after rewriting, the prompt is hard-blocked — ensuring zero data leakage even if the neural model makes an error.

**Admin Dashboard**: A React (Vite + TypeScript + TailwindCSS) frontend provides security officers with real-time visibility into intercepted threats, latency metrics, searchable audit trails, and configurable policy toggles for each threat category.

---

## End-to-End Workflow

The project follows a strict, phased engineering workflow:

### Phase 1 — Synthetic Data Generation
Real enterprise security data is sensitive and scarce. Instead, we use teacher LLMs (Groq `llama-3.3-70b-versatile`) to synthetically generate tens of thousands of training samples.

- `generate_layer1.py` produces `SAFE`, `SUSPICIOUS`, and `UNSAFE` classification samples with reasoning, sensitive elements, and attack vectors.
- `generate_layer2.py` produces complex rewriting samples: original prompts paired with entity extractions, threat subcategories, confidence scores, and sanitized rewrites.
- Both scripts use true `asyncio` parallelism across multiple API keys with automatic rate-limit handling, exponential backoff, and chunked `.jsonl` output.

### Phase 2 — Quality Filtering & Data Preprocessing
Raw LLM outputs contain hallucinations, malformed JSON, and duplicates. We clean the data through multiple stages:

- **Automated Filtering** (`quality_filter.py`): N-gram hashing and exact-match deduplication to remove overlapping samples. Drops malformed JSON lines and samples from unreliable model endpoints.
- **Deep Pandas Preprocessing** (Jupyter Notebooks): Manual exploratory data analysis to catch edge cases the automated filters miss — hallucinated dictionary structures inside string arrays, invalid action/severity values, missing `rewritten_prompt` fields, and stray regex or raw Markdown leaked into text columns.
- **Interactive Visualization** (Streamlit Dashboards): Custom dashboards for Layer 1 and Layer 2 with distribution charts, threat-vs-action heatmaps, entity type rankings, and filterable data explorers to verify dataset health before training.

### Phase 3 — ChatML Conversion
The cleaned JSON datasets are transformed into the conversational `ChatML` format required by HuggingFace's `SFTTrainer`.

- `convert_layer1.py` maps each sample into a `system → user → assistant` conversation where the assistant outputs the classification label and reasoning.
- `convert_layer2.py` maps each sample into a structured security analysis response including threat type, severity, detected entities, reasoning, and the sanitized prompt.

### Phase 4 — QLoRA Fine-Tuning on AWS
Models are fine-tuned using parameter-efficient techniques on AWS GPU instances.

- **Infrastructure**: EC2 `g5.2xlarge` (A10G 24GB) for the 3B Router; `g5.12xlarge` (4×A10G) for the 8B Rewriter. AWS Deep Learning AMI with 200GB EBS NVMe storage.
- **Training Stack**: Unsloth for 2–5× faster training and 70% less VRAM via custom Triton kernels and padding-free packing. QLoRA with 4-bit NF4 quantization, Paged AdamW optimizer, cosine scheduler.
- **Hyperparameters**: LoRA Rank 64, Alpha 128, Dropout 0.05. Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. 3 epochs, LR 2e-4, 5% warmup, max sequence length 2048.
- **Post-Training**: LoRA adapters are merged back into the base models and exported as standalone Safetensors / GGUF artifacts, then published to HuggingFace.

### Phase 5 — Backend & API Development
A FastAPI application orchestrates the multi-layer pipeline and exposes an OpenAI-compatible proxy endpoint.

- vLLM serves the merged Qwen models with continuous batching and massive concurrency.
- The API intercepts `POST /v1/chat/completions` requests, routes them through Layer 0 → 1 → 2 → 3, and forwards the sanitized prompt to the destination LLM.
- PostgreSQL stores structured audit logs (timestamps, user IDs, original/rewritten prompts, detected entities, triggered policies, per-layer latency).

### Phase 6 — Containerization & AWS Deployment
The entire stack is containerized and deployed to AWS for production use.

- **Docker Compose**: Separate containers for the FastAPI backend (Uvicorn), React frontend (Nginx), PostgreSQL database, and vLLM GPU inference server — all orchestrated with a single `docker compose up`.
- **AWS Production**: Amazon RDS for managed PostgreSQL, ECS on Fargate for the backend/frontend containers, ECS on EC2 GPU instances (`g5.xlarge`) for vLLM inference, and ALB with HTTPS/SSL via AWS ACM for secure traffic routing.
- **CI/CD**: GitHub Actions for automated linting, testing, and type-checking on every pull request.

---

## Engineering Highlights

A summary of the deeper engineering decisions and optimizations applied across the project:

| Area | Technique |
|------|-----------|
| **Data Generation** | Async parallelism across multiple API keys with exponential backoff and automatic rate-limit recovery |
| **Deduplication** | N-gram fuzzy hashing + exact-match deduplication to eliminate overlapping synthetic samples |
| **Hallucination Removal** | Boolean masking to detect and purge LLM-generated artifacts (raw Markdown tables, regex patterns, JSON blobs) leaked into text fields |
| **Type Consistency** | Custom `apply()` functions to normalize hallucinated dictionary structures (e.g., `{"type": "API_KEY"}`) back into flat string lists |
| **Missing Value Strategy** | Conditional `.loc[]` fills — copying `original_prompt` into empty `rewritten_prompt` only for `ALLOW` actions; filling `BLOCK` actions with explicit `"BLOCKED"` tokens |
| **Model Selection** | Qwen2.5 chosen for its 151k-token vocabulary (superior entropy detection), native multilingual support, and reliable structured JSON output |
| **Training Optimization** | Unsloth's custom Triton kernels and padding-free packing for 2–5× speedup and 70% VRAM reduction |
| **Post-Rewrite Safety Net** | After Layer 2 rewrites, the prompt is re-scanned by Layer 0's regex/entropy detectors — any remaining secrets trigger an automatic hard block |
| **Inference Serving** | vLLM for continuous batching, PagedAttention, and OpenAI-compatible API serving |
| **Structured Audit Logging** | Every scan is logged to PostgreSQL with per-layer latency, detected entities, and policy decisions for compliance reporting |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11+ |
| **Backend Framework** | FastAPI, Uvicorn |
| **Layer 1 Model** | Qwen2.5-3B-Instruct (Fine-tuned QLoRA) |
| **Layer 2 Model** | Qwen2.5-8B-Instruct (Fine-tuned QLoRA) |
| **Training Library** | Unsloth, TRL, HuggingFace Transformers |
| **Inference Engine** | vLLM / llama.cpp |
| **Data Engineering** | Pandas, Plotly, Streamlit, Jupyter |
| **Frontend** | React, TypeScript, Vite, TailwindCSS |
| **Database** | PostgreSQL (Audit Logs, Policies, RBAC) |
| **Containerization** | Docker, Docker Compose |
| **Cloud Infrastructure** | AWS EC2, S3, SageMaker, RDS, ECS, ALB |
| **CI/CD** | GitHub Actions |
| **Evaluation Judge** | GPT-4.1 / Claude / Gemini |
| **Benchmarking Targets** | Microsoft Presidio, spaCy, LLM Guard, NVIDIA NeMo Guardrails |

---

## Evaluation Strategy

The project measures performance across multiple dimensions:

- **Classification Metrics**: Precision, Recall, F1-Score, ROC AUC, and Confusion Matrices for Layer 1 routing accuracy.
- **Latency Profiling**: P50, P95, and P99 latency measurements per layer and end-to-end.
- **Throughput**: Tokens/second and Requests/second under load.
- **Comparative Benchmarking**: Side-by-side evaluation against Microsoft Presidio, spaCy NER, LLM Guard, Guardrails AI, and NVIDIA NeMo Guardrails.
- **Ablation Studies**: Systematic experiments comparing Regex-only → Regex + 3B → Regex + 3B + 8B configurations, and LoRA rank sweeps (16, 32, 64, 128) to demonstrate engineering judgment in architectural decisions.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

All datasets, fine-tuned model adapters, and source code are freely available.
