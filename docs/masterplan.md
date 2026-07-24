# Origo Masterplan
# Enterprise AI Security Gateway

This document serves as the end-to-end implementation plan and project roadmap for Origo, outlining the strategic phases from data generation to cloud deployment.

## Phase 1: Data Synthesis & Pipeline Foundation (Currently Active)
**Goal:** Generate the synthetic datasets required to fine-tune the Layer 1 and Layer 2 models. Since handling real enterprise security data is dangerous, we use teacher models to generate highly diverse synthetic data.

- **Tasks:**
  - [x] Configure Async generation scripts (`generate_layer1.py`, `generate_layer2.py`) utilizing Groq models (`llama-3.3-70b-versatile`, `openai/gpt-oss-120b`).
  - [x] Design rigid System Prompts with strict Enums for categories (e.g., `CREDIT_CARD`, `API_KEY`).
  - [x] Develop edge cases for model restraint (`safe_baseline`, `false_positive_safe`).
  - [ ] Run generation pipeline to hit volume targets (7,000 for Layer 1; 10,000 for Layer 2).
  - [ ] Run `quality_filter.py` to ensure dataset uniqueness (N-Gram hashing).
  - [ ] Convert JSON records into ChatML format via `convert_layer1.py` and `convert_layer2.py`.

## Phase 2: Model Training & Evaluation on AWS
**Goal:** Fine-tune Qwen2.5 models on the synthetic data using AWS infrastructure to lock their weights into a specialized security-focused task.

- **Tasks:**
  - [ ] Provision AWS environment (EC2 g5.xlarge) with PyTorch and Unsloth.
  - [ ] Fine-tune Layer 1 (Qwen2.5-3B-Instruct) for tri-class classification (SAFE, SUSPICIOUS, UNSAFE).
  - [ ] Fine-tune Layer 2 (Qwen2.5-VL-8B-Instruct) for complex reasoning and entity extraction.
  - [ ] Export LoRA adapters and merge them into the base models.
  - [ ] Benchmark against Microsoft Presidio, spaCy, and cloud models using the `benchmarks/` suite.

## Phase 3: Core Engine & FastAPI Backend
**Goal:** Build the runtime inference architecture to orchestrate the layers into a seamless security gateway API.

- **Tasks:**
  - [ ] Develop Layer 0 (Fast Deterministic Scanner) using Python regex and high-entropy secret detection.
  - [ ] Develop Layer 3 (Policy Engine) to enforce redactions, token replacement, and block/allow logic based on company policies.
  - [ ] Integrate vLLM / llama.cpp for high-throughput model inference.
  - [ ] Orchestrate the `0 -> 1 -> 2 -> 3` pipeline.
  - [ ] Expose pipeline via FastAPI `POST /scan` endpoint.

## Phase 4: Admin Dashboard & Audit System
**Goal:** Provide an enterprise dashboard for monitoring security events and configuring policies.

- **Tasks:**
  - [ ] Set up PostgreSQL database for secure audit logging.
  - [ ] Build React (Vite) admin UI using TypeScript and TailwindCSS.
  - [ ] Implement Analytics dashboard (threat breakdown, latency trends).
  - [ ] Implement Scan History view for auditing.
  - [ ] Implement Policy Configuration panel (adjusting thresholds and activating rules).

## Phase 5: Deployment & Integration
**Goal:** Package the entire system for scalable cloud deployment.

- **Tasks:**
  - [ ] Dockerize the FastAPI backend and React frontend.
  - [ ] Create `docker-compose.yml` for simplified orchestration.
  - [ ] Deploy to production infrastructure (AWS).
