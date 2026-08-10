# Implementation Plan: End-to-End Fine-Tuning Pipeline

This plan outlines the exact, chronological, step-by-step execution strategy to set up the AWS infrastructure **manually via the AWS GUI**, run the QLoRA fine-tuning for both models, evaluate their performance, and merge them for production.

## Recommended AWS Instance: `g6.2xlarge`
After researching the requirements for training an 8B model with Unsloth (which requires around ~12-16GB of VRAM during QLoRA) against current AWS pricing, I am firmly targeting the **`g6.2xlarge`** instance for this guide.
- **Cost-Effective**: At approximately **~$0.97 per hour**, it is significantly cheaper than standard A10G (`g5.2xlarge` at ~$1.21/hr) or A100 instances. Your $200 credits will easily stretch for over 200 hours of uptime.
- **Powerful GPU**: It features the newer Nvidia L4 GPU (24GB VRAM, Ada Lovelace architecture), which trains faster and more efficiently than older generations.
- **System Stability**: It provides 32GB of system RAM, ensuring your instance doesn't crash when initially loading the 8B model into memory before moving it to the GPU.

## Proposed Changes (Step-by-Step Workflow)

### Step 1: Environment Provisioning (AWS GUI & Manual Setup)
Instead of an automated script, we will create a detailed, step-by-step instructional guide so you can learn exactly how to provision and configure high-performance cloud GPUs manually.

#### [NEW] [docs/deployment.md](file:///c:/Users/Techn/Programming/Projects/Enterprise%20AI%20Gateway/docs/deployment.md)
- **Action**: Create a comprehensive markdown guide for manual AWS setup and deployment.
- **Details**: 
  - **GUI Steps**: Detailed instructions on navigating the AWS EC2 Console, selecting the correct AMI (Deep Learning AMI Ubuntu 22.04), configuring the instance type (`g5.2xlarge`), setting up EBS storage (200GB), configuring Security Groups for SSH, and launching the instance.
  - **Manual Terminal Steps**: The exact commands you will paste into your SSH terminal to manually install Miniconda, create the `origo-train` environment, install PyTorch (CUDA 12.1), and install Unsloth/HuggingFace dependencies.

---

### Step 2: Training Configuration Setup
We will isolate the hyperparameters into YAML files so they can be easily modified for ablation studies without breaking the core python scripts.

#### [NEW] [training/layer1_router/config.yaml](file:///c:/Users/Techn/Programming/Projects/Enterprise%20AI%20Gateway/training/layer1_router/config.yaml)
#### [NEW] [training/layer2_specialist/config.yaml](file:///c:/Users/Techn/Programming/Projects/Enterprise%20AI%20Gateway/training/layer2_specialist/config.yaml)
- **Action**: Create configuration files holding the values specified in `idea.md`.
- **Details**: 
  - **LoRA**: Rank=64, Alpha=128, Dropout=0.05, Target Modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
  - **Training**: Epochs=3, LR=2e-4, Batch Size=1, Gradient Acc=16 (effective batch size 16), Warmup=5%, Scheduler=Cosine, Mixed Precision=bf16, Paged AdamW.

---

### Step 3: Layer 1 (Router) Training
We will build the script to fine-tune the 3B parameter model to classify prompts as SAFE, SUSPICIOUS, or UNSAFE.

#### [NEW] [training/layer1_router/train.py](file:///c:/Users/Techn/Programming/Projects/Enterprise%20AI%20Gateway/training/layer1_router/train.py)
- **Action**: Create the Unsloth training script for Layer 1.
- **Details**: Instantiates `FastLanguageModel` with `Qwen/Qwen2.5-3B-Instruct`, applies the 4-bit LoRA config, loads `train.jsonl` and `val.jsonl` from `datasets/formatted/layer1/`, and executes the `SFTTrainer` loop. Saves checkpoints to `training/layer1_router/checkpoints/`.

---

### Step 4: Layer 2 (Specialist) Training
We will build the script to fine-tune the 8B parameter model for complex semantic extraction, threat classification, and prompt sanitization.

#### [NEW] [training/layer2_specialist/train.py](file:///c:/Users/Techn/Programming/Projects/Enterprise%20AI%20Gateway/training/layer2_specialist/train.py)
- **Action**: Create the Unsloth training script for Layer 2.
- **Details**: Identical pipeline structure to Layer 1, but loads `Qwen/Qwen3-8B` and targets the `datasets/formatted/layer2/` data. Saves checkpoints to `training/layer2_specialist/checkpoints/`.

---

### Step 5: Model Evaluation & Benchmarking
After training, we must mathematically evaluate how well the models perform on the unseen `test.jsonl` data.

#### [NEW] [training/shared/evaluate.py](file:///c:/Users/Techn/Programming/Projects/Enterprise%20AI%20Gateway/training/shared/evaluate.py)
- **Action**: Create an evaluation script.
- **Details**: Runs inference on the test sets using the trained LoRA adapters. 
  - **Layer 1 Evaluation**: Calculates overall Precision, Recall, and F1-Scores for the classification labels (`SAFE`, `SUSPICIOUS`, `UNSAFE`). Generates a Confusion Matrix to spot routing errors.
  - **Layer 2 Evaluation**: Evaluates multiple complex dimensions: F1-Scores for Action determination (`REDACT`, `REWRITE`, `BLOCK`), Threat Classification accuracy, and Entity Extraction accuracy (evaluating whether it successfully identifies all sensitive values like API keys without hallucinations).

---

### Step 6: Model Merging & Export
To deploy the models at maximum speed in Phase 3 (vLLM inference), the trained LoRA weights must be natively merged into the base model.

#### [NEW] [training/shared/merge_lora.py](file:///c:/Users/Techn/Programming/Projects/Enterprise%20AI%20Gateway/training/shared/merge_lora.py)
- **Action**: Create a LoRA merging script.
- **Details**: Uses Unsloth's native `save_pretrained_merged` to bake the LoRA weights directly into the base Qwen models, exporting them as 16-bit safetensors and/or 4-bit GGUF files ready for production deployment.

## Verification Plan
1. **Approval**: You will review this step-by-step plan and approve the creation of the files.
2. **Infrastructure**: You will read through `docs/aws_training_guide.md`, manually provision your EC2 instance through the AWS GUI, and run the manual setup commands via SSH.
3. **Smoke Testing**: You will execute a "dry-run" of `train.py` (with `max_steps=10`) for both layers to verify VRAM usage and gradient updates before committing to the full 3-epoch runs.
4. **Execution**: Run the full training, evaluate using `evaluate.py`, and merge the final models.
