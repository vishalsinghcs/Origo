# AWS Deployment & Training Setup Guide

This guide will walk you through manually provisioning an AWS EC2 instance via the graphical user interface (GUI) and configuring the high-performance Unsloth training environment via SSH.

We are targeting a **`g6.2xlarge`** instance, which utilizes the Nvidia L4 GPU (24GB VRAM) and 32GB of system RAM — the perfect cost-to-performance ratio for training an 8B parameter model.

---

## Phase 1: Launching the EC2 Instance (AWS GUI)

1. **Log into AWS**: Navigate to the [AWS Management Console](https://console.aws.amazon.com/) and search for **EC2**.
2. **Launch Instance**: Click the orange **Launch instance** button.
3. **Name your Server**: Under _Name and tags_, enter `Origo-Training-Server`.
4. **Choose the AMI (Operating System)**:
   - Under _Application and OS Images_, search for: `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)`, it will be available in community.
   - _Why?_ This AMI comes pre-loaded with the exact Nvidia drivers required for AI workloads, saving you from complex kernel installations.
5. **Select Instance Type**:
   - Under _Instance type_, search for and select **`g6.2xlarge`**.
   - **Handling vCPU Quota Errors**: If you get a "vCPU limit of 0" error, your account needs permission to run GPU instances.
     1. Open a new tab and search for **Service Quotas** in the top AWS search bar.
     2. Click **AWS services** on the left, then click **Amazon Elastic Compute Cloud (Amazon EC2)**.
     3. Search for the quota named **Running On-Demand G and VT instances**.
     4. Select it, click **Request quota increase**, and request a value of **`8`** (which covers the 8 vCPUs needed for a `g6.2xlarge`).
        _(Note: Approval can take from a few minutes to 24 hours. If denied, reply to the support ticket explaining you are a developer fine-tuning an AI model)._
6. **Create a Key Pair (Crucial for Access)**:
   - Click **Create new key pair**.
   - Name it `origo-key`.
   - Select **RSA** and **.pem**.
   - Click **Create key pair**. _The `.pem` file will immediately download to your computer. Keep it safe; you cannot download it again!_
7. **Configure Security Group (Networking)**:
   - Under _Network settings_, check the box for **Allow SSH traffic from**.
   - Set it to **My IP** (most secure) or **Anywhere (0.0.0.0/0)**.
8. **Configure Storage**:
   - Under _Configure storage_, change the root volume size to **`200` GB**. Leave the type as `gp3`.
   - _Why 200GB?_ Even though models are loaded in 4-bit (quantized) and datasets might be small, HuggingFace initially caches the unquantized 16-bit weights (taking ~25GB). Furthermore, Ubuntu, CUDA, and Miniconda environments consume ~20GB. Finally, saving multiple training checkpoints (which include optimizer states) and merging LoRA adapters requires duplicating model footprints on disk. Running out of storage will immediately crash your training job. At ~$0.08/GB, this 200GB buffer acts as a very cheap safety net.
9. **Launch**: Click the orange **Launch instance** button on the bottom right.

---

## Phase 2: Connecting to Your Server

1. Open your terminal (PowerShell or Windows Terminal).
2. Navigate to the folder where your `origo-key.pem` was downloaded (usually `Downloads`).
3. **Crucial Windows Step (Fixing Key Permissions):** SSH on Windows will reject your key with a "WARNING: UNPROTECTED PRIVATE KEY FILE!" error if the file permissions are too open. Run these exact commands in PowerShell to lock down the key so only your user account can read it:
   ```powershell
   icacls.exe origo-key.pem /reset
   icacls.exe origo-key.pem /grant:r "$($env:USERNAME):(R)"
   icacls.exe origo-key.pem /inheritance:r
   ```
4. Connect using SSH. Replace `<YOUR-EC2-PUBLIC-IP>` with the public IPv4 address found on your EC2 Dashboard:
   ```bash
   ssh -i origo-key.pem ubuntu@<YOUR-EC2-PUBLIC-IP>
   ```
5. Type `yes` when prompted to accept the fingerprint.

---

## Phase 3: Manual Environment Setup (Terminal)

Once logged into your Ubuntu server, run these commands step-by-step to perfectly configure your environment.

### 1. Install Miniconda

Miniconda isolates your Python packages so they don't break the system OS.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init bash
```

_Close your SSH terminal and reconnect to apply the conda initialization._

### 2. Create the Training Environment

Create an environment specifically built for PyTorch and Unsloth.
_(Note: Anaconda recently introduced a prompt requiring users to accept their Terms of Service for free/personal use before creating environments)._

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n origo-train python=3.11 -y
conda activate origo-train
```

### 3. Install PyTorch via Pip (Bypassing Conda Bugs)

```bash
# Install PyTorch with CUDA 12.4 support FIRST
# This must happen before anything else to prevent CPU-only torch from being pulled
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA is working BEFORE proceeding
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print(f'✓ PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Install Unsloth and Fix `torchao` Conflicts

```bash
# Step 4a: Install Unsloth WITHOUT letting it override PyTorch
# --no-deps prevents Unsloth from pulling conflicting torch versions
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps

# Step 4b: Install Unsloth's companion package
pip install unsloth-zoo --no-deps

# Step 4c: Install ONLY the HF dependencies we need, manually
# We install these AFTER torch is locked to prevent version conflicts
pip install transformers==4.46.3  # Stable version, no torchao auto-import
pip install trl peft accelerate bitsandbytes

# Step 4d: Install xformers compatible with torch 2.6.0 + CUDA 12.4
# If this fails, skip it — Unsloth works without xformers
pip install xformers --index-url https://download.pytorch.org/whl/cu124

# Step 4e: Nuke torchao BEFORE it can cause the register_constant crash
# This MUST happen after transformers install but before first Unsloth import
pip uninstall torchao -y
pip cache purge

# Step 4f: Final verification
python -c "from unsloth import FastLanguageModel; print('✓ Unsloth imported successfully')"
```

# Install utilities
pip install wandb pandas pyyaml
```

### 5. HuggingFace Login

You need to authenticate to download models and push your final LoRA adapters.

```bash
hf auth login
```

_Paste your HuggingFace token (must have Write access) when prompted._

---

## Phase 4: Transferring Your Files

To run the training, you need to copy your `datasets` and `training` folders to the EC2 server.

Open a **new, local terminal** (not connected via SSH) on your Windows machine, navigate to the `Enterprise AI Gateway` project folder, and run:

```bash
# Create the training directory on the server
ssh -i "path/to/origo-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP> "mkdir -p ~/training"

# Only copy the essential training code (ignores data generation scripts and markdown files)
scp -i "path/to/origo-key.pem" -r training/layer1_router training/layer2_specialist training/shared ubuntu@<YOUR-EC2-PUBLIC-IP>:~/training/

# Create the dataset directory on the server
ssh -i "path/to/origo-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP> "mkdir -p ~/datasets"

# Only copy the 'formatted' datasets (ignores heavy raw data and virtual environments)
scp -i "path/to/origo-key.pem" -r datasets/formatted ubuntu@<YOUR-EC2-PUBLIC-IP>:~/datasets/
```

---

## You are Ready to Train!

Once your files are transferred, reconnect to your EC2 instance, activate your environment (`conda activate origo-train`), navigate to the `training/layer1_router` directory, and begin!
