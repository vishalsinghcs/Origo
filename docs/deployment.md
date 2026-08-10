# AWS Deployment & Training Setup Guide

This guide will walk you through manually provisioning an AWS EC2 instance via the graphical user interface (GUI) and configuring the high-performance Unsloth training environment via SSH. 

We are targeting a **`g6.2xlarge`** instance, which utilizes the Nvidia L4 GPU (24GB VRAM) and 32GB of system RAM — the perfect cost-to-performance ratio for training an 8B parameter model.

---

## Phase 1: Launching the EC2 Instance (AWS GUI)

1. **Log into AWS**: Navigate to the [AWS Management Console](https://console.aws.amazon.com/) and search for **EC2**.
2. **Launch Instance**: Click the orange **Launch instance** button.
3. **Name your Server**: Under *Name and tags*, enter `Origo-Training-Server`.
4. **Choose the AMI (Operating System)**: 
   - Under *Application and OS Images*, search for: `Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04)`, it will be available in community.
   - *Why?* This AMI comes pre-loaded with the exact Nvidia drivers required for AI workloads, saving you from complex kernel installations.
5. **Select Instance Type**:
   - Under *Instance type*, search for and select **`g6.2xlarge`**. 
   - **Handling vCPU Quota Errors**: If you get a "vCPU limit of 0" error, your account needs permission to run GPU instances.
     1. Open a new tab and search for **Service Quotas** in the top AWS search bar.
     2. Click **AWS services** on the left, then click **Amazon Elastic Compute Cloud (Amazon EC2)**.
     3. Search for the quota named **Running On-Demand G and VT instances**.
     4. Select it, click **Request quota increase**, and request a value of **`8`** (which covers the 8 vCPUs needed for a `g6.2xlarge`). 
     *(Note: Approval can take from a few minutes to 24 hours. If denied, reply to the support ticket explaining you are a developer fine-tuning an AI model).*
6. **Create a Key Pair (Crucial for Access)**:
   - Click **Create new key pair**.
   - Name it `origo-key`.
   - Select **RSA** and **.pem**.
   - Click **Create key pair**. *The `.pem` file will immediately download to your computer. Keep it safe; you cannot download it again!*
7. **Configure Security Group (Networking)**:
   - Under *Network settings*, check the box for **Allow SSH traffic from**.
   - Set it to **My IP** (most secure) or **Anywhere (0.0.0.0/0)**.
8. **Configure Storage**:
   - Under *Configure storage*, change the root volume size to **`200` GB**. Leave the type as `gp3`. 
   - *Why 200GB?* Even though models are loaded in 4-bit (quantized) and datasets might be small, HuggingFace initially caches the unquantized 16-bit weights (taking ~25GB). Furthermore, Ubuntu, CUDA, and Miniconda environments consume ~20GB. Finally, saving multiple training checkpoints (which include optimizer states) and merging LoRA adapters requires duplicating model footprints on disk. Running out of storage will immediately crash your training job. At ~$0.08/GB, this 200GB buffer acts as a very cheap safety net.
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
*Close your SSH terminal and reconnect to apply the conda initialization.*

### 2. Create the Training Environment
Create an environment specifically built for PyTorch and Unsloth. 
*(Note: Anaconda recently introduced a prompt requiring users to accept their Terms of Service for free/personal use before creating environments).*
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n origo-train python=3.11 -y
conda activate origo-train
```

### 3. Install PyTorch (CUDA 12.1)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 4. Install Unsloth and HuggingFace Dependencies
```bash
# Install Unsloth natively
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install xformers for memory efficient attention
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Install utilities
pip install wandb pandas pyyaml
```

### 5. HuggingFace Login
You need to authenticate to download models and push your final LoRA adapters.
```bash
hf auth login
```
*Paste your HuggingFace token (must have Write access) when prompted.*

---

## Phase 4: Transferring Your Files

To run the training, you need to copy your `datasets` and `training` folders to the EC2 server. 

Open a **new, local terminal** (not connected via SSH) on your Windows machine, navigate to the `Enterprise AI Gateway` project folder, and run:

```bash
# Copy the training folder
scp -i "path/to/origo-key.pem" -r training ubuntu@<YOUR-EC2-PUBLIC-IP>:~/

# Copy the datasets folder
scp -i "path/to/origo-key.pem" -r datasets ubuntu@<YOUR-EC2-PUBLIC-IP>:~/
```

---

## You are Ready to Train!
Once your files are transferred, reconnect to your EC2 instance, activate your environment (`conda activate origo-train`), navigate to the `training/layer1_router` directory, and begin!
