# Implementation Plan: End-to-End Fine-Tuning Pipeline

This plan outlines the exact, chronological, step-by-step execution strategy to set up the AWS infrastructure **manually via the AWS GUI**, run the QLoRA fine-tuning for both models, evaluate their performance, and merge them for production.

## Recommended AWS Instance: `g6.2xlarge`
After researching the requirements for training an 8B model with Unsloth (which requires around ~12-16GB of VRAM during QLoRA) against current AWS pricing, I am firmly targeting the **`g6.2xlarge`** instance for this guide.
- **Cost-Effective**: At approximately **~$0.97 per hour**, it is significantly cheaper than standard A10G (`g5.2xlarge` at ~$1.21/hr) or A100 instances. Your $200 credits will easily stretch for over 200 hours of uptime.
- **Powerful GPU**: It features the newer Nvidia L4 GPU (24GB VRAM, Ada Lovelace architecture), which trains faster and more efficiently than older generations.
- **System Stability**: It provides 32GB of system RAM, ensuring your instance doesn't crash when initially loading the 8B model into memory before moving it to the GPU.

### Why Raw EC2 Instead of SageMaker for Training?
While AWS SageMaker is famous for managed machine learning, we are explicitly avoiding it for the *training* phase for three strategic reasons:
1. **The "Managed" Tax:** SageMaker charges a significant premium (often 20% to 40% more) on top of the raw hardware cost. Renting the raw EC2 `g6.2xlarge` directly gets you the absolute cheapest price possible.
2. **Loss of Control:** Unsloth (our training library) requires highly specific, custom-written GPU kernels (Triton) to achieve its speed. SageMaker tries to force you into using their pre-built "Docker Containers" which routinely conflict with Unsloth's strict CUDA and PyTorch requirements. Getting Unsloth to work in a SageMaker automated training job is notoriously difficult.
3. **Resume Value:** By provisioning a raw Linux EC2 instance, installing Conda, managing Nvidia drivers, and running the training manually via SSH, you demonstrate deep **MLOps Infrastructure skills**. Relying on SageMaker hides all of that under a GUI. 
*(Note: Using SageMaker Endpoints later for deployment/hosting is still a very valid enterprise choice!)*

---

## Execution Steps (Running on AWS)

Now that your EC2 instance is provisioned, the environment is configured, and your files are transferred (as per the `deployment.md` guide), follow these steps to execute the training.

### Step 1: Connect and Activate
First, SSH into your EC2 instance and activate the Conda environment you built:
```bash
ssh -i origo-key.pem ubuntu@<YOUR-EC2-PUBLIC-IP>
conda activate origo-train
```

### Step 2: Smoke Test (Dry Run)
Before committing to hours of training, it is crucial to ensure the GPU handles the batch sizes without Out-Of-Memory (OOM) errors. 
To do a smoke test, quickly open your config file using a terminal text editor like `nano`:
```bash
nano training/layer1_router/config.yaml
```
Temporarily change `num_train_epochs: 3` to `max_steps: 10` (or just let the script run for a minute and hit `Ctrl+C` to cancel it). If it starts printing loss numbers and your GPU doesn't crash, the setup is perfect. (Be sure to change it back to 3 epochs for the real run).

### Step 3: Train Layer 1 (Semantic Router)
Navigate to the Layer 1 folder and execute the script:
```bash
cd ~/training/layer1_router
python train.py
```
- **Monitoring:** Open a second SSH terminal to your EC2 instance and run `watch -n 1 nvidia-smi`. This will show you a live, refreshing view of your GPU memory and utilization.
- **Output:** Once finished, the final LoRA adapters will be saved in `~/training/layer1_router/checkpoints/final`.

### Step 4: Train Layer 2 (Specialist Rewriter)
Once Layer 1 is complete, move to the Layer 2 folder:
```bash
cd ~/training/layer2_specialist
python train.py
```
Because this model is larger (8B vs 3B), `nvidia-smi` will show significantly higher VRAM usage. It will take longer to train.

### Step 5: Download Checkpoints Safely
When training is complete, you MUST download the final adapters back to your local Windows machine before terminating the EC2 instance to ensure you don't lose them.

Run this in a **local Windows PowerShell** terminal (not SSH):
```powershell
scp -i origo-key.pem -r ubuntu@<YOUR-EC2-PUBLIC-IP>:~/training/layer1_router/checkpoints/final ./layer1_final
scp -i origo-key.pem -r ubuntu@<YOUR-EC2-PUBLIC-IP>:~/training/layer2_specialist/checkpoints/final ./layer2_final
```

> [!WARNING]
> After downloading your checkpoints safely, remember to **STOP** your EC2 instance in the AWS Console to halt the ~$0.97/hr billing! Do not "Terminate" it unless you are entirely finished with the project, as terminating deletes the hard drive.