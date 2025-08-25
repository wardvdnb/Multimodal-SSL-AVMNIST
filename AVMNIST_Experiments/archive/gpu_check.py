import os
import torch

print("SLURM_CPUS_PER_TASK:", os.getenv("SLURM_CPUS_PER_TASK"))
print("SLURM_GPUS:", os.getenv("SLURM_GPUS"))
print("SLURM_JOB_GPUS:", os.getenv("SLURM_JOB_GPUS"))
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
