import os

def update_hardware_config(config):
    # Detect SLURM-specific environment variables for HPC
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", 4))  # Default to 4 if not set
    num_gpus = int(os.getenv("SLURM_GPUS", 0))  # Detect GPUs assigned by SLURM
    gpu_ids = os.getenv("SLURM_JOB_GPUS", None)  # List of GPU IDs
    
    # Update hardware-related settings in config
    config['hardware']['num_workers'] = num_cpus
    config['hardware']['device'] = "gpu" if num_gpus > 0 else "cpu"
    config['hardware']['num_gpus'] = num_gpus

    print(f"Updated config: {config['hardware']}")
    return config