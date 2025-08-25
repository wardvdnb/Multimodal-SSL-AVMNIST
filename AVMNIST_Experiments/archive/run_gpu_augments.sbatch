#!/bin/bash
#SBATCH --output="/scratch/brussel/111/vsc11197/AVMNIST/debugging/experiment_%A_%a.out"
#SBATCH --error="/scratch/brussel/111/vsc11197/AVMNIST/debugging/experiment_%A_%a.err"
#SBATCH --partition='pascal_gpu,ampere_gpu' 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00

# load modules
# to find the modules use `module spider MODULENAME`
# NOTE: to use GPU we need to load the CUDA specific version
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 Lightning/2.2.1-foss-2023a-CUDA-12.1.1 Optuna/3.5.0-foss-2023a Seaborn/0.13.2-gfbf-2023a SciPy-bundle/2023.07-gfbf-2023a Braindecode/0.8.1-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1

export CUBLAS_WORKSPACE_CONFIG=:4096:8  # Ensures deterministic behavior see (https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility)

python3 ${VSC_HOME}/AVMNIST/run_dino_augments.py
