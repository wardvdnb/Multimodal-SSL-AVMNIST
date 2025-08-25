#!/bin/bash
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=gpu_check.out
#SBATCH --error=gpu_check.err

# load modules
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 Lightning/2.2.1-foss-2023a-CUDA-12.1.1 Optuna/3.5.0-foss-2023a Seaborn/0.13.2-gfbf-2023a SciPy-bundle/2023.07-gfbf-2023a Braindecode/0.8.1-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1

python gpu_check.py
