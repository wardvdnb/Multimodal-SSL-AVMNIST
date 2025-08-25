#!/bin/bash
#SBATCH --output="/scratch/brussel/111/vsc11197/AVMNIST/debugging/experiment_%A_%a.out"
#SBATCH --error="/scratch/brussel/111/vsc11197/AVMNIST/debugging/experiment_%A_%a.err"
#SBATCH --partition='zen4' 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00

# Model selection (change this to the desired model)
MODEL="multimodal_dino"  # Options: unimodal_image, unimodal_audio, centralnet, multimodal_dino

# NOTE: Out of memory, or slow -> scale up
# leave cpus per task at 8
# load modules
# to find the modules use `module spider MODULENAME`
module load PyTorch-bundle/2.1.2-foss-2023a Lightning/2.2.1-foss-2023a Optuna/3.5.0-foss-2023a Seaborn/0.13.2-gfbf-2023a SciPy-bundle/2023.07-gfbf-2023a


python3 ${VSC_HOME}/AVMNIST/run_dino.py --model "$MODEL"
