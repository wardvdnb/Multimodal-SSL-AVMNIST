#!/bin/bash
#SBATCH --partition='pascal_gpu,ampere_gpu' 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem=16G
#SBATCH --time=72:00:00

# load modules
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 Lightning/2.2.1-foss-2023a-CUDA-12.1.1 Optuna/3.5.0-foss-2023a Seaborn/0.13.2-gfbf-2023a SciPy-bundle/2023.07-gfbf-2023a Braindecode/0.8.1-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Arguments
MODEL=$1
TRAINING_MODE=$2
CONFIG=$3
METRIC=$4
HYPERPARAMETER_TUNE=$5
HYPERPARAMETER_TUNE_AUGMENTS=$6

# Define lists of models
MULTIMODAL_MODELS=("multi_simple" "multi_simple_gated" "multi_lstm" "multi_vit" "multi_dual_vit" "multi_mobile_vit" "multi_resnet" "multi_cross_attention" "multi_central")
UNIMODAL_MODELS=("image_simple" "spectrogram_simple" "spectrogram_central" "spectrogram_lstm" "spectrogram_resnet" "spectrogram_vit" "spectrogram_mobile_vit")

# Initialize command
CMD="python3 ${VSC_HOME}/AVMNIST/run_dino.py"

# Detect whether model is unimodal or multimodal
if [[ " ${MULTIMODAL_MODELS[*]} " == *" $MODEL "* ]]; then
    CMD="$CMD --model $MODEL"
elif [[ " ${UNIMODAL_MODELS[*]} " == *" $MODEL "* ]]; then
    CMD="$CMD --unimodal_model $MODEL"
else
    echo "Error: Unknown model '$MODEL'. Not in multimodal or unimodal list."
    exit 1
fi

# Append config, metric, and training mode
CMD="$CMD --config $CONFIG --metric $METRIC --training_mode $TRAINING_MODE"

# Optional tuning flags
if [[ "$HYPERPARAMETER_TUNE" == "true" ]]; then
    CMD="$CMD --hyperparameter_tune"
fi

if [[ "$HYPERPARAMETER_TUNE_AUGMENTS" == "true" ]]; then
    CMD="$CMD --hyperparameter_tune_augments"
fi

# Run command
echo "Running command: $CMD"
$CMD
