import subprocess
import argparse
import time

# Example usage: python submit_models.py --models multi_vit multi_resnet --training_mode semi_supervised --hyperparameter_tune_augments

# List of all models
all_models = [
    "multi_simple", 
    "multi_simple_gated", 
    "multi_lstm", 
    "multi_vit", 
    "multi_dual_vit",
    "multi_mobile_vit", 
    "multi_resnet", 
    "multi_cross_attention",
    "multi_central",
    "image_simple",
    "spectrogram_simple",
    "spectrogram_central",
    "spectrogram_lstm",
    "spectrogram_resnet",
    "spectrogram_vit",
    "spectrogram_mobile_vit"
]

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--models", 
    nargs="+", 
    choices=all_models,
    help="List of models to submit. If not specified, all models will be submitted."
)
parser.add_argument(
    "--training_mode",
    type=str, 
    default="default",
    choices=["default", "semi_supervised", "mse", "infonce"],
    help="Training mode to use (multimodal only)."
)
parser.add_argument(
    "--config", 
    type=str, 
    default="config_multimodal_dino.yaml", 
    help="Path to the config file."
)
parser.add_argument(
    "--metric", 
    type=str, 
    default="mlp_acc", 
    choices=["mlp_acc", "train_loss"], 
    help="Metric to track."
)
parser.add_argument(
    "--hyperparameter_tune", 
    action="store_true", 
    help="Run hyperparameter tuning."
)
parser.add_argument(
    "--hyperparameter_tune_augments", 
    action="store_true", 
    help="Run hyperparameter tuning for augmentations."
)
args = parser.parse_args()

# Use all models if none are specified
models_to_run = args.models if args.models else all_models

# Timestamp for filenames
timestamp = time.strftime("%d%m%Y_%H%M%S")

training_mode_name = "" if args.training_mode == "default" else f"_{args.training_mode}"

# Submit jobs
for model in models_to_run:
    output_file = f"/scratch/brussel/111/vsc11197/AVMNIST/debugging/{model}{training_mode_name}_{args.metric}_{timestamp}.out"
    error_file = f"/scratch/brussel/111/vsc11197/AVMNIST/debugging/{model}{training_mode_name}_{args.metric}_{timestamp}.err"

    cmd = [
        "sbatch",
        f"--output={output_file}",
        f"--error={error_file}",
        "run_gpu.sbatch",
        model,
        args.training_mode,
        args.config,
        args.metric,
        "true" if args.hyperparameter_tune else "false",
        "true" if args.hyperparameter_tune_augments else "false",
    ]

    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd)