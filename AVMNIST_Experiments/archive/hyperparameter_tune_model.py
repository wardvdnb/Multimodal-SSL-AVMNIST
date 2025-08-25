import subprocess
import argparse
import time

models = [
    "multi_simple", 
    "multi_simple_gated", 
    "multi_lstm", 
    "multi_vit", 
    "multi_dual_vit",
    "multi_mobile_vit", 
    "multi_resnet", 
    "multi_cross_attention",
    "multi_central"
]

config_file = "config_multimodal_dino.yaml"

parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, default="mlp_acc", choices=["mlp_acc", "train_loss"],)
parser.add_argument("--model", type=str, default="multi_simple", choices=models, help="Specify a single model to run")
parser.add_argument("--hyperparameter_tune", action="store_true", help="Run hyperparameter tuning (default: False)")
parser.add_argument("--hyperparameter_tune_augments", action="store_true", help="Run hyperparameter tuning for augmentations (default: False)")
args = parser.parse_args()

# Timestamp
timestamp = time.strftime("%d%m%Y_%H%M%S")

# Dynamic output and error filenames
output_file = f"/scratch/brussel/111/vsc11197/AVMNIST/debugging/{args.model}_{args.metric}_{timestamp}.out"
error_file = f"/scratch/brussel/111/vsc11197/AVMNIST/debugging/{args.model}_{args.metric}_{timestamp}.err"

cmd = [
    "sbatch",
    f"--output={output_file}",
    f"--error={error_file}",
    "run_gpu.sbatch", 
    args.model, 
    config_file, 
    args.metric, 
    "true" if args.hyperparameter_tune else "false",
    "true" if args.hyperparameter_tune_augments else "false",
]
print("Submitting:", " ".join(cmd))
subprocess.run(cmd)