import os
from utils.reproducibility import set_seed
set_seed()
import shutil
import yaml
import argparse
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import copy

from training_structures.dino_train import compute_classification_metrics, train_downstream, train_knn_classifier
from configs.update_config import update_hardware_config
from models.dino import CrossAttentionMultiModalEncoder, DualViTMultiModalEncoder, \
GatedMultiModalEncoder, LSTMMultiModalEncoder, MobileViTMultiModalEncoder, \
MultiModalDINOLightning, ResNetMultiModalEncoder, SimpleMultiModalEncoder, UniModalDINOLightning, \
ViTMultiModalEncoder, CentralMultiModalEncoder, SpectrogramEncoder, \
SpectrogramEncoderCentral, SpectrogramEncoderLSTM, SpectrogramEncoderResNet, SpectrogramEncoderViT, \
SpectrogramEncoderMobileViT, ImageEncoder, MultiModalDINOSemiSupervisedLightning, \
MultiModalDINOWithINFONCELightning, MultiModalDINOWithMSELightning

from utils.get_data import AVMNISTDataModule, AVMNISTDinoDataModule, AVMNISTDinoDataModuleExtended, MultiModalAugmentation
from datetime import datetime
from pathlib import Path
import optuna

import time
import torch
torch.set_float32_matmul_precision("high")

import numpy as np
from torchinfo import summary # For calculating GFLOPs
from utils.plots_trials import create_plots_for_study, load_all_versions, save_versions_to_csv, plot_loss
from utils.visualisations import pca_plot_dataloaders, pca_plot_multiclass, visualize_prediction_matrix, tsne_plot_multiclass
from optuna.storages import RetryFailedTrialCallback, RDBStorage
from hyperparameter_tuning.objective_dino import objective
from hyperparameter_tuning.objective_augment import objective as objective_augment
from hyperparameter_tuning.objective_augment import process_augment_config

def search_augmentation_hyperparameters(config, config_name, model_dir, model):
    direction = 'maximize' if config['hyperparameters']['metric'] == 'mlp_acc' else 'minimize'

    storage = RDBStorage(
        url=f"sqlite:///{model_dir}/optuna_studies.db",  # Use a persistent SQLite database
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3)
    )

    augment_names_global = ['time_warp', 'frequency_mask', 'time_mask', 'gaussian_noise', 
                    'random_resized_crop', 'random_affine']
    augment_names_local = augment_names_global + ['grouped_masking']

    # Define constraints for augmentation probabilities
    def constraints(trial):
        params = trial.params
        constraints = []
        
        # Ensure local view probabilities >= global view probabilities
        for aug in augment_names_global:
            if f'global_views.{aug}.p' in params and f'local_views.{aug}.p' in params:
                constraints.append(params[f'local_views.{aug}.p'] >= params[f'global_views.{aug}.p'])
        
        return constraints

    study = optuna.create_study(
        study_name=config['model']['name'],
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(constraints_func=constraints)
    )

    try:
        # Calculate remaining trials with timeout
        remaining_trials = config['optuna']['n_trials'] - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        if remaining_trials > 0:
            # Reduce n_jobs if still facing issues
            study.optimize(
                lambda trial: objective_augment(trial, config, model_dir, model), 
                n_trials=remaining_trials, 
                n_jobs=config['optuna']['num_parallel_trials'],
                timeout=config['optuna'].get('study_timeout', 86400),
                catch=(Exception,)  # Catch all exceptions to prevent study failure
            )
        
        # Update best_augments with nested structure
        if len(study.best_trials) > 0:
            best_params = study.best_trial.params
            
            # Initialize nested structure if not exists
            if 'best_augments' not in config:
                config['best_augments'] = {'global_views': {}, 'local_views': {}}
            
            # Process global views
            config['best_augments']['global_views'] = {
                aug: {
                    **{k.split('.')[-1]: v 
                     for k, v in best_params.items() 
                     if k.startswith(f'global_views.{aug}') and k.split('.')[-1] != 'p'},
                    'p': best_params.get(f'global_views.{aug}.p', 0.0)
                }
                for aug in augment_names_global
                if any(k.startswith(f'global_views.{aug}') for k in best_params)
            }
            
            # Process local views
            config['best_augments']['local_views'] = {
                aug: {
                    **{k.split('.')[-1]: v 
                     for k, v in best_params.items() 
                     if k.startswith(f'local_views.{aug}') and k.split('.')[-1] != 'p'},
                    'p': best_params.get(f'local_views.{aug}.p', 0.0)
                }
                for aug in augment_names_local
                if any(k.startswith(f'local_views.{aug}') for k in best_params)
            }
            
            if config['optuna']['save_hyperparameters']:
                with open(config_name, 'w') as f:
                    yaml.safe_dump(config, f)
    
    except Exception as e:
        print(f"Study optimization failed with error: {str(e)}")
        # Save current best trial if available
        if len(study.trials) > 0:
            best_trial = study.best_trial
            if best_trial:
                # Same update logic as above
                pass
    
    return study

def search_hyperparameters(config, config_name, model_dir, encoder_class, is_unimodal_model=False):
    direction = 'maximize' if config['hyperparameters']['metric'] == 'mlp_acc' else 'minimize'
    
    # Switch from SQLite to JournalFileStorage (apparently more stable for parallel optimization)
    # storage = JournalStorage(JournalFileStorage(f"{model_dir}/optuna_journal_storage.log"), )

    storage = RDBStorage(
        url=f"sqlite:///{model_dir}/optuna_studies.db",  # Use a persistent SQLite database
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3)
    )

    study = optuna.create_study(
        study_name=config['model']['name'],
        storage=storage,
        direction=direction,
        load_if_exists=True
    )

    # # Print details of all trials
    # for trial in study.trials:
    #     print(f"Trial {trial.number}: State = {trial.state}, Params = {trial.params}")

    try:
        # Calculate remaining trials with timeout
        remaining_trials = config['optuna']['n_trials'] - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        if remaining_trials > 0:
            # Reduce n_jobs if still facing issues
            study.optimize(
                lambda trial: objective(trial, config, model_dir, encoder_class, is_unimodal_model), 
                n_trials=remaining_trials, 
                n_jobs=config['optuna']['num_parallel_trials'],
                timeout=config['optuna'].get('study_timeout', 86400)  # 24 hour default
            )
        
        # Only update config if we have successful trials
        if len(study.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            config['hyperparameters'].update(study.best_params)
            
            if config['optuna']['save_hyperparameters']:
                with open(config_name, 'w') as f:
                    yaml.safe_dump(config, f)

    except Exception as e:
        print(f"Study optimization failed with error: {str(e)}")
        # Save current state before exiting
        if len(study.trials) > 0:
            config['hyperparameters'].update(study.best_params)
            if config['optuna']['save_hyperparameters']:
                with open(config_name, 'w') as f:
                    yaml.safe_dump(config, f)
    return study

class ModelStatsCallback(pl.Callback):
    """Callback to track model statistics including runtime and parameters."""
    def __init__(self):
        super().__init__()
        self.batch_times = []
        self.epoch_start_time = None
        self.training_start_time = None
        
    def on_train_start(self, trainer, pl_module):
        self.training_start_time = time.time()
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        self.batch_times = []
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        avg_batch_time = np.mean(self.batch_times) if self.batch_times else 0
        pl_module.log("epoch_time", epoch_time)
        pl_module.log("avg_batch_time", avg_batch_time)
        
    def on_train_end(self, trainer, pl_module):
        total_training_time = time.time() - self.training_start_time
        # Log directly to the logger
        if pl_module.logger:
            pl_module.logger.log_metrics({"total_training_time": total_training_time})
        else:
            print(f"Total training time: {total_training_time:.2f} seconds")  # Fallback for no logger

def move_to_device(obj, device):
    """ 
    Helper function for calculate_gflops:
    Recursively move tensors, lists, tuples, and dicts to the specified device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj  # handle None, int, etc.

def calculate_gflops(model, datamodule, is_extended_data_module=False):
    """
    Calculate GFLOPs using the first real batch from the dataloader.
    Assumes model.forward(batch)
    """
    # Save current mode (train/eval)
    was_training = model.training
    model.eval()
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    batch = next(iter(loader))
    if is_extended_data_module:
        # remove the third element (which is labels)
        batch = (batch[0], batch[1], batch[3])  # (image, audio, (global_images, global_audios, local_images, local_audios))

    # Move everything to model's device
    device = next(model.parameters()).device
    batch_on_device = move_to_device(batch, device)

    with torch.no_grad():
        model_summary = summary(
            model,
            input_data=(batch_on_device,),  # <- Important: tuple of a tuple
            col_names=["output_size", "num_params", "mult_adds"],
            verbose=0,
            device=device
        )

    # Restore training mode if it was previously on
    if was_training:
        model.train()

    # Normalize by batch size to get per-sample GFLOPs
    batch_size = batch_on_device[0].shape[0]
    gflops = model_summary.total_mult_adds / 1e9 / batch_size
    params = model_summary.total_params

    return gflops, params

def experiment(config, model, ModelClass, model_name, model_dir_scratch, 
               model_dir_data, extended_data_module=False, study=None):
    """ 
    Run the main experiment loop for training and evaluation.
    This function handles the setup, training, and evaluation of the model.
    Args:
        config (dict): Configuration dictionary containing hyperparameters and settings.
        model (pl.LightningModule): The model to train.
        ModelClass (type): The class of the lightning model being trained.
        model_name (str): Name of the model for logging and saving.
        model_dir_scratch (str): Directory for saving model checkpoints and logs.
        model_dir_data (str): Directory for saving data-related outputs (like plots).
        extended_data_module (bool): Flag indicating if an extended data module is used.
        study (optuna.Study): Optuna study object for hyperparameter tuning, if applicable.
    Returns:
        None
    """
    initial_model_weights = copy.deepcopy(model.state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_augments = process_augment_config(None, config, is_hyperparameter_search=False) # IMPORTANT, set to FALSE
    multimodal_augments = MultiModalAugmentation(augment_values=best_augments)

    # Set data loaders for evaluation
    avmnist_data = AVMNISTDataModule(
        data_dir=config['data']['data_dir'], 
        num_workers=config['hardware']['num_workers'], 
        batch_size=config['hyperparameters']['batch_size'],
        type=config['hyperparameters'].get('data_augmentation', 'burst_noise'),
    )
    avmnist_data.setup()
    traindata, validdata, testdata = avmnist_data.train_dataloader(), avmnist_data.val_dataloader(), avmnist_data.test_dataloader()

    DinoDataModuleCls = AVMNISTDinoDataModuleExtended if extended_data_module else AVMNISTDinoDataModule
    data = DinoDataModuleCls(
        data_dir=config['data']['data_dir'], 
        num_workers=config['hardware']['num_workers'], 
        batch_size=config['hyperparameters']['batch_size'],
        n_global_views=config['hyperparameters'].get('n_global_views', 2),  # Default to 2 if not specified
        n_local_views=config['hyperparameters'].get('n_local_views', 4),     # Default to 4 if not specified
        type=config['hyperparameters'].get('data_augmentation', 'burst_noise'),
        augmentations=multimodal_augments  # Default to "burst_noise" if not specified
    )

    # Extract the augmentation configuration from the data module
    n_global_views = data.n_global_views
    n_local_views = data.n_local_views
    
    mode = "max" if config['hyperparameters']['metric'] == 'mlp_acc' else "min"
    
    # Set up callbacks (no early stopping, because we want to give the final model a chance to train)
    checkpoint = ModelCheckpoint(dirpath=model_dir_scratch, monitor=config['hyperparameters']['metric'], save_top_k=1, mode=mode)
    stats_callback = ModelStatsCallback()
    
    # Calculate GFLOPs
    gflops, params = calculate_gflops(
        model, 
        datamodule=data,
        is_extended_data_module=extended_data_module
    )
    
    # Log the view configuration used for GFLOPs calculation
    print(f"GFLOPs calculated with {n_global_views} global views and {n_local_views} local views")
    
    # Compute and save additional results
    seeds = [1, 2, 3]
    test_accuracies_knn = []
    test_accuracies_mlp = []

    for seed in seeds:
        print(f"Running seed: {seed}")
        set_seed(seed)
        model.load_state_dict(copy.deepcopy(initial_model_weights))
        logger = CSVLogger(f"{model_dir_scratch}", name=f"logs_seed{seed}")
        trainer = pl.Trainer(
            max_epochs=config['hyperparameters']['num_epochs'],
            devices="auto",
            strategy="ddp" if config['hardware']['num_gpus'] > 1 else "auto",
            precision='16-mixed',
            log_every_n_steps=10,
            logger=logger,
            callbacks=[checkpoint, stats_callback],
            deterministic=True
        )

        model.train()
        
        # Start training timer
        training_start = time.time()
        
        # Fit the model
        trainer.fit(model, data)
        
        # Calculate total training time
        training_time = time.time() - training_start
        
        # Save checkpoint
        trainer.save_checkpoint(f"{model_dir_scratch}/{model_name}.ckpt")
        
        # Collect stats from logs
        metrics = trainer.callback_metrics
        avg_batch_time = metrics.get("avg_batch_time", None)
        epoch_time = metrics.get("epoch_time", None)
        
        model = ModelClass.load_from_checkpoint(checkpoint.best_model_path)

        pretrained_dino = model.model.to(device) # Get the underlying pytorch model out of the pytorch lightning wrapper
        knn_acc, mlp_acc, mlp_classifier = \
            compute_accuracies(pretrained_dino, traindata, validdata, 
                               testdata, model_dir_scratch, model_name)
        test_accuracies_knn.append(knn_acc)
        test_accuracies_mlp.append(mlp_acc)

    knn_acc = np.mean(test_accuracies_knn)
    mlp_acc = np.mean(test_accuracies_mlp)
    knn_acc_std = np.std(test_accuracies_knn)
    mlp_acc_std = np.std(test_accuracies_mlp)

    # Aggregate results
    print(f"kNN Accuracy: {knn_acc:.2f} ± {knn_acc_std:.2f}")
    print(f"MLP Accuracy: {mlp_acc:.2f} ± {mlp_acc_std:.2f}")

    visualize_train_results(pretrained_dino, mlp_classifier, testdata, 
                            os.path.join(model_dir_scratch, f"logs_seed{seeds[-1]}"), 
                            model_dir_data, config)
    
    # Gather all results based on the last seed's training
    results = {
        "model": config["model"]["name"],
        "best_train_loss": trainer.callback_metrics.get("train_loss", None),
        "best_mlp_acc": trainer.callback_metrics.get("mlp_acc", None),
        "learning_rate": config["hyperparameters"]["learning_rate"],
        "batch_size": config["hyperparameters"]["batch_size"],
        "momentum": config["hyperparameters"]["momentum"],
        "center_momentum": config["hyperparameters"]["center_momentum"],
        "projection_dim": config["hyperparameters"]["projection_dim"],
        "output_dim": config["hyperparameters"]["output_dim"],
        "data_augmentation": config["hyperparameters"].get("data_augmentation", "burst_noise"),
        # View configuration
        "n_global_views": n_global_views,
        "n_local_views": n_local_views,
        # Performance metrics
        "gflops": gflops,
        "params": params,
        "params_millions": params / 1e6,
        "total_training_time": training_time,
        "avg_epoch_time": epoch_time,
        "avg_batch_time": avg_batch_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save detailed metrics
    df = pd.DataFrame([results])
    df.to_csv(os.path.join(model_dir_data, f"final_results_{config['model']['name']}.csv"), index=False)

    # Also save a performance summary file
    perf_summary = {
        "model_name": config["model"]["name"],
        "parameters": f"{params/1e6:.2f}M",
        "gflops": f"{gflops:.2f}",
        "n_global_views": n_global_views,
        "n_local_views": n_local_views,
        "training_time_hours": f"{training_time/3600:.2f}",
        "avg_epoch_time_minutes": f"{epoch_time/60:.2f}" if epoch_time else "N/A",
        f"best_train_loss": f"{float(trainer.callback_metrics.get('train_loss', 0)):.4f}",
        f"best_{config['hyperparameters']['metric']}": f"{float(trainer.callback_metrics.get(config['hyperparameters']['metric'], 0)):.4f}",
        "downstream_mlp_acc": f"{mlp_acc:.4f}",
        "downstream_knn_accuracy": f"{knn_acc:.4f}",
        "downstream_mlp_acc_std": f"{mlp_acc_std:.4f}",
        "downstream_knn_accuracy_std": f"{knn_acc_std:.4f}",
    }
    
    # === Extract learnable gate values and include in performance summary ===
    try:
        gate_audio = model.model.student.gate_audio.item()
        gate_image = model.model.student.gate_image.item()

        perf_summary["final_audio_gate"] = gate_audio
        perf_summary["final_image_gate"] = gate_image

    except AttributeError as e:
        print(f"[!] Failed to extract gate values: {e}")
        perf_summary["final_audio_gate"] = "N/A"
        perf_summary["final_image_gate"] = "N/A"

    with open(os.path.join(model_dir_data, "performance_summary.txt"), "w") as f:
        for key, value in perf_summary.items():
            f.write(f"{key}: {value}\n")
        
        # Append augmentation config
        if hasattr(data, "augmentations"):
            f.write("\n# Augmentation Summary\n")
            f.write(str(data.augmentations) + "\n")

    versions_path = os.path.join(model_dir_scratch, 'logs')
    plots_trials_path = os.path.join(model_dir_data, 'plots_trials')
    if study is not None:
        create_plots_for_study(study, versions_path=versions_path, plots_path=plots_trials_path)

def compute_accuracies(pretrained_dino, traindata, validdata, 
                       testdata, model_dir_scratch, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, knn_accuracy = train_knn_classifier(pretrained_dino, traindata, testdata, n_neighbors=5)
    
    mlp_classifier = train_downstream(
        pretrained_dino,
        traindata,
        validdata,
        testdata,
        num_epochs=10,
        device=device,
        save_path=f'{model_dir_scratch}/downstream/{model_name}.pt',
        train_log_path=f'{model_dir_scratch}/downstream/{model_name}_train_log.csv',
        test_log_path=f'{model_dir_scratch}/downstream/{model_name}_test_log.csv',
    )
    
    mlp_accuracy = compute_classification_metrics(mlp_classifier, testdata, device)['accuracy']

    return knn_accuracy, mlp_accuracy, mlp_classifier

def visualize_train_results(pretrained_dino, classifier, testdata, log_path, 
                            model_dir_data, config):
    """
    Save Plot results (e.g. pca for frozen encoders) and 
    Accuracies (for downstream tasks (KNN and MLP on 10 epochs))
    """
    pca_plot_path = os.path.join(model_dir_data, 'pca_plots')
    confusion_matrix_path = os.path.join(model_dir_data, 'confusion_matrix')

    _ = pca_plot_dataloaders(pretrained_dino, testdata, selected_digits=[5, 8], 
                             dirpath=pca_plot_path, show_plots=False)
    _ = pca_plot_multiclass(pretrained_dino, testdata, 
                            selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                            dirpath=pca_plot_path, show_plots=False)
    _ = tsne_plot_multiclass(pretrained_dino, testdata, 
                             selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                             dirpath=pca_plot_path, show_plots=False, 
                             random_seed=config['experiment']['seed'])
    _ = visualize_prediction_matrix(classifier, testdata, dirpath=confusion_matrix_path, 
                                    show_plots=False)
    
    metrics_df = load_all_versions(log_path)
    save_versions_to_csv(metrics_df, log_path)
    plot_loss(metrics_df, plot_dir=model_dir_data)

def main():
    
    MODEL_MAP = {
        "multi_simple": SimpleMultiModalEncoder,
        "multi_simple_gated": GatedMultiModalEncoder,
        "multi_lstm": LSTMMultiModalEncoder,
        "multi_vit": ViTMultiModalEncoder,
        "multi_dual_vit": DualViTMultiModalEncoder,
        "multi_mobile_vit": MobileViTMultiModalEncoder,
        "multi_resnet": ResNetMultiModalEncoder,
        "multi_cross_attention": CrossAttentionMultiModalEncoder,
        "multi_central": CentralMultiModalEncoder
    }

    UNIMODAL_MODEL_MAP = {
        "image_simple": ImageEncoder,
        "spectrogram_simple": SpectrogramEncoder,
        "spectrogram_central": SpectrogramEncoderCentral,
        "spectrogram_lstm": SpectrogramEncoderLSTM,
        "spectrogram_resnet": SpectrogramEncoderResNet,
        "spectrogram_vit": SpectrogramEncoderViT,
        "spectrogram_mobile_vit": SpectrogramEncoderMobileViT,
    }

    MULTIMODAL_WRAPPERS = {
        "default": MultiModalDINOLightning,
        "semi_supervised": MultiModalDINOSemiSupervisedLightning,
        "mse": MultiModalDINOWithMSELightning,
        "infonce": MultiModalDINOWithINFONCELightning,
    }

    # ------------------ IMPORTANT! SET CONFIG BEFORE RUNNING ------------------ 
    parser = argparse.ArgumentParser()

    # --- Mutually exclusive model selection ---
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, choices=list(MODEL_MAP.keys()), 
                            help=f"Select a multimodal model from: {list(MODEL_MAP.keys())}")
    model_group.add_argument("--unimodal_model", type=str, choices=list(UNIMODAL_MODEL_MAP.keys()), 
                            help=f"Select a unimodal model from: {list(UNIMODAL_MODEL_MAP.keys())}")
    
    # --- Optional training mode (only valid with --model) ---
    parser.add_argument(
        "--training_mode", type=str, default="default",
        choices=list(MULTIMODAL_WRAPPERS.keys()),
        help="Select training mode for multimodal models. Ignored if using a unimodal model."
    )

    parser.add_argument("--config", type=str, required=True, help="Config file name")
    parser.add_argument("--metric", type=str, default="mlp_acc", choices=["mlp_acc", "train_loss"],)
    parser.add_argument("--hyperparameter_tune", action="store_true", 
                        help="Run hyperparameter tuning (default: False)")
    parser.add_argument("--hyperparameter_tune_augments", action="store_true", 
                        help="Run hyperparameter tuning for augmentations (default: False)")
    args = parser.parse_args()
    
    if args.unimodal_model and args.training_mode != "default":
        raise ValueError(f"--training_mode '{args.training_mode}' is only compatible with --model (multimodal models).")

    if args.model:
        chosen_model = args.model
        ModelClass = MODEL_MAP[chosen_model]
        MultiModalLightningClass = MULTIMODAL_WRAPPERS.get(args.training_mode, MultiModalDINOLightning)
    else:
        chosen_model = args.unimodal_model
        ModelClass = UNIMODAL_MODEL_MAP[chosen_model]

    config_path = os.path.join(Path.cwd().parent, 'configs', args.config)

    config = yaml.safe_load(open(config_path))
    config = update_hardware_config(config)

    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    training_mode_name = f"_{args.training_mode}" if args.training_mode != "default" else ""
    model_name = f"{chosen_model}{training_mode_name}_{args.metric}_{timestamp}"
    model_dir_scratch = f"{config['model']['model_dir_scratch']}/{model_name}"
    model_dir_data = f"{config['model']['model_dir_data']}/{model_name}"

    for path in [model_dir_data, model_dir_scratch]:
        os.makedirs(path, exist_ok=True)

    # Work with a copied config file since hyperparameter tuning will modify it's values
    config_path = shutil.copy(config_path, os.path.join(model_dir_scratch, "config.yaml"))
    config = yaml.safe_load(open(config_path))
    config['model']['name'] = chosen_model
    config['hyperparameters']['metric'] = args.metric

    pl.seed_everything(config['experiment']['seed'], workers=True)

    if args.hyperparameter_tune and args.hyperparameter_tune_augments:
        print("Running both hyperparameter and augmentation tuning...")

    if args.hyperparameter_tune:
        if args.model: # multimodal
            study = search_hyperparameters(config, config_path, model_dir_scratch, ModelClass, is_unimodal_model=False)
        else:          # unimodal
            study = search_hyperparameters(config, config_path, model_dir_scratch, ModelClass, is_unimodal_model=True)
    else:
        study = None
    
    # hyperparameter search will have updated the config with the best params (under "hyperparameters")
    if args.model:
        model = MultiModalLightningClass(
            data_dir = config['data']['data_dir'], 
            data_augmentation = config['hyperparameters'].get('data_augmentation', 'burst_noise'),
            dino_model = None,
            encoder_class = ModelClass,
            encoder_kwargs = None,
            projection_dim = config['hyperparameters']['projection_dim'],
            output_dim = config['hyperparameters']['output_dim'],
            encoder_output_dim = config['hyperparameters']['encoder_output_dim'],
            momentum = config['hyperparameters']['momentum'],
            center_momentum = config['hyperparameters']['center_momentum'],
            student_temperature=config['hyperparameters']['student_temperature'],
            teacher_temperature=config['hyperparameters']['teacher_temperature'],
            learning_rate = config['hyperparameters']['learning_rate'],
            use_mixed_precision=True,
            num_epochs=config['hyperparameters']['num_epochs'],
            weight_decay=config['hyperparameters']['weight_decay'],
            dropout=config['hyperparameters']['dropout'],
        )
    else:
        model = UniModalDINOLightning(
            encoder_class = ModelClass,
            data_dir = config['data']['data_dir'], 
            dropout = config['hyperparameters']['dropout'],
            learning_rate = config['hyperparameters']['learning_rate'],
            projection_dim = config['hyperparameters']['projection_dim'],
            output_dim = config['hyperparameters']['output_dim'],
            momentum = config['hyperparameters']['momentum'],
            center_momentum = config['hyperparameters']['center_momentum'],
            teacher_temperature = config['hyperparameters']['teacher_temperature'],
            weight_decay = config['hyperparameters']['weight_decay'],
            cosine_loss_alpha = config['hyperparameters']['cosine_loss_alpha'],
            num_epochs = config['hyperparameters']['num_epochs'],
            data_augmentation = config['hyperparameters'].get('data_augmentation', 'burst_noise')
        )

    if args.hyperparameter_tune_augments:
        study = search_augmentation_hyperparameters(config, config_path, model_dir_scratch, model)

    needs_extended_data_module = True if (args.training_mode != "default" and not args.unimodal_model) else False
    LightningClass = UniModalDINOLightning if args.unimodal_model else MultiModalLightningClass
    
    experiment(config, model, LightningClass, model_name, model_dir_scratch, model_dir_data,
               extended_data_module=needs_extended_data_module, study=study)

if __name__ == '__main__':
    main()
