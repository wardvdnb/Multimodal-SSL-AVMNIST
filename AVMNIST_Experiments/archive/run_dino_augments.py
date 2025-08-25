import os
from pathlib import Path
import yaml
import pandas as pd
import lightning.pytorch as pl
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping

from configs.update_config import update_hardware_config
from hyperparameter_tuning.objective_augment import objective, process_augment_config
from datetime import datetime
import time
import torch
import numpy as np
import lightning.pytorch as pl
from torchinfo import summary # For calculating GFLOPs

from models.dino import UniModalDINOLightning, SpectrogramEncoderMobileViT
from utils.get_data import AVMNISTDinoDataModule, MultiModalAugmentation
from datetime import datetime
from utils.plots_trials import create_plots_for_study, load_all_versions, save_versions_to_csv, plot_loss
from utils.visualisations import pca_plot_dataloaders, pca_plot_multiclass, visualize_prediction_matrix
from utils.get_data import get_dataloader_augmented
from training_structures.dino_train import train_downstream, train_knn_classifier
from optuna.storages import RetryFailedTrialCallback, RDBStorage

def search_hyperparameters_unimodal(config, config_name, model_dir, model):
    direction = 'maximize' if config['optuna']['metric'] == 'mlp_acc' else 'minimize'

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{model_dir}/optuna_studies.db",
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3)
    )

    # Define constraints for augmentation probabilities
    def constraints(trial):
        params = trial.params
        constraints = []
        
        # Ensure local view probabilities >= global view probabilities
        for aug in ['time_warp', 'frequency_mask', 'time_mask', 'gaussian_noise']:
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
        remaining_trials = config['optuna']['n_trials'] - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        if remaining_trials > 0:
            study.optimize(
                lambda trial: objective(trial, config, model_dir, model), 
                n_trials=remaining_trials, 
                n_jobs=config['optuna']['num_parallel_trials'],
                timeout=config['optuna'].get('study_timeout', 86400),
                catch=(Exception,)  # Catch all exceptions to prevent study failure
            )
        
        # Properly update best_augments with nested structure
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
                for aug in ['time_warp', 'frequency_mask', 'time_mask', 'gaussian_noise']
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
                for aug in ['time_warp', 'frequency_mask', 'time_mask', 'gaussian_noise', 'grouped_masking']
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

def calculate_gflops(model, input_shape, n_global_views=2, n_local_views=4):
    """Calculate GFLOPs for a model given input shape and view configuration."""
    device = next(model.parameters()).device

    # For multimodal, create batches with correct numbers of views
    image_shape, audio_shape = input_shape

    # Create dummy inputs matching the actual view structure
    global_images = torch.randn(1, n_global_views, *image_shape).to(device)
    global_audios = torch.randn(1, n_global_views, *audio_shape).to(device)
    local_images = torch.randn(1, n_local_views, *image_shape).to(device)
    local_audios = torch.randn(1, n_local_views, *audio_shape).to(device)

    dummy_input = (global_images, global_audios, local_images, local_audios)

    # Perform profiling using torchinfo
    with torch.no_grad():
        model_summary = summary(
            model,
            input_data=(dummy_input,),  # Pass the dummy input directly
            col_names=["output_size", "num_params", "mult_adds"],
            verbose=0,
            device=device
        )

    # Extract GFLOPs and total params
    gflops = model_summary.total_mult_adds / 1e9  # Convert to GFLOPs
    params = model_summary.total_params

    return gflops, params

def experiment(config, model, model_name, model_dir_scratch, model_dir_data, study=None):
    best_augments = process_augment_config(None, config, is_hyperparameter_search=False) # IMPORTANT, set to FALSE
    multimodal_augments = MultiModalAugmentation(augment_values=best_augments)
    
    # Initialize data module
    data = AVMNISTDinoDataModule(
        data_dir=config['data']['data_dir'], 
        num_workers=config['hardware']['num_workers'], 
        batch_size=config['hyperparameters']['batch_size'],
        n_global_views=config['hyperparameters'].get('n_global_views', 2),  # Default to 2 if not specified
        n_local_views=config['hyperparameters'].get('n_local_views', 4),     # Default to 4 if not specified
        type=config['hyperparameters'].get('data_augmentation', 'burst_noise'),  # Default to "burst_noise" if not specified
        augmentations=multimodal_augments,
    )

    # Extract the augmentation configuration from the data module
    n_global_views = data.n_global_views
    n_local_views = data.n_local_views
    
    logger = CSVLogger(f"{model_dir_scratch}", name="logs")
    mode = "max" if config['optuna']['metric'] == 'mlp_acc' else "min"
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(dirpath=model_dir_scratch, monitor=config['optuna']['metric'], save_top_k=1, mode=mode)
    early_stopping = EarlyStopping(monitor=config['optuna']['metric'], patience=3, mode=mode)
    stats_callback = ModelStatsCallback()

    trainer = pl.Trainer(
        max_epochs=config['hyperparameters']['num_epochs'],
        devices="auto",
        strategy="ddp" if config['hardware']['num_gpus'] > 1 else "auto",
        precision='16-mixed',
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint, early_stopping, stats_callback],
        deterministic=True
    )
    
    # Default values
    input_shape = ((1, 28, 28), (1, 112, 112)) # Base shapes with batch dimension
    
    # Calculate GFLOPs with the correct view configuration
    gflops, params = calculate_gflops(
        model, 
        input_shape,
        n_global_views=n_global_views, 
        n_local_views=n_local_views
    )
    
    # Log the view configuration used for GFLOPs calculation
    print(f"GFLOPs calculated with {n_global_views} global views and {n_local_views} local views")

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
    
    # Gather all results
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

    # Compute and save additional results
    knn_acc, mlp_acc = compute_train_results(model, model_dir_scratch, model_dir_data, model_name, config)

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
        f"best_{config['optuna']['metric']}": f"{float(trainer.callback_metrics.get(config['optuna']['metric'], 0)):.4f}",
        "downstream_mlp_acc": f"{mlp_acc:.4f}",
        "downstream_knn_accuracy": f"{knn_acc:.4f}",
    }
    
    with open(os.path.join(model_dir_data, "performance_summary.txt"), "w") as f:
        for key, value in perf_summary.items():
            f.write(f"{key}: {value}\n")

    versions_path = os.path.join(model_dir_scratch, 'logs')
    plots_trials_path = os.path.join(model_dir_data, 'plots_trials')
    if study is not None:
        create_plots_for_study(study, versions_path=versions_path, plots_path=plots_trials_path)

def compute_train_results(module, model_dir_scratch, model_dir_data, model_name, config):
    """
    Save Plot results (e.g. pca for frozen encoders) and 
    Accuracies (for downstream tasks (KNN and MLP on 10 epochs))
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pca_plot_path = os.path.join(model_dir_data, 'pca_plots')
    confusion_matrix_path = os.path.join(model_dir_data, 'confusion_matrix')
    log_path = os.path.join(model_dir_scratch, 'logs')
    data_dir = config['data']['data_dir']
    data_augmentation = config['hyperparameters'].get('data_augmentation', 'burst_noise')

    pretrained_dino = module.model.to(device)

    traindata, validdata, testdata = get_dataloader_augmented(data_dir, type=data_augmentation, batch_size=128, num_workers=0)
    knn_model, knn_accuracy = train_knn_classifier(pretrained_dino, traindata, testdata, n_neighbors=5)

    classifier = train_downstream(
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
    _ = pca_plot_dataloaders(pretrained_dino, testdata, selected_digits=[5, 8], dirpath=pca_plot_path, show_plots=False)
    _ = pca_plot_multiclass(pretrained_dino, testdata, selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dirpath=pca_plot_path, show_plots=False)
    prediction_results = visualize_prediction_matrix(classifier, testdata, dirpath=confusion_matrix_path, show_plots=False)
    
    metrics_df = load_all_versions(log_path)
    save_versions_to_csv(metrics_df, log_path)
    plot_loss(metrics_df, plot_dir=model_dir_data)

    return knn_accuracy, prediction_results['accuracy']


def main():
    # ------------------ IMPORTANT! SET THIS BEFORE RUNNING ------------------ 
    config_path = os.path.join(Path.cwd().parent, 'configs', 'config_augments.yaml')
    encoder_class = SpectrogramEncoderMobileViT
    # ------------------------------------------------------------------------
    config = yaml.safe_load(open(config_path))
    config = update_hardware_config(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config['model']['name']}_{timestamp}"
    model_dir_scratch = f"{config['model']['model_dir_scratch']}/{model_name}"
    model_dir_data = f"{config['model']['model_dir_data']}/{model_name}"

    for path in [model_dir_data, model_dir_scratch]:
        os.makedirs(path, exist_ok=True)

    pl.seed_everything(config['experiment']['seed'], workers=True)

    # Model remains static throughout the search/experiment, however augments change dynamically based on config
    model = UniModalDINOLightning(
            encoder_class = encoder_class,
            data_dir=config['data']['data_dir'], 
            dropout=config['hyperparameters']['dropout'],
            learning_rate = config['hyperparameters']['learning_rate'],
            projection_dim = config['hyperparameters']['projection_dim'],
            output_dim = config['hyperparameters']['output_dim'],
            momentum = config['hyperparameters']['momentum'],
            center_momentum = config['hyperparameters']['center_momentum'],
            teacher_temperature=config['hyperparameters']['teacher_temperature'],
            weight_decay=config['hyperparameters']['weight_decay'],
            cosine_loss_alpha=config['hyperparameters']['cosine_loss_alpha'],
            num_epochs=config['hyperparameters']['num_epochs'],
            data_augmentation=config['hyperparameters'].get('data_augmentation', 'burst_noise')
        )
    
    study = search_hyperparameters_unimodal(config, config_path, model_dir_scratch, model) if config['experiment']['hyperparameter_search'] else None
    
    experiment(config, model, model_name, model_dir_scratch, model_dir_data, study)

if __name__ == '__main__':
    main()