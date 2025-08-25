from models.dino import UniModalDINOLightning
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from utils.get_data import AVMNISTDataModule, AVMNISTDinoDataModule, MultiModalAugmentation

def process_augment_config(trial, config, is_hyperparameter_search=True):
    """
    Process augmentation configuration for either:
    - Hyperparameter search (using Optuna trial)
    - Final training (using best_augments from config)

    Returns:
        {
            'augmentations': nested dict of augmentation args (without p),
            'augmentation_probabilities': nested dict of augmentation probabilities (p only)
        }
    """
    if is_hyperparameter_search:
        augmentations = {'global_views': {}, 'local_views': {}}
        augmentation_probabilities = {'global_views': {}, 'local_views': {}}

        for view_setting in ['global_views', 'local_views']:
            view_params = config["optuna"]["augmentations"][view_setting].items()
            for aug, params in view_params:
                aug_params = {}
                aug_prob = None

                for param_name, param_info in params.items():
                    if param_name == "p":
                        aug_prob = trial.suggest_float(
                            f"{view_setting}.{aug}.p",
                            param_info["low"],
                            param_info["high"]
                        )
                        augmentation_probabilities[view_setting][aug] = aug_prob

                    elif param_info["type"] == "uniform":
                        aug_params[param_name] = trial.suggest_float(
                            f"{view_setting}.{aug}.{param_name}",
                            param_info["low"],
                            param_info["high"]
                        )

                    elif param_info["type"] == "int":
                        aug_params[param_name] = trial.suggest_int(
                            f"{view_setting}.{aug}.{param_name}",
                            param_info["low"],
                            param_info["high"],
                            step=param_info.get("step", 1)
                        )

                    elif param_info["type"] == "categorical":
                        aug_params[param_name] = trial.suggest_categorical(
                            f"{view_setting}.{aug}.{param_name}",
                            param_info["choices"]
                        )

                    else:
                        raise ValueError(f"Unknown parameter type: {param_info['type']} for {param_name}")

                if aug_params:
                    augmentations[view_setting][aug] = aug_params

        return {
            "augmentations": augmentations,
            "augmentation_probabilities": augmentation_probabilities
        }
    else:
        # Final training mode - use fixed best_augments
        if "best_augments" not in config:
            raise ValueError("best_augments not found in config for final training")

        augmentations = {
            'global_views': {},
            'local_views': {}
        }

        augmentation_probabilities = {
            'global_views': {},
            'local_views': {}
        }

        for view_setting in ['global_views', 'local_views']:
            for aug, params in config["best_augments"][view_setting].items():
                aug_params = {k: v for k, v in params.items() if k != 'p'}
                if aug_params:
                    augmentations[view_setting][aug] = aug_params
                if 'p' in params:
                    augmentation_probabilities[view_setting][aug] = params['p']

        return {
            "augmentations": augmentations,
            "augmentation_probabilities": augmentation_probabilities
        }
    
def objective(trial, config, model_dir, model):
    try:
        augment_values = process_augment_config(trial, config, is_hyperparameter_search=True) # IMPORTANT, set to TRUE
        multimodal_augments = MultiModalAugmentation(augment_values=augment_values)
        batch_size=config['hyperparameters']['batch_size']
        n_global_views=config['hyperparameters'].get('n_global_views', 2)  # Default to 2 if not specified
        n_local_views=config['hyperparameters'].get('n_local_views', 4)   # Default to 4 if not specified
        
        # Initialize data module
        data = AVMNISTDinoDataModule(
            data_dir=config['data']['data_dir'], 
            num_workers=config['hardware']['num_workers'], 
            batch_size=batch_size,
            n_global_views=n_global_views,  # Default to 2 if not specified
            n_local_views=n_local_views,     # Default to 4 if not specified
            augmentations=multimodal_augments,
            type=config['hyperparameters']['data_augmentation']  # Default to "burst_noise" if not specified
        )

        logger = CSVLogger(f"{model_dir}", name="logs")
        logger.log_hyperparams({
            "batch_size": batch_size,
            "epochs_per_trial": config['optuna']['epochs_per_trial'],
            "num_global_views": n_global_views,
            "num_local_views": n_local_views,
        })
        
        mode = "max" if config['hyperparameters']['metric'] == 'mlp_acc' else "min"
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=config['hyperparameters']['metric']) # For pruning, the mode is inferred from the optuna study's direction
        early_stopping = EarlyStopping(monitor=config['hyperparameters']['metric'], patience=5, mode=mode)
        
        trainer = pl.Trainer(
            max_epochs=config['hyperparameters']['num_epochs'],
            logger=logger,
            callbacks=[pruning_callback, early_stopping],
            enable_progress_bar=False,
            devices="auto",
            strategy="ddp" if config['hardware']['num_gpus'] > 1 else "auto",
        )

        trainer.fit(model, data)
        
        return trainer.callback_metrics[config['hyperparameters']['metric']].item()
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        return float('nan')
