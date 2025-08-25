import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from utils.get_data import AVMNISTDinoDataModule
from models.dino import MultiModalDINOLightning, UniModalDINOLightning
import inspect

def objective(trial, config, model_dir, encoder_class, is_unimodal_model=False):
    try:
        # Assuming `config` contains your YAML hyperparameter configuration
        batch_size = trial.suggest_int(
            "batch_size", 
            config["optuna"]["batch_size"]["low"], 
            config["optuna"]["batch_size"]["high"], 
            step=config["optuna"]["batch_size"]["step"]
        )
        learning_rate = trial.suggest_float(
            "learning_rate", 
            float(config["optuna"]["learning_rate"]["low"]),
            float(config["optuna"]["learning_rate"]["high"]),
            log=True  # loguniform requires log scaling
        )
        projection_dim = trial.suggest_int(
            "projection_dim", 
            config["optuna"]["projection_dim"]["low"], 
            config["optuna"]["projection_dim"]["high"], 
            step=config["optuna"]["projection_dim"]["step"]
        )
        output_dim = trial.suggest_int(
            "output_dim", 
            config["optuna"]["output_dim"]["low"], 
            config["optuna"]["output_dim"]["high"], 
            step=config["optuna"]["output_dim"]["step"]
        )
        encoder_output_dim = trial.suggest_int(
            "encoder_output_dim", 
            config["optuna"]["encoder_output_dim"]["low"], 
            config["optuna"]["encoder_output_dim"]["high"], 
            step=config["optuna"]["encoder_output_dim"]["step"]
        )
        momentum = trial.suggest_float(
            "momentum", 
            float(config["optuna"]["momentum"]["low"]), 
            float(config["optuna"]["momentum"]["high"])
        )
        center_momentum = trial.suggest_float(
            "center_momentum", 
            float(config["optuna"]["center_momentum"]["low"]), 
            float(config["optuna"]["center_momentum"]["high"])
        )
        n_global_views = trial.suggest_int(
            "n_global_views", 
            config["optuna"]["n_global_views"]["low"], 
            config["optuna"]["n_global_views"]["high"], 
            step=config["optuna"]["n_global_views"]["step"]
        )
        n_local_views = trial.suggest_int(
            "n_local_views", 
            config["optuna"]["n_local_views"]["low"], 
            config["optuna"]["n_local_views"]["high"], 
            step=config["optuna"]["n_local_views"]["step"]
        )
        student_temperature = trial.suggest_float(
            "student_temperature", 
            float(config["optuna"]["student_temperature"]["low"]), 
            float(config["optuna"]["student_temperature"]["high"]), 
            log=True  # loguniform scale for small, sensitive ranges
        )
        teacher_temperature = trial.suggest_float(
            "teacher_temperature", 
            float(config["optuna"]["teacher_temperature"]["low"]), 
            float(config["optuna"]["teacher_temperature"]["high"]), 
            log=True  # loguniform scale for small, sensitive ranges
        )
        weight_decay = trial.suggest_float(
            "weight_decay", 
            float(config["optuna"]["weight_decay"]["low"]), 
            float(config["optuna"]["weight_decay"]["high"]), 
            log=True  # loguniform for better control over small values
        )
        dropout = trial.suggest_float(
            "dropout", 
            float(config["optuna"]["dropout"]["low"]), 
            float(config["optuna"]["dropout"]["high"])
        )

        if is_unimodal_model:
            model: pl.LightningModule = UniModalDINOLightning(
                data_dir=config['data']['data_dir'],  # keeping this unchanged, since it's not part of Optuna search
                encoder_class=encoder_class,
                data_augmentation=config['hyperparameters']['data_augmentation'],
                projection_dim=projection_dim,
                output_dim=output_dim,
                momentum=momentum,
                center_momentum=center_momentum,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
                learning_rate=learning_rate,
                cosine_loss_alpha=config['hyperparameters']['cosine_loss_alpha'],  # left as is if it's not tuned by Optuna
                num_epochs=config['optuna']['epochs_per_trial'],
                weight_decay=weight_decay,
                dropout=dropout,
            )
        else:
            model: pl.LightningModule = MultiModalDINOLightning(
                data_dir=config['data']['data_dir'],  # keeping this unchanged, since it's not part of Optuna search
                data_augmentation=config['hyperparameters']['data_augmentation'],
                encoder_class=encoder_class,
                projection_dim=projection_dim,
                output_dim=output_dim,
                encoder_output_dim=encoder_output_dim,
                momentum=momentum,
                center_momentum=center_momentum,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
                learning_rate=learning_rate,
                use_mixed_precision=True,
                num_epochs=config['optuna']['epochs_per_trial'],
                weight_decay=weight_decay,
                dropout=dropout,
            )

        # Initialize data module
        data = AVMNISTDinoDataModule(
            data_dir=config['data']['data_dir'], 
            num_workers=config['hardware']['num_workers'], 
            batch_size=batch_size,
            n_global_views=n_global_views,
            n_local_views=n_local_views,
            type=config['hyperparameters'].get('data_augmentation', 'burst_noise')
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