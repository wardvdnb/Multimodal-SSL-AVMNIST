import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from utils.get_data import AVMNISTDataModule

def objective(trial, config, ModelClass):
    learning_rate = trial.suggest_float(
        "lr", float(config["optuna"]["lr"]["low"]), float(config["optuna"]["lr"]["high"]), log=True
    )
    batch_size = trial.suggest_int(
        "batch_size", 
        config["optuna"]["batch_size"]["low"], 
        config["optuna"]["batch_size"]["high"], 
        step=config["optuna"]["batch_size"]["step"]
    )
    dropout = trial.suggest_float(
        "dropout", 
        config["optuna"]["dropout"]["low"], 
        config["optuna"]["dropout"]["high"]
    )

    model: pl.LightningModule = ModelClass(
        num_classes=config['model']['num_classes'], 
        dropout_prob=dropout, 
        learning_rate=learning_rate
    )
    
    data = AVMNISTDataModule(
        data_dir=config['data']['data_dir'], 
        num_workers=config['hardware']['num_workers'], 
        batch_size=batch_size
    )

    logger = CSVLogger(config['logs']['log_dir'], name=f"{config['model']['name']}")
    logger.log_hyperparams({
        "lr": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout
    })
    
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor=config['optuna']['metric'])
    early_stopping = EarlyStopping(monitor=config['optuna']['metric'], patience=3, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=config['hyperparameters']['num_epochs'],
        logger=logger,
        callbacks=[pruning_callback, early_stopping],
        devices="auto",
        strategy="ddp" if config['hardware']['num_gpus'] > 1 else "auto",
    )

    trainer.fit(model, data)
    
    return trainer.callback_metrics[config['optuna']['metric']].item()

