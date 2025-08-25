import os
import yaml
import argparse
import pandas as pd
import lightning.pytorch as pl
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping

from configs.update_config import update_hardware_config
from models.dino import MultiModalDINOLightning, MultiModalDINOV2Lightning
from models.dino_vit import MultiModalViTDINOLightning
from utils.get_data import AVMNISTDinoDataModule
from hyperparameter_tuning.objective_dino import objective
from datetime import datetime

def experiment(config, model):
    data = AVMNISTDinoDataModule(
        data_dir=config['data']['data_dir'], 
        num_workers=config['hardware']['num_workers'], 
        batch_size=config['hyperparameters']['batch_size']
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger(config['logs']['log_dir'], f"final_{config['model']['name2']}_{timestamp}")
    checkpoint = ModelCheckpoint(dirpath=config["model"]["model_dir"], monitor=config['model']['metric'], save_top_k=1)
     # Set up callbacks
    early_stopping = EarlyStopping(monitor=config['model']['metric'], patience=3, mode="min")

    trainer = pl.Trainer(
        max_epochs=config['hyperparameters']['num_epochs'], # train on 100 epoch for final model
        devices="auto",
        strategy="ddp" if config['hardware']['num_gpus'] > 1 else "auto",
        precision='16-mixed',
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint, early_stopping],
        deterministic=True
    )
    
    trainer.fit(model, data)
    trainer.save_checkpoint(f'{config["model"]["model_dir"]}/{config["model"]["name2"]}.ckpt')
    
    if 'csv_dir' in config['csv']:
        results = {
            "model": config["model"]["name2"],
            "best_train_loss": trainer.callback_metrics.get("train_loss", None),
            "learning_rate": config["hyperparameters"]["lr"],
            "batch_size": config["hyperparameters"]["batch_size"],
            "momentum": config["hyperparameters"]["momentum"],
            "center_momentum": config["hyperparameters"]["center_momentum"],
            "projection_dim": config["hyperparameters"]["projection_dim"],
            "output_dim": config["hyperparameters"]["output_dim"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        df = pd.DataFrame([results])
        os.makedirs(config['csv']['csv_dir'], exist_ok=True)
        df.to_csv(os.path.join(config['csv']['csv_dir'], f"final_results_{config['model']['name2']}.csv"), index=False)

def main():
    model = MultiModalDINOV2Lightning()
    config_name = "config_dino_temp.yaml"

    config = yaml.safe_load(open(config_name))
    config = update_hardware_config(config)
    
    pl.seed_everything(config['experiment']['seed'], workers=True)

    experiment(config, model)

if __name__ == '__main__':
    main()
