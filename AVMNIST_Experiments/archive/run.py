import os
import yaml
import argparse
import pandas as pd
import lightning.pytorch as pl
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from configs.update_config import update_hardware_config
from models.unimodal import UnimodalImage, CentralUnimodalAudio as UnimodalAudio #TODO: change
from models.centralnet.centralnet import SimpleAV_CentralNet as CentralNet
from models.dino import MultiModalDINO as DINO
from utils.get_data import AVMNISTDataModule
from temp_files.objective import objective
from datetime import datetime

# Map model names to classes
MODEL_MAP = {
    "unimodal_image": [UnimodalImage, "config_unimodal_image.yaml"],
    "unimodal_audio": UnimodalAudio,
    "centralnet": CentralNet,
    "dino": DINO,
}

def search_hyperparameters(config, config_name, ModelClass):
    direction = 'maximize' if config['optuna']['metric'] == 'accuracy' else 'minimize'
    
    os.makedirs(config["optuna"]["study_dir"], exist_ok=True)
    storage = f'sqlite:///{config["optuna"]["study_dir"]}studies.db'  # SQLite backend
    
    study = optuna.create_study(
        study_name=config['optuna']['study_name'],
        storage=storage,
        direction=direction,
        load_if_exists=True
    )
    
    study.optimize(
        lambda trial: objective(trial, config, ModelClass), 
        n_trials=(config['optuna']['n_trials'] - len(study.trials)), 
        n_jobs=config['optuna']['num_parallel_trials']
    )
    
    config['hyperparameters'].update(study.best_params)
    
    if config['optuna']['save_hyperparameters']:
        with open(config_name, 'w') as f:
            yaml.safe_dump(config, f)
    
    return study

def experiment(config, ModelClass):
    model = ModelClass(
        num_classes=config['model']['num_classes'], 
        dropout_prob=config['hyperparameters']['dropout'],
        learning_rate=config['hyperparameters']['lr']
    )
    
    data = AVMNISTDataModule(
        data_dir=config['data']['data_dir'], 
        num_workers=config['hardware']['num_workers'], 
        batch_size=config['hyperparameters']['batch_size']
    )

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = CSVLogger(config['logs']['log_dir'], f"final_{config['model']['name']}_{timestamp}")
    checkpoint = ModelCheckpoint(dirpath=config["model"]["model_dir"], monitor='val_loss', save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=config['hyperparameters']['num_epochs'],
        devices="auto",
        strategy="ddp" if config['hardware']['num_gpus'] > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint],
        deterministic=True  # Ensures reproducibility
    )

    trainer.fit(model, data)
    trainer.save_checkpoint(f'{config["model"]["model_dir"]}/{config["model"]["name"]}.ckpt')

     # Save final experiment results to csv_dir
    if 'csv_dir' in config['csv']:
        results = {
            "model": config["model"]["name"],
            "best_val_loss": trainer.callback_metrics.get("val_loss", None),
            "best_accuracy": trainer.callback_metrics.get("accuracy", None),
            "learning_rate": config["hyperparameters"]["lr"],
            "batch_size": config["hyperparameters"]["batch_size"],
            "dropout": config["hyperparameters"]["dropout"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        df = pd.DataFrame([results])
        os.makedirs(config['csv']['csv_dir'], exist_ok=True)
        df.to_csv(os.path.join(config['csv']['csv_dir'], f"final_results_{config['model']['name']}.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()), 
                        help="Select the model to run: unimodal_image, unimodal_audio, centralnet, dino")
    args = parser.parse_args()

    ModelClass = MODEL_MAP[args.model][0]
    config_name = MODEL_MAP[args.model][1]

    config = yaml.safe_load(open(config_name))
    config = update_hardware_config(config)

    pl.seed_everything(config['experiment']['seed'], workers=True)
    
    study = search_hyperparameters(config, config_name, ModelClass) if config['experiment']['hyperparameter_search'] else None
    experiment(config, ModelClass)
    # create_plots(config, study)

if __name__ == '__main__':
    main()
