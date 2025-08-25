# Thesis code - Multimodal Self-Supervised Learning

Accompanying code for my thesis on **Multimodal Self-Supervised Learning**.

## Steps to reproduce:
1. Install ``requirements.txt``
2. Go to ``Audio_Gen/FSDD/`` and run ``audio_gen.ipynb`` (or download .zip directly from this repository)
3. Move AVMNIST data files to ``data/avmnist/``

### For Supervised benchmark (CentralNet)
- Run ``benchmarks.ipynb``

### For DINO models (unimodal, multimodal, mse, infonce and semi-supervised):
1. Adjust appropriate settings (folders locations, hyperparameters space, etc.) in ``config_dino.yaml`` or create own config file
2. Go to ``AVMNIST_Experiments/batch_files/`` and run ``submit_model.py``

    Example usage: ```python submit_models.py --models "multi_central" --config "config_multimodal_dino_old_augments.yaml" --metric "train_loss" --training_mode "mse"```

### For other SSL methods (multimodal/audio simclr, audio autoencoders, multimodal info_nce):
1. Go to ``AVMNIST_Experiments/other_ssl/``
2. Run the appropriate notebook



### TODO and limitations:
- No hyperparameter tuning for SSL models other than DINO (multimodal and unimodal) -> adjust objective functions
- No in-depth analysis on other SSL models


