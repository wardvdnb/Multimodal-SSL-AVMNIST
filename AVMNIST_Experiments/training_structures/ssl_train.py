# Training methods for self-supervised learning on the AVMNIST dataset (other than DINO)

import sys
sys.path.append("../") # Add the parent directory to the path
from utils.reproducibility import set_seed
set_seed(1)
import torch
import torch.nn as nn
import os
import numpy as np
import lightning.pytorch as pl
from utils.plots_trials import load_all_versions, save_versions_to_csv, plot_loss
from utils.visualisations import pca_plot_dataloaders, pca_plot_multiclass, visualize_prediction_matrix, tsne_plot_multiclass
from training_structures.dino_train import train_downstream, train_knn_classifier, compute_gflops, compute_classification_metrics
from lightning.pytorch.loggers import CSVLogger
import copy
import time

def compute_train_results(pretrained_model, model_path, model_name, traindata, validdata, testdata):
    """
    Save Plot results (e.g. pca for frozen encoders) and 
    Accuracies (for downstream tasks (KNN and MLP on 10 epochs))
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device)

    knn_model, knn_accuracy = train_knn_classifier(pretrained_model, traindata, testdata, n_neighbors=5, is_dino_based=False)
    classifier = train_downstream(
        pretrained_model,
        traindata,
        validdata,
        testdata,
        num_epochs=10,
        device=device,
        save_path=f'{model_path}/downstream/{model_name}.pt',
        train_log_path=f'{model_path}/downstream/{model_name}_train_log.csv',
        test_log_path=f'{model_path}/downstream/{model_name}_test_log.csv',
        is_dino_based=False,
    )
    mlp_accuracy = compute_classification_metrics(classifier, testdata, device)['accuracy']

    return knn_accuracy, mlp_accuracy, classifier

def visualize_results(model_path, model_name, encoder, classifier, testdata):
    """
    Visualize results for the given model
    Args:
        model_path: Path to save the visualizations
        model_name: Name of the model
        encoder: The encoder model to visualize
        classifier: The classifier model to visualize
        testdata: The test data loader
    """

    pca_plot_path = os.path.join(model_path, f'{model_name}_pca_plots')
    confusion_matrix_path = os.path.join(model_path, f'{model_name}_confusion_matrix')
    log_path = os.path.join(model_path, f'logs')
    
    _ = pca_plot_dataloaders(encoder, testdata, selected_digits=[5, 8], 
                             dirpath=pca_plot_path, show_plots=False, is_dino_based=False)
    _ = pca_plot_multiclass(encoder, testdata, selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                            dirpath=pca_plot_path, show_plots=False, is_dino_based=False)
    _ = visualize_prediction_matrix(classifier, testdata, dirpath=confusion_matrix_path, show_plots=False)
    _ = tsne_plot_multiclass(encoder, testdata, selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                            dirpath=pca_plot_path, show_plots=False, random_seed=0, is_dino_based=False)
    
        # Logs (only if logs exist or get created during training)
    if os.path.exists(log_path):
        metrics_df = load_all_versions(log_path)
        save_versions_to_csv(metrics_df, model_path)
        plot_loss(metrics_df, plot_dir=model_path)
    else:
        print(f"[Warning] No logs found in {log_path}, skipping log-based visualizations.")

def train_and_evaluate_ssl(module, train_data_module, test_data_module,
                                        model_dir, model_name,
                                        input_data=[
                                            (torch.randn(1, 1, 28, 28),
                                            torch.randn(1, 1, 112, 112),  
                                            torch.tensor([1]))      
                                        ],
                                        modalities=["audio", "image"],
                                        seeds=[1, 2, 3], num_epochs=100):
    """
    Train and evaluate the model for multiple seeds
    Args:
        module: The PyTorch Lightning module to train
        train_data_module: The data module for training
        test_data_module: The data module for testing
        model_dir: Directory to save the model and logs
        model_name: Name of the model
        input_data: Sample input data for computing GFLOPs and parameters
        modalities: List of modalities to train on (audio, image or both)
        seeds: List of random seeds for reproducibility
        num_epochs: Number of epochs to train the model
    """

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)
    test_data_module.setup()  # Ensure the test data module is set up for evaluation

    initial_model_weights = copy.deepcopy(module.state_dict())

    gflops, params = compute_gflops(
        module.model,
        input_data,
    )

    test_accuracies_knn_image = []
    test_accuracies_mlp_image = []
    test_accuracies_knn_audio = []
    test_accuracies_mlp_audio = []
    training_times = []

    for seed in seeds:
        print(f"Training with seed {seed}...")
        set_seed(seed)
        module.load_state_dict(copy.deepcopy(initial_model_weights))

        logger = CSVLogger(f"{model_dir}", name=f"logs", version=f"version_seed{seed}")

        model_checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="train_loss",
            dirpath=f"{model_dir}/checkpoints/",
            filename=f"model_seed{seed}"+"-{epoch:02d}-{train_loss:.2f}",
            save_top_k=1,
            mode="min",
        )

        # early_stopping = pl.callbacks.EarlyStopping(
        #     monitor="train_loss",
        #     patience=5,
        #     verbose=True,
        #     mode="min",
        # )

        # Start training timer
        training_start = time.time()

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="gpu",
            devices=1,
            precision=16 if module.use_mixed_precision else 32,
            callbacks=[
                model_checkpoint,
                # early_stopping,
            ],
            logger=logger,
        )

        trainer.fit(module, train_data_module)  # Train the model

        # Calculate total training time
        training_time = time.time() - training_start

        trainer.save_checkpoint(f"{model_dir}/{model_name}_seed{seed}.ckpt")  # Save the model checkpoint
        
        pretrained_model = module.model  # Get the pretrained model

        if "audio" in modalities:
            knn_accuracy_audio, mlp_acc_audio, classifier_audio = compute_train_results(
                pretrained_model.audio_encoder,
                model_dir,
                f"{model_name}_audio",
                test_data_module.train_dataloader(),
                test_data_module.val_dataloader(),
                test_data_module.test_dataloader()
            )
            test_accuracies_knn_audio.append(knn_accuracy_audio)
            test_accuracies_mlp_audio.append(mlp_acc_audio)
            print(f"Audio KNN Accuracy: {knn_accuracy_audio}, MLP Accuracy: {mlp_acc_audio}")

        if "image" in modalities:
            knn_accuracy_image, mlp_acc_image, classifier_image = compute_train_results(
                pretrained_model.image_encoder,
                model_dir,
                f"{model_name}_image",
                test_data_module.train_dataloader(),
                test_data_module.val_dataloader(),
                test_data_module.test_dataloader()
            )
            test_accuracies_knn_image.append(knn_accuracy_image)
            test_accuracies_mlp_image.append(mlp_acc_image)
            print(f"Image KNN Accuracy: {knn_accuracy_image}, MLP Accuracy: {mlp_acc_image}")

        training_times.append(training_time)

    if "image" in modalities:
        knn_accuracy_image = np.mean(test_accuracies_knn_image)
        knn_acc_std_image = np.std(test_accuracies_knn_image)
        mlp_acc_image = np.mean(test_accuracies_mlp_image)
        mlp_acc_std_image = np.std(test_accuracies_mlp_image)

    if "audio" in modalities:
        knn_accuracy_audio = np.mean(test_accuracies_knn_audio)
        knn_acc_std_audio = np.std(test_accuracies_knn_audio)
        mlp_acc_audio = np.mean(test_accuracies_mlp_audio)
        mlp_acc_std_audio = np.std(test_accuracies_mlp_audio)

    training_time = np.mean(training_times)

    # Also save a performance summary file
    # Also save a performance summary file
    perf_summary = {
        "model_name": model_name,
        "parameters": f"{params/1e6:.2f}M",
        "gflops": f"{gflops:.2f}",
        "seeds": seeds,
        "training_time_hours": f"{training_time/3600:.2f}",
    }

    if "image" in modalities:
        perf_summary["downstream_mlp_acc_image"] = f"{mlp_acc_image:.4f} ± {mlp_acc_std_image:.4f}"
        perf_summary["downstream_knn_accuracy_image"] = f"{knn_accuracy_image:.4f} ± {knn_acc_std_image:.4f}"

    if "audio" in modalities:
        perf_summary["downstream_mlp_acc_audio"] = f"{mlp_acc_audio:.4f} ± {mlp_acc_std_audio:.4f}"
        perf_summary["downstream_knn_accuracy_audio"] = f"{knn_accuracy_audio:.4f} ± {knn_acc_std_audio:.4f}"

    with open(f"{model_name}_performance_summary.txt", "w") as f:
        for key, value in perf_summary.items():
            f.write(f"{key}: {value}\n")

    # Visualize for last seed's encoders (both audio and image)

    if "image" in modalities:
        visualize_results(
            model_path=model_dir,
            model_name=f"{model_name}_image",
            encoder=pretrained_model.image_encoder,
            classifier=classifier_image,
            testdata=test_data_module.test_dataloader()
        )

    if "audio" in modalities:
        visualize_results(
            model_path=model_dir,
            model_name=f"{model_name}_audio",
            encoder=pretrained_model.audio_encoder,
            classifier=classifier_audio,
            testdata=test_data_module.test_dataloader()
        )

class LateFusionEncoder(nn.Module):
    def __init__(self, audio_encoder, image_encoder, fusion="concat", proj_dim=None):
        """
        Args:
            audio_encoder: pretrained audio encoder
            image_encoder: pretrained image encoder
            fusion: "concat", "sum", or "mean"
            proj_dim: if not None, projects fused features to this dimension
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
        self.fusion = fusion

        # Freeze encoders for evaluation
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        if fusion == "concat":
            in_dim = self.audio_encoder.output_dim + self.image_encoder.output_dim
        else:
            in_dim = self.audio_encoder.output_dim  # assumes same size

        self.proj = None
        if proj_dim is not None:
            self.proj = nn.Linear(in_dim, proj_dim)
            self.output_dim = proj_dim
        else:
            self.output_dim = in_dim

    def forward(self, image, audio):
        z_i = self.image_encoder(images=image, spectrograms=None)
        z_a = self.audio_encoder(images=None, spectrograms=audio)

        if self.fusion == "concat":
            z = torch.cat([z_i, z_a], dim=1)
        elif self.fusion == "sum":
            z = z_i + z_a
        elif self.fusion == "mean":
            z = (z_i + z_a) / 2
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion}")

        if self.proj is not None:
            z = self.proj(z)
        return z

def evaluate_multimodal_ssl(
        lightning_cls,
        ckpt_paths,  # list of ckpt paths (../multimodal_simclr_model_seed1.ckpt, etc.)
        test_data_module,
        model_name,
        fusion="concat",
        proj_dim=None,
    ):
    """
    Evaluate late fusion of pretrained encoders from multiple checkpoints.
    """
    test_data_module.setup()
    test_accuracies_knn = []
    test_accuracies_mlp = []

    classifiers = []

    for ckpt_path in ckpt_paths:
        print(f"Loading checkpoint {ckpt_path}...")
        module = lightning_cls.load_from_checkpoint(ckpt_path)
        audio_enc = module.model.audio_encoder
        image_enc = module.model.image_encoder

        fusion_encoder = LateFusionEncoder(audio_enc, image_enc, fusion=fusion, proj_dim=proj_dim)

        knn_acc, mlp_acc, classifier = compute_train_results(
            pretrained_model=fusion_encoder,
            model_path=".",
            model_name=f"{model_name}_fusion",
            traindata=test_data_module.train_dataloader(),
            validdata=test_data_module.val_dataloader(),
            testdata=test_data_module.test_dataloader(),
        )

        test_accuracies_knn.append(knn_acc)
        test_accuracies_mlp.append(mlp_acc)
        classifiers.append(classifier)

        print(f"Fusion KNN Accuracy: {knn_acc}, MLP Accuracy: {mlp_acc}")

    # Aggregate results
    knn_mean = np.mean(test_accuracies_knn)
    knn_std = np.std(test_accuracies_knn)
    mlp_mean = np.mean(test_accuracies_mlp)
    mlp_std = np.std(test_accuracies_mlp)

    summary = {
        "fusion": fusion,
        "downstream_knn_accuracy_fusion": f"{knn_mean:.4f} ± {knn_std:.4f}",
        "downstream_mlp_acc_fusion": f"{mlp_mean:.4f} ± {mlp_std:.4f}",
    }

    with open(f"{model_name}_fusion_performance_summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    # Visualize for last seed
    visualize_results(
        model_path=".",
        model_name=f"{model_name}_fusion",
        encoder=fusion_encoder,
        classifier=classifiers[-1],
        testdata=test_data_module.test_dataloader(),
    )

    return summary
