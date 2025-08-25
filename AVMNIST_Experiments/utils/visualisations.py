from datetime import datetime
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import pandas as pd
from training_structures.dino_train import train_downstream, train_knn_classifier, compute_classification_metrics
import torch
from sklearn.decomposition import PCA
import random
from tqdm import tqdm
import torch.nn.functional as F
from models.dino import CrossAttentionMultiModalEncoder, SimpleMultiModalEncoder, \
ViTMultiModalEncoder, LSTMMultiModalEncoder, MultiModalDINOLightning
from sklearn.manifold import TSNE

def show_images_augmentations(image, spectrograms, spectrogram_titles, label):
    """
    Display image and multiple spectrograms in rows of maximum 5 items with enhanced axes information.
    
    Args:
        image (torch.Tensor): The image tensor.
        spectrograms (list of torch.Tensor): List of spectrogram tensors.
        spectrogram_titles (list of str): List of titles for each spectrogram.
        label (int): The label of the image and spectrograms.
    """
    def process_image(image):
        image = image.numpy().transpose((1, 2, 0))  # Convert (C, H, W) to (H, W, C)
        return image  # No longer multiplying by 255 since we want to keep original values

    # Calculate layout
    n_spectrograms = len(spectrograms)
    items_per_row = 5
    n_rows = (n_spectrograms + 1 + items_per_row - 1) // items_per_row  # +1 for the original image
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(4*items_per_row, 4*n_rows))
    
    # Process the image
    image_processed = process_image(image)
    
    # Create grid spec to handle layout
    gs = plt.GridSpec(n_rows, items_per_row, figure=fig)
    
    # Plot the image in the first position
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image_processed*255, cmap='gray', vmin=0, vmax=255) # unnormalized image values
    ax_img.set_title(f"Image\nLabel: {label}")
    
    # Plot each spectrogram TODO: get this checked because they're based on 0-255 pixel values so the conversions might be off
    for i, (spectrogram, title) in enumerate(zip(spectrograms, spectrogram_titles)):
        row = (i + 1) // items_per_row  # +1 because we started with the image
        col = (i + 1) % items_per_row
        
        ax = fig.add_subplot(gs[row, col])
        
        # Process and plot spectrogram
        spectrogram_processed = process_image(spectrogram)
        
        # Plot spectrogram with proper axes
        im = ax.imshow(
            spectrogram_processed,
            cmap='viridis',
            aspect='auto',
            origin='lower',
        )
        
        ax.set_title(f"{title}\nLabel: {label}")

        # Since this is a 112x112 image, use pixel indices for axes
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Frequency Index")
        
        # Since this is a 112x112 image, use pixel indices for axes
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Frequency Index")
        
        # Only show a few ticks to avoid crowding
        ax.set_xticks([0, 55, 111])
        ax.set_yticks([0, 55, 111])

    plt.tight_layout()
    plt.show()

def show_images(image, audio, label, save_path=None):
    """
    Display image and audio side by side with a shared title label.
    Optionally save the figure to an SVG file if save_path is provided.
    """

    def process_image(img):
        img = img.clone().numpy().transpose((1, 2, 0))  # (C, H, W) → (H, W, C)
        img *= 255  # Unnormalize
        return img

    image1 = process_image(image)
    image2 = process_image(audio)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Label: {label}", fontsize=16, weight='bold')

    axes[0].imshow(image1.squeeze(), cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Visual Modality", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(image2.squeeze(), cmap='viridis', vmin=0, vmax=255)
    axes[1].set_title("Audio Modality", fontsize=12)
    axes[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space for suptitle

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')

    plt.show()

# This plot() code comes from: https://pytorch.org/vision/0.10/auto_examples/plot_transforms.html
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    plt.rcParams["savefig.bbox"] = 'tight'
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

# Function to compute and visualize key metrics
def evaluate_results(true_labels, predicted_probs, num_classes=10):
    predicted_labels = np.argmax(predicted_probs, axis=1)
    predicted_probs = np.array(predicted_probs)
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels))

    # Compute and plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Binarize the labels for multiclass AUPRC
    true_labels_bin = np.array(label_binarize(true_labels, classes=np.arange(num_classes)))
    
    # Compute precision-recall curve and AUPRC for each class
    auprc_values = []
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], predicted_probs[:, i])
        auprc = auc(recall, precision)
        auprc_values.append(auprc)
        plt.plot(recall, precision, marker='.', label=f'Class {i} (AUPRC = {auprc:.4f})')

    mean_auprc = np.mean(auprc_values)
    print(f"Mean AUPRC: {mean_auprc:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Per Class)")
    plt.legend()
    plt.show()

    return mean_auprc

def plot_training_results_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot training loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='x')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    # Plot training accuracy
    axes[1].plot(df["epoch"], df["val_accuracy"], label="Val Acc", color='g', marker='o')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_training_results_from_csvs(csv_files, model_names):
    """
    Plots training loss and validation accuracy for multiple models.
    
    Args:
        csv_files (list of str): List of CSV file paths, one per model.
        model_names (list of str): List of model names for labeling plots.
    """
    num_models = len(csv_files)
    rows_loss = (num_models + 1) // 2  # Ensure rows of 2 plots per row for loss
    rows_acc = (num_models + 1) // 2  # Separate rows for accuracy
    
    fig_loss, axes_loss = plt.subplots(rows_loss, 2, figsize=(12, 6 * rows_loss))
    fig_acc, axes_acc = plt.subplots(rows_acc, 2, figsize=(12, 6 * rows_acc))
    
    if num_models == 1:
        axes_loss = [[axes_loss]]  # Ensure axes is always a 2D array
        axes_acc = [[axes_acc]]
    elif rows_loss == 1:
        axes_loss = [axes_loss]  # Make it a list of lists for consistency
        axes_acc = [axes_acc]
    
    for idx, (csv_file, model_name) in enumerate(zip(csv_files, model_names)):
        df = pd.read_csv(csv_file)
        row, col = divmod(idx, 2)
        
        # Plot training loss
        axes_loss[row][col].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
        axes_loss[row][col].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='x')
        axes_loss[row][col].set_xlabel("Epoch")
        axes_loss[row][col].set_ylabel("Loss")
        axes_loss[row][col].set_title(f"{model_name} - Training Loss")
        axes_loss[row][col].legend()
        
        # Plot validation accuracy separately with fixed scale 0-100%
        axes_acc[row][col].plot(df["epoch"], df["val_accuracy"], label="Val Acc", color='g', marker='o')
        axes_acc[row][col].set_xlabel("Epoch")
        axes_acc[row][col].set_ylabel('Accuracy (%)')
        axes_acc[row][col].set_title(f"{model_name} - Validation Accuracy")
        axes_acc[row][col].set_ylim(0, 100)  # Set fixed scale from 0 to 100%
        axes_acc[row][col].legend()
    
    plt.tight_layout()
    # fig_loss.suptitle("Training and Validation Loss")
    # fig_acc.suptitle("Validation Accuracy")
    plt.show()

def pca_plot_dataloaders(model, dataloader, selected_digits=None, device='cuda', max_samples_per_digit=100, dirpath=None, show_plots=True, is_dino_based=True):
    """
    Create a PCA visualization for embeddings from a multimodal DINO model using a dataloader.
    
    Args:
        model: The pretrained DINO model
        dataloader: DataLoader containing (image, audio, label) triplets
        selected_digits: List of 2 digit classes to visualize (if None, randomly selects 2)
        device: Device to run the model on
        max_samples_per_digit: Maximum number of samples to collect per digit
    """
    os.makedirs(dirpath, exist_ok=True)

    model.eval()
    
    # Dictionary to store samples by digit class
    digit_samples = {}
    
    # Collect samples per digit
    print("Collecting samples from dataloader...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, audios, labels = batch[0].to(device), batch[1].to(device), batch[2].numpy()
            
            # Process each sample in the batch
            for i, label in enumerate(labels):
                label = int(label)  # Ensure it's an integer
                
                if label not in digit_samples:
                    digit_samples[label] = {"images": [], "audios": []}
                
                # Only collect up to max_samples_per_digit for each digit
                if len(digit_samples[label]["images"]) < max_samples_per_digit:
                    digit_samples[label]["images"].append(images[i].unsqueeze(0))
                    digit_samples[label]["audios"].append(audios[i].unsqueeze(0))
    
    # If no specific digits selected, randomly select 2 digits that have samples
    available_digits = list(digit_samples.keys())
    if selected_digits is None:
        if len(available_digits) < 2:
            raise ValueError("Not enough different digits found in dataloader")
        selected_digits = random.sample(available_digits, 2)
    else:
        # Ensure the selected digits have samples
        for digit in selected_digits:
            if digit not in digit_samples:
                raise ValueError(f"Selected digit {digit} not found in dataloader")
    
    print(f"Selected digits for visualization: {selected_digits}")
    
    # Extract the embeddings for the selected digits
    all_embeddings = []
    all_labels = []
    
    for digit in selected_digits:
        # Stack all images and audios for this digit
        digit_images = torch.cat(digit_samples[digit]["images"], dim=0)
        digit_audios = torch.cat(digit_samples[digit]["audios"], dim=0)
        
        # Get embeddings from the model (assuming student encoder gives embeddings)
        with torch.no_grad():
            # Adjust this line based on your model's architecture
            # If using the full DINO model, access the student encoder directly
            # Convert input tensors to float32
            digit_images = digit_images.float()  # Convert from double to float
            digit_audios = digit_audios.float()  # Convert from double to float

            # Then get embeddings from the model
            embeddings = model.student(digit_images, digit_audios) if is_dino_based else model(digit_images, digit_audios)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(np.full(len(digit_images), digit))
    
    # Combine all embeddings and labels
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    # Apply PCA to reduce dimensions to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Visualize the embeddings
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot for each digit
    colors = ["blue", "orange"]
    for i, digit in enumerate(selected_digits):
        idx = labels == digit
        plt.scatter(
            reduced_embeddings[idx, 0], 
            reduced_embeddings[idx, 1], 
            color=colors[i], 
            label=f"Digit {digit}", 
            alpha=0.7
        )
    
    # Add title and labels
    plt.title("PCA Visualization of Embeddings")
    plt.xlabel(f"PC 1 (Explained variance: {pca.explained_variance_ratio_[0]:.3f})")
    plt.ylabel(f"PC 2 (Explained variance: {pca.explained_variance_ratio_[1]:.3f})")
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save and display the plot
    plt.savefig(os.path.join(dirpath, f"pca_digits_{selected_digits[0]}_{selected_digits[1]}.png"), dpi=300, bbox_inches="tight")
    plt.show() if show_plots else plt.close()
    
    return reduced_embeddings, labels, pca

# For visualizing more than 2 digits, you can use this modified function:
def pca_plot_multiclass(model, dataloader, selected_digits=None, device='cuda', 
                        max_samples_per_digit=100, dirpath=None, show_plots=True, is_dino_based=True):
    """
    Create a PCA visualization for embeddings from multiple digits using a dataloader.
    
    Args:
        model: The pretrained DINO model
        dataloader: DataLoader containing (image, audio, label) triplets
        selected_digits: List of digit classes to visualize (if None, uses all available)
        device: Device to run the model on
        max_samples_per_digit: Maximum number of samples to collect per digit
    """
    os.makedirs(dirpath, exist_ok=True)

    model.eval()
    
    # Dictionary to store samples by digit class
    digit_samples = {}
    
    # Collect samples per digit
    print("Collecting samples from dataloader...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, audios, labels = batch[0].to(device), batch[1].to(device), batch[2].numpy()
            
            # Process each sample in the batch
            for i, label in enumerate(labels):
                label = int(label)  # Ensure it's an integer
                
                if label not in digit_samples:
                    digit_samples[label] = {"images": [], "audios": []}
                
                # Only collect up to max_samples_per_digit for each digit
                if len(digit_samples[label]["images"]) < max_samples_per_digit:
                    digit_samples[label]["images"].append(images[i].unsqueeze(0))
                    digit_samples[label]["audios"].append(audios[i].unsqueeze(0))
    
    # If no specific digits selected, use all available
    available_digits = list(digit_samples.keys())
    if selected_digits is None:
        selected_digits = available_digits
    else:
        # Ensure the selected digits have samples
        for digit in selected_digits:
            if digit not in digit_samples:
                raise ValueError(f"Selected digit {digit} not found in dataloader")
    
    print(f"Visualizing digits: {selected_digits}")
    
    # Extract the embeddings for the selected digits
    all_embeddings = []
    all_labels = []
    
    for digit in selected_digits:
        if len(digit_samples[digit]["images"]) > 0:
            # Stack all images and audios for this digit
            digit_images = torch.cat(digit_samples[digit]["images"], dim=0)
            digit_audios = torch.cat(digit_samples[digit]["audios"], dim=0)
            
            # Get embeddings from the model
            with torch.no_grad():
                # Convert input tensors to float32
                digit_images = digit_images.float()  # Convert from double to float
                digit_audios = digit_audios.float()  # Convert from double to float

                # Then get embeddings from the model
                embeddings = model.student(digit_images, digit_audios) if is_dino_based else model(digit_images, digit_audios)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(np.full(len(digit_images), digit))
    
    # Combine all embeddings and labels
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    # Apply PCA to reduce dimensions to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Visualize the embeddings
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot for each digit with a different color
    for digit in selected_digits:
        idx = labels == digit
        plt.scatter(
            reduced_embeddings[idx, 0], 
            reduced_embeddings[idx, 1], 
            label=f"Digit {digit}", 
            alpha=0.7
        )
    
    # Add title and labels
    plt.title("PCA Visualization of Embeddings for Multiple Digits")
    plt.xlabel(f"PC 1 (Explained variance: {pca.explained_variance_ratio_[0]:.3f})")
    plt.ylabel(f"PC 2 (Explained variance: {pca.explained_variance_ratio_[1]:.3f})")
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save and display the plot
    plt.savefig(os.path.join(dirpath, "pca_multiclass.png"), dpi=300, bbox_inches="tight")
    plt.show() if show_plots else plt.close()
    
    return reduced_embeddings, labels, pca

def tsne_plot_multiclass(
    model, 
    dataloader, 
    selected_digits=None, 
    device='cuda', 
    max_samples_per_digit=100, 
    dirpath=None, 
    show_plots=True, 
    is_dino_based=True,
    tsne_perplexity=30,
    tsne_n_iter=1000,
    random_seed=0
):
    """
    Create a t-SNE visualization for embeddings from multiple digits using a dataloader.
    
    Args:
        model: The pretrained DINO model
        dataloader: DataLoader containing (image, audio, label) triplets
        selected_digits: List of digit classes to visualize (if None, uses all available)
        device: Device to run the model on
        max_samples_per_digit: Maximum number of samples to collect per digit
        dirpath: Path to save the output plot
        show_plots: Whether to display the plot after saving
        is_dino_based: Whether the model is a DINO model (uses .student for embeddings)
        tsne_perplexity: Perplexity parameter for t-SNE
        tsne_n_iter: Number of iterations for t-SNE optimization
        random_seed: Seed for reproducibility
    """
    os.makedirs(dirpath, exist_ok=True)

    model.eval()
    digit_samples = {}

    print("Collecting samples from dataloader...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, audios, labels = batch[0].to(device), batch[1].to(device), batch[2].numpy()
            
            for i, label in enumerate(labels):
                label = int(label)
                if label not in digit_samples:
                    digit_samples[label] = {"images": [], "audios": []}
                
                if len(digit_samples[label]["images"]) < max_samples_per_digit:
                    digit_samples[label]["images"].append(images[i].unsqueeze(0))
                    digit_samples[label]["audios"].append(audios[i].unsqueeze(0))

    available_digits = list(digit_samples.keys())
    if selected_digits is None:
        selected_digits = available_digits
    else:
        for digit in selected_digits:
            if digit not in digit_samples:
                raise ValueError(f"Selected digit {digit} not found in dataloader")

    print(f"Visualizing digits: {selected_digits}")

    all_embeddings = []
    all_labels = []

    for digit in selected_digits:
        if len(digit_samples[digit]["images"]) > 0:
            digit_images = torch.cat(digit_samples[digit]["images"], dim=0).float()
            digit_audios = torch.cat(digit_samples[digit]["audios"], dim=0).float()

            with torch.no_grad():
                embeddings = model.student(digit_images, digit_audios) if is_dino_based else model(digit_images, digit_audios)
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(np.full(len(digit_images), digit))

    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)

    # Apply t-SNE to reduce dimensions to 2D
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, n_iter=tsne_n_iter, random_state=random_seed)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Visualize the embeddings
    plt.figure(figsize=(12, 10))
    for digit in selected_digits:
        idx = labels == digit
        plt.scatter(
            reduced_embeddings[idx, 0],
            reduced_embeddings[idx, 1],
            label=f"Digit {digit}",
            alpha=0.7
        )

    plt.title("t-SNE Visualization of Embeddings for Multiple Digits")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(dirpath, "tsne_multiclass.png"), dpi=300, bbox_inches="tight")
    plt.show() if show_plots else plt.close()

    return reduced_embeddings, labels, tsne

def visualize_prediction_matrix(model, testloader, device='cuda', dirpath=None, show_plots=True):
    """
    Create and save a confusion matrix visualization for MNIST digit predictions.
    
    Args:
        model: The downstream classifier model
        testloader: DataLoader containing test data (image, audio, label)
        device: Device to run the model on ('cuda' or 'cpu')
        dirpath: Directory path to save the visualization
    """

    metrics = compute_classification_metrics(model, testloader, device=device)
    cm = metrics['confusion_matrix']
    cm_normalized = metrics['normalized_confusion_matrix']
    accuracy = metrics['accuracy']
    per_class_acc = metrics['per_class_accuracy']
    report = metrics['classification_report']
    all_preds = metrics['predictions']
    all_labels = metrics['true_labels']
    all_probs = metrics['probabilities']

    # Create directory if it doesn't exist

    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f'DINO MNIST Classifier Evaluation\nAccuracy: {accuracy:.2f}%', fontsize=16)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix (counts)')
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix (normalized)')
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].set_ylabel('True Label')

    axes[1, 0].bar(range(10), per_class_acc * 100)
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_xlabel('Digit')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Per-Class Accuracy')
    axes[1, 0].set_ylim([0, 100])
    for i, acc in enumerate(per_class_acc):
        axes[1, 0].text(i, acc * 100 + 2, f'{acc * 100:.1f}%', ha='center')

    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.1, f"Classification Report:\n\n{report}", fontfamily='monospace', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    if dirpath:
        plt.savefig(os.path.join(dirpath, "dino_mnist_classification_matrix.svg"), format='svg', bbox_inches="tight")
    plt.show() if show_plots else plt.close()
    
    #----------- CONFIDENCE ANALYSIS -----------#

    # Confidence analysis:
    # - Check the top-5 most confident predictions for each digit
    # - If the top-5 confident predictions are all green (correct):
    # The model strongly recognizes this digit and is not making confident errors.
    # - If some of the bars are red (incorrect):
    # The model is overconfident in its mistakes, which may indicate a systemic error.
    # - If confidence values are close to 1.0:
    # The model is highly certain about these predictions, which could be good or bad.
    # - If a class has lower confidence than others:
    # The model struggles with that class, even when it gets them right.

    # Additional visualization: Most confident examples
    plt.figure(figsize=(15, 15))
    
    # Find the minimum confidence level across all correct predictions
    # This will help us set a better y-axis scale
    min_confidence = 0.7  # Default minimum
    
    # Dictionary to store confidence values by digit
    digit_confidences = {}
    
    for digit in range(10):
        # Find where true label matches this digit
        digit_indices = np.where(all_labels == digit)[0]
        
        if len(digit_indices) > 0:
            # Get probability for the correct digit class
            correct_probs = all_probs[digit_indices, digit]
            
            # Store confidence values
            digit_confidences[digit] = correct_probs
            
            # Update minimum confidence if needed
            if len(correct_probs) > 0:
                digit_min = np.min(correct_probs)
                if digit_min < min_confidence:
                    min_confidence = max(0.5, digit_min - 0.1)  # Set a reasonable floor
    
    # Now create the plots with a better y-scale
    for digit in range(10):
        plt.subplot(5, 2, digit+1)
        
        # Find where true label matches this digit
        digit_indices = np.where(all_labels == digit)[0]
        
        if len(digit_indices) > 0:
            # Get probability for the correct digit class
            correct_probs = all_probs[digit_indices, digit]
            
            # Find most confident examples (highest probability for correct class)
            most_conf_idx = digit_indices[np.argsort(correct_probs)[-5:]]
            conf_values = correct_probs[np.argsort(correct_probs)[-5:]]
            
            # Get prediction for these examples
            conf_preds = all_preds[most_conf_idx]
            
            # Plot confidence values
            plt.ylim([min_confidence, 1.02])  # Add a small margin at the top
            plt.title(f'Digit {digit}: Most Confident Predictions')
            plt.xlabel('Example')
            plt.ylabel('Confidence')
            
            # Add grid lines for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Calculate confidence range for this digit to mention in the title
            if digit in digit_confidences and len(digit_confidences[digit]) > 0:
                digit_conf_min = np.min(digit_confidences[digit])
                digit_conf_max = np.max(digit_confidences[digit])
                plt.title(f'Digit {digit}: Confidence Range [{digit_conf_min:.3f} - {digit_conf_max:.3f}]')
            
            # Annotate with prediction
            for i, (pred, conf) in enumerate(zip(conf_preds, conf_values)):
                correct = pred == digit
                color = 'green' if correct else 'red'
                plt.text(i, conf + 0.01, f'{pred} ({conf:.3f})', ha='center', color=color, fontweight='bold')
    
    plt.tight_layout()
    
    # Save and display the plot
    if dirpath:
        plt.savefig(os.path.join(dirpath, "dino_mnist_confidence_analysis.png"), dpi=300, bbox_inches="tight")
    plt.show() if show_plots else plt.close()
    
    #----------- CONFIDENCE DISTRIBUTION -----------#

    # - A right-skewed distribution (high confidence)
    # The model is confident in its predictions for this digit.
    # Good if accuracy is also high but could indicate overconfidence if misclassification rates are high.

    # - A more uniform distribution (spread confidence)
    # The model is uncertain about this digit.
    # Could mean the digit is harder to classify (e.g., digits 3, 5, 8 often look similar).

    # - If mean and median are close
    # The model consistently assigns similar confidence scores.

    # - If median is much lower than the mean
    # There are some very confident predictions, but many uncertain ones.
    # The model may be making confident mistakes.

    plt.figure(figsize=(15, 10))
    
    # Plot histogram of confidence values for each digit
    for digit in range(10):
        plt.subplot(5, 2, digit+1)
        
        digit_indices = np.where(all_labels == digit)[0]
        if len(digit_indices) > 0:
            # Get probabilities for the correct class
            correct_probs = all_probs[digit_indices, digit]
            
            # Plot histogram
            plt.hist(correct_probs, bins=20, alpha=0.7)
            plt.title(f'Digit {digit}: Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.xlim([0, 1.05])
            
            # Add mean and median lines
            mean_conf = np.mean(correct_probs)
            median_conf = np.median(correct_probs)
            plt.axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.3f}')
            plt.axvline(median_conf, color='green', linestyle='-.', label=f'Median: {median_conf:.3f}')
            plt.legend()
    
    plt.tight_layout()
    
    # Save and display the plot
    if dirpath:
        plt.savefig(os.path.join(dirpath, "dino_mnist_confidence_distribution.png"), dpi=300, bbox_inches="tight")
    plt.show() if show_plots else plt.close()
    
    # Return metrics for further analysis if needed
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }


### BROKEN CODE BELOW ### ----------------------------------------------------------------

def find_best_checkpoint(model_dir):
    """Find the best checkpoint file in the given model directory."""
    # Look for checkpoint files
    checkpoint_files = glob.glob(os.path.join(model_dir, "*.ckpt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
    
    # If there's only one, return it
    if len(checkpoint_files) == 1:
        return checkpoint_files[0]
    
    # If multiple, try to find the best one (usually has the highest epoch or step)
    # We'll sort by the file modification time as a fallback
    return sorted(checkpoint_files, key=os.path.getmtime)[-1]

def load_model_from_checkpoint(checkpoint_path, encoder_class, device='cuda'):
    """Load a model from checkpoint without knowing its specific architecture."""
    try:
        print(f"Attempting to load checkpoint with Lightning: {checkpoint_path}")
        # module = MultiModalDINOLightning(encoder_class=encoder_class)
        module = MultiModalDINOLightning.load_from_checkpoint(checkpoint_path, map_location=device, encoder_class=encoder_class)
        
        # Try to access the model attribute if it exists
        if hasattr(module, 'model'):
            print("Found model attribute in Lightning module")
            return module.model
        else:
            print("No model found in Lightning module")
            return
            
    except Exception as e:
        print(f"Lightning loading failed with error: {e}")

def evaluate_models(model_dirs, dataloaders, output_base_dir="model_saves", device="cuda", n_neighbors=5, num_epochs=10, batch_size=128):
    """
    Evaluate multiple models from a list of model directories.
    
    Args:
        model_dirs: List of model directory names
        data_path: Path to the data (e.g., 'data/avmnist')
        output_base_dir: Base directory for saving results
        device: Device to run on ('cuda' or 'cpu')
        n_neighbors: Number of neighbors for KNN classifier
        num_epochs: Number of epochs for training the downstream classifier
        batch_size: Batch size for dataloaders
    """
    # Prepare output directories
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Prepare summary results dataframe
    summary_results = []
    
    # Process each model directory
    for model_dirname, encoder_class in model_dirs.items():
        print(f"\n{'='*80}\nProcessing model: {model_dirname}\n{'='*80}")
        
        model_dir = os.path.join("model_saves", model_dirname)
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory {model_dir} does not exist. Skipping...")
            continue
        
        # Set up paths for this model
        model_results_dir = os.path.join(output_base_dir, model_dirname)
        pca_plot_path = os.path.join(model_results_dir, "pca_plots")
        confusion_matrix_path = os.path.join(model_results_dir, "confusion_matrix")
        downstream_save_dir = os.path.join("model_saves", model_dirname, "downstream")
        downstream_train_log_dir = os.path.join("training_logs", model_dirname, "downstream")
        downstream_test_log_dir = os.path.join("test_logs", model_dirname, "downstream")
        
        # Create necessary directories
        os.makedirs(model_results_dir, exist_ok=True)
        os.makedirs(pca_plot_path, exist_ok=True)
        os.makedirs(confusion_matrix_path, exist_ok=True)
        os.makedirs(downstream_save_dir, exist_ok=True)
        os.makedirs(downstream_train_log_dir, exist_ok=True)
        os.makedirs(downstream_test_log_dir, exist_ok=True)
        
        # try:
        # Find and load the best checkpoint
        best_model_path = find_best_checkpoint(model_dir)
        print(f"Found best checkpoint: {best_model_path}")
        
        # Load the model
        model = load_model_from_checkpoint(best_model_path, encoder_class, device)
        
        # Display model gating information if available
        if hasattr(model, 'student'):
            if hasattr(model.student, 'gate_audio'):
                print("Audio gate:", model.student.gate_audio)
            if hasattr(model.student, 'gate_image'):
                print("Image gate:", model.student.gate_image)
        
        # Get data loaders
        traindata, validdata, testdata = dataloaders
        
        # Train KNN classifier
        print("Training KNN classifier...")
        knn_model, knn_accuracy = train_knn_classifier(model, traindata, testdata, n_neighbors=n_neighbors)
        print(f"KNN classifier accuracy: {knn_accuracy:.2f}%")
        
        # Train downstream classifier
        print("Training downstream classifier...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        classifier = train_downstream(
            model,
            traindata,
            validdata,
            testdata,
            num_epochs=num_epochs,
            device=device,
            save_path=os.path.join(downstream_save_dir, f'downstream_model_{timestamp}.pt'),
            train_log_path=os.path.join(downstream_train_log_dir, f'downstream_train_log_{timestamp}.csv'),
            test_log_path=os.path.join(downstream_test_log_dir, f'downstream_test_log_{timestamp}.csv')
        )
        
        # Generate PCA plots
        print("Generating PCA plots...")
        pca_plot_dataloaders(model, testdata, selected_digits=[9, 8], dirpath=pca_plot_path)
        pca_plot_multiclass(model, testdata, selected_digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dirpath=pca_plot_path)
        
        # Generate confusion matrix visualization
        print("Generating confusion matrix visualization...")
        metrics = visualize_prediction_matrix(classifier, testdata, device=device, dirpath=confusion_matrix_path)
        
        # Add to summary results
        summary_results.append({
            'model_name': model_dirname,
            'knn_accuracy': knn_accuracy*100,
            'mlp_accuracy': metrics['accuracy'],
            'checkpoint_path': best_model_path,
            'per_class_accuracies': metrics['per_class_accuracy']
        })
        
        print(f"Finished processing model: {model_dirname}")
            
        # except Exception as e:
        #     print(f"Error processing model {model_dirname}: {e}")
        #     import traceback
        #     traceback.print_exc()
    
    # Create a summary report
    if summary_results:
        print("\nCreating summary report...")
        
        # Basic summary table
        summary_df = pd.DataFrame([
            {
                'model_name': r['model_name'],
                'knn_accuracy': r['knn_accuracy'],
                'mlp_accuracy': r['mlp_accuracy']
            } for r in summary_results
        ])
        
        # Save summary table
        summary_df.to_csv(os.path.join(output_base_dir, 'model_summary.csv'), index=False)
        
        # Create comparative visualizations
        plt.figure(figsize=(12, 8))
        
        # Sort by MLP accuracy
        summary_df = summary_df.sort_values('mlp_accuracy', ascending=False)
        
        # Plot comparative performance
        x = np.arange(len(summary_df))
        width = 0.35
        
        plt.bar(x - width/2, summary_df['knn_accuracy'], width, label='KNN Accuracy')
        plt.bar(x + width/2, summary_df['mlp_accuracy'], width, label='MLP Accuracy')
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.title('Comparative Model Performance')
        plt.xticks(x, summary_df['model_name'], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_base_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Summary saved to {os.path.join(output_base_dir, 'model_summary.csv')}")
    
    return summary_results