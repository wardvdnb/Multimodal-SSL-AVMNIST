"""Implements training pipeline for unimodal comparison."""
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import csv
import os

# Initialize the model, optimizer, loss, and other components
def train(model, args, train_loader, device, modalnum=0, val_loader=None, log_file="training_log.csv", save_model="model.pt"):
    """
    Trains the given model using the provided training data loader and arguments.
    Args:
        model (torch.nn.Module): The model to be trained.
        args (Namespace): A namespace containing training arguments such as criterion, learning_rate, and epochs.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        device (torch.device): The device to run the training on (e.g., 'cpu' or 'cuda').
        modalnum (int, optional): The index of the modality to use from the input data. Defaults to 0.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation data. Defaults to None.
        log_file (str, optional): Path to the CSV file for logging training and validation metrics. Defaults to "training_log.csv".
    Returns:
        None
    """

    # Loss function and optimizer
    criterion = args.criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_acc = 0
    metadata = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs, # Can also be deduced from the amount of rows in the log file
        "criterion": args.criterion.__class__.__name__,
        "optimizer": "Adam",
        "model_name": "Unimodal_Image" if modalnum == 0 else "Unimodal_Audio"
    }
    metadata_str = "# " + str(metadata) + "\n"

    # Generate timestamps
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H-%M-%S")

    log_file = log_file.replace(".csv", f"_{start_time_str}.csv")
    save_model = save_model.replace(".pt", f"_{start_time_str}.pt")

    # Ensure log file exists
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", metadata_str])

    # Training loop
    for epoch in (progress_bar := tqdm(range(args.epochs), desc="Training Epochs")):
        model.train()
        total_loss = 0

        for batch_idx, image_audio_label in enumerate(train_loader):
            # Move inputs to device
            modality, label = image_audio_label[modalnum].float().to(device), image_audio_label[-1].to(device)

            # Forward pass
            out = model(modality)

            # Compute the loss
            loss = criterion(out, label)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the loss
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        # Validation (if provided)
        val_loss, val_acc = None, None
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device, modalnum=modalnum)
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"Saving Best")
                torch.save(model, save_model)

        # Log metrics to CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss, val_loss, val_acc])

    print("Training Complete!")
    return save_model

# Validation loop
def validate(model, val_loader, criterion, device, modalnum=0):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for image_audio_label in tqdm(val_loader, desc="Validating"):
            # Move inputs to device
            modality, label = image_audio_label[modalnum].float().to(device), image_audio_label[-1].to(device)

            # Forward pass
            out = model(modality)

            # Compute the loss
            loss = criterion(out, label)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(out, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# Test function to evaluate final model performance
def test(model, test_loader, criterion, device, modalnum=0, test_log_file="test_results.csv"):
    """
    Evaluates the given model on the test dataset and logs the results.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to perform computations on (e.g., 'cpu' or 'cuda').
        modalnum (int, optional): Index of the modality to be used from the input data. Defaults to 0.
        test_log_file (str, optional): Path to the CSV file where test results will be logged. Defaults to "test_results.csv".
    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss over the test dataset.
            - accuracy (float): The accuracy of the model on the test dataset.
            - all_labels (list): List of true labels for the test dataset.
            - all_probs (list): List of predicted probabilities for the test dataset.
    """

    model.to(device)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H-%M-%S")
    
    test_log_file = test_log_file.replace(".csv", f"_{start_time_str}.csv")

    # Ensure test log file exists
    with open(test_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", "predicted_label", "probabilities"])

    with torch.no_grad():
        for image_audio_label in tqdm(test_loader, desc="Testing"):
            # Move inputs to device
            modality, label = image_audio_label[modalnum].float().to(device), image_audio_label[-1].to(device)

            # Forward pass
            out = model(modality)

            # Compute the loss
            loss = criterion(out, label)
            total_loss += loss.item()

            # Compute accuracy
            probs = torch.softmax(out, dim=1)  # Needed for AUPRC
            _, predicted = torch.max(probs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Log predictions to CSV
            with open(test_log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for i in range(len(label)):
                    writer.writerow([label[i].item(), predicted[i].item(), probs[i].tolist()])

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy, all_labels, all_probs