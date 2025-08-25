import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime

def train_centralnet(model, args, train_loader, device, val_loader=None, log_file="training_log.csv", save_model="best_model.pth"):
    # Loss function and optimizer
    criterion = args.criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_acc = 0
    
    metadata = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs, # Can also be deduced from the amount of rows in the log file
        "optimizer": "Adam",
        "criterion": args.criterion.__class__.__name__,
        "model_name": "CentralNet",
        "channels" : args.channels,         # Base convolution channels
        "fusingmix" : args.fusingmix, # Fusion strategy
        "fusetype" : args.fusetype,     # Weighted sum fusion
        "num_outputs" : args.num_outputs,      # Number of classes (AVMNIST)
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

        for batch_idx, (image, audio, label) in enumerate(train_loader):
            # Move inputs to device
            image, audio, label = image.float().to(device), audio.float().to(device), label.to(device)

            # Forward pass
            audio_out, image_out, fusion_out = model(audio, image)

            # Compute the loss
            loss_audio = criterion(audio_out, label)
            loss_image = criterion(image_out, label)
            loss_fusion = criterion(fusion_out, label)

            # Total loss as per CentralNet's multimodal training objective
            loss = loss_audio + loss_image + loss_fusion

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
            val_loss, val_acc = validate_centralnet(model, val_loader, criterion, device)
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

def validate_centralnet(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for image, audio, label in val_loader:
            # Move inputs to device
            image, audio, label = image.float().to(device), audio.float().to(device), label.to(device)

            # Forward pass
            _, _, fusion_out = model(audio, image)

            # Compute the loss
            loss = criterion(fusion_out, label)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(fusion_out, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# Test function to evaluate final model performance
def test_centralnet(model, test_loader, criterion, device, test_log_file="test_results.csv"):
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
        for image, audio, label in test_loader:
            # Move inputs to device
            image, audio, label = image.float().to(device), audio.float().to(device), label.to(device)

            # Forward pass
            _, _, fusion_out = model(audio, image)

            # Compute the loss
            loss = criterion(fusion_out, label)
            total_loss += loss.item()

            # Compute accuracy
            probs = torch.softmax(fusion_out, dim=1)  # Needed for AUPRC
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