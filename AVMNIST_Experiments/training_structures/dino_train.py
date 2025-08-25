import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import csv
from datetime import datetime
import json
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from models.dino import DownstreamClassifier, FeatureExtractor
from torchinfo import summary

def compute_gflops(model, input_data):
    """
    Compute GFLOPs for a model given an input size.
    
    Args:
        model: PyTorch model
        input_data: dummy input data (assumed batch size 1) for the model to compute GFLOPs
        
    Returns:
        gflops: Computed GFLOPs
        params: Number of parameters in the model
    """
    device = next(model.parameters()).device
    
    # Use torchinfo to compute GFLOPs
    model_summary = summary(
        model,
        input_data=input_data,
        col_names=["output_size", "num_params", "mult_adds"],
        verbose=0,
        device=device
    )
    # Compute GFLOPs and parameters
    gflops = model_summary.total_mult_adds / 1e9  # Convert to GFLOPs
    params = model_summary.total_params

    print(f"Model GFLOPs: {gflops:.2f} GFLOPs")
    print(f"Model Parameters: {params / 1e6:.2f} M parameters")

    return gflops, params

def compute_classification_metrics(model, dataloader, device='cuda'):
    """
    Computes classification metrics from model predictions.

    Returns:
        A dictionary with the following keys:
            - 'confusion_matrix': Raw confusion matrix
            - 'normalized_confusion_matrix': Row-normalized confusion matrix
            - 'accuracy': Overall accuracy (float)
            - 'per_class_accuracy': Accuracy per class (np.array)
            - 'classification_report': Sklearn classification report (string)
            - 'predictions': Model predictions (np.array)
            - 'true_labels': Ground truth labels (np.array)
            - 'probabilities': Predicted class probabilities (np.array)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            images, audios, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            images = images.to(dtype=torch.float32)
            audios = audios.to(dtype=torch.float32)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images, audios)

            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy = 100 * (all_preds == all_labels).mean()
    per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
    report = classification_report(all_labels, all_preds, digits=4)

    return {
        'confusion_matrix': cm,
        'normalized_confusion_matrix': cm_normalized,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'classification_report': report,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }

def pretrain_dino(model, trainloader, dino_loss, align=False, num_epochs=100, learning_rate=0.0001, 
                  save_path='pretrained_dino.pt', log_path='pretrain_log.csv'):
    
    # Ensure necessary directories exist
    for path in [save_path, log_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    device = next(model.parameters()).device
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Setup logging
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    save_path = save_path.replace('.pt', f'_{start_time_str}.pt')
    log_path = log_path.replace('.csv', f'_{start_time_str}.csv')

    train_info = {
        'start_time': start_time,
        'learning_rate': learning_rate,
        'batch_size': trainloader.batch_size,
        'epochs': num_epochs,
        'model_name': 'MultiModalDINO'
    }
    
    # Create log file
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', f"# {json.dumps(train_info)}"])
    
    best_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # For mixed precision training
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            global_images, global_audios, local_images, local_audios = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                student_out, teacher_out = model((global_images, global_audios, local_images, local_audios))
                if align:
                    alignment_loss = model.student.loss_align  # Extract alignment loss
                    loss = dino_loss(student_out, teacher_out, alignment_loss, tau_s=0.1, tau_t=0.04)
                else:
                    loss = dino_loss(student_out, teacher_out, tau_s=0.1, tau_t=0.04)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            model.update_teacher()  # Update teacher network using momentum encoder
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        
        # Log epoch results
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss])
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f'Saved best model with loss: {best_loss:.4f}')
    
    return model

def train_downstream(pretrained_model, trainloader, validloader, testloader, num_epochs=10,
                    device='cuda', learning_rate=0.001, save_path='downstream_model.pt',
                    train_log_path='downstream_train_log.csv', test_log_path='downstream_test_log.csv', 
                    is_dino_based=True):
    
    # Ensure necessary directories exist
    for path in [save_path, train_log_path, test_log_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    model = DownstreamClassifier(pretrained_model, is_dino_based=is_dino_based).to(device)
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Setup logging
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time_str = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    save_path = save_path.replace('.pt', f'_{start_time_str}.pt')
    train_log_path = train_log_path.replace('.csv', f'_{start_time_str}.csv')
    test_log_path = test_log_path.replace('.csv', f'_{start_time_str}.csv')

    train_info = {
        'start_time': start_time,
        'learning_rate': learning_rate,
        'batch_size': trainloader.batch_size,
        'epochs': num_epochs,
        'criterion': 'CrossEntropyLoss',
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'model_name': 'DinoClassifier'
    }
    
    # Create train log file
    with open(train_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 
                        f"# {json.dumps(train_info)}"])
    
    best_val_acc = 0
    scaler = torch.cuda.amp.GradScaler(enabled=True) # For mixed precision training
    
    @torch.no_grad()
    def evaluate(model, dataloader):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in dataloader:
            images, audios, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            images = images.to(dtype=torch.float32)
            audios = audios.to(dtype=torch.float32)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images, audios)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return (total_loss / len(dataloader), 100 * correct / total, 
                np.array(all_labels), np.array(all_preds), np.array(all_probs))
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            images, audios, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            optimizer.zero_grad()

            images = images.to(dtype=torch.float32)
            audios = audios.to(dtype=torch.float32)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images, audios)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        train_loss = total_loss / num_batches
        val_loss, val_acc, _, _, _ = evaluate(model, validloader)
        
        # Log epoch results
        with open(train_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_acc])
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': best_val_acc,
            }, save_path)
            print(f'Saved best model with validation accuracy: {best_val_acc:.2f}%')
    
    # Evaluate on test set using best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_labels, test_preds, test_probs = evaluate(model, testloader)
    
    # Save test results
    with open(test_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_label', 'predicted_label', 'probabilities'])
        for label, pred, prob in zip(test_labels, test_preds, test_probs):
            writer.writerow([label, pred, ','.join(map(str, prob))])
    
    print(f'\nTest Accuracy: {test_acc:.2f}%')
    return model

def feature_extraction_loop(device, model, dataloader):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, spectrograms, labels = batch
            images = images.float().to(device)
            spectrograms = spectrograms.float().to(device)
            
            features = model(images, spectrograms)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_features), np.concatenate(all_labels)

def train_knn_classifier(pretrained_dino, train_dataloader, test_dataloader, n_neighbors=5, device='cuda', is_dino_based=True):
    # Create feature extractor
    feature_extractor = FeatureExtractor(pretrained_dino, is_dino_based=is_dino_based).to(device)
    
    # Extract features from training and test sets
    print("Extracting training features...")
    train_features, train_labels = feature_extraction_loop(device, feature_extractor, train_dataloader)
    
    print("Extracting test features...")
    test_features, test_labels = feature_extraction_loop(device, feature_extractor, test_dataloader)
    
    # Train KNN classifier
    print("Training KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_features, train_labels)
    
    # Evaluate
    accuracy = 100 * knn.score(test_features, test_labels)
    print(f"KNN Accuracy (k={n_neighbors}): {accuracy:.4f}%")
    
    return knn, accuracy