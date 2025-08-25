from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy

class UnimodalLightningModule(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=0.001, modalnum=0, num_epochs=100):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.modalnum = modalnum  # 0 for image modality, 1 for audio modality
        self.num_epochs = num_epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.model = None # Placeholder for the model, to be defined in subclasses
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Handle the image_audio_label tuple format
        image_audio_label = batch
        modality = image_audio_label[self.modalnum].float()
        label = image_audio_label[-1]
        
        # Forward pass
        output = self(modality)
        loss = self.criterion(output, label)
        
        # Calculate accuracy
        # Pass raw logits directly, softmax is used internally
        acc = self.train_accuracy(output, label)
        
        # Log metrics
        # NOTE: this doesn't work with on_step=False/on_epoch=True for some reason
        # have tried with initializing dummy losses and accs but get 
        # "ValueError: dict contains fields not in fieldnames: 'train_acc', 'train_loss'"
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image_audio_label = batch
        modality = image_audio_label[self.modalnum].float()
        label = image_audio_label[-1]
        
        output = self(modality)
        loss = self.criterion(output, label)
        acc = self.val_accuracy(output, label)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        image_audio_label = batch
        modality = image_audio_label[self.modalnum].float()
        label = image_audio_label[-1]
        
        output = self(modality)
        loss = self.criterion(output, label)
        acc = self.test_accuracy(output, label)
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs) # TODO: look into warmup for ADAM
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"  # Monitors validation loss for scheduler steps
            }
        }
    
class UnimodalImage(UnimodalLightningModule):
    def __init__(self, dropout_prob=0.5, with_head=True, num_classes=10, learning_rate=0.001, num_epochs=100):
        super().__init__(num_classes=num_classes, learning_rate=learning_rate, modalnum=0, num_epochs=num_epochs)
        
        self.with_head = with_head

        self.model = CentralUnimodalImage(dropout_prob=dropout_prob, with_head=with_head)

class UnimodalAudio(UnimodalLightningModule):
    def __init__(self, dropout_prob=0.5, with_head=True, num_classes=10, learning_rate=0.001, num_epochs=100):
        super().__init__(num_classes=num_classes, learning_rate=learning_rate, modalnum=1, num_epochs=num_epochs)
        
        self.with_head = with_head

        self.model = CentralUnimodalAudio(dropout_prob=dropout_prob, with_head=with_head)

class CentralUnimodalImage(nn.Module):
    """Implements a Central Unimodal Image Model."""
    def __init__(self, dropout_prob=0.5, with_head=False):
        super(CentralUnimodalImage, self).__init__()
        
        self.with_head = with_head

        # Conv1: Input is 1x28x28, output will be 32x14x14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv2: Input is 32x14x14, output will be 64x7x7
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Dropout layer (used after the second convolution)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layers: Flatten the 64x5x5 into a vector of size 64*5*5
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 10)  # 10 output units for MNIST classes
    
    def forward(self, x):
        # Conv1 + BatchNorm + ReLU + MaxPool
        x = self.conv1(x) # 28x28
        x = self.bn1(x)
        x = F.relu(x)
        # x = F.avg_pool2d(x, 2) # 14x14
        x = F.max_pool2d(x, 2) # 14x14 NOTE: original LeNet used avg pooling

        # Conv2 + BatchNorm + ReLU + MaxPool
        x = self.conv2(x) # 10x10
        x = self.bn2(x)
        x = F.relu(x)
        # x = F.avg_pool2d(x, 2) # 5x5
        x = F.max_pool2d(x, 2)  # 5x5

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1) # (1, 5x5x64) = (1, 1600)
        
        if(self.with_head):
            # Fully connected layer 1 + Dropout
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            
            # Fully connected layer 2 (final output)
            x = self.fc2(x)
        
        return x

class CentralUnimodalAudio(nn.Module):
    """Implements a Central Unimodal Audio Model."""
    def __init__(self, dropout_prob=0.5, with_head=False):
        super(CentralUnimodalAudio, self).__init__()
        
        self.with_head = with_head

        # Conv1: Input is 1x112x112, output will be 8x56x56
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Conv2: Input is 8x56x56, output will be 16x28x28
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)

        # Conv3: Input is 16x28x28, output will be 32x14x14
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Conv4: Input is 32x14x14, output will be 64x7x7
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(64)

        # Dropout layer (used after the second convolution)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layers: Flatten the 64x7x7 into a vector of size 64*7*7
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)  # 10 output units for MNIST classes
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = F.avg_pool2d(x, 2)
        x = F.max_pool2d(x, 2) # 56x56 NOTE: original LeNet used avg pooling

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = F.avg_pool2d(x, 2)
        x = F.max_pool2d(x, 2) # 28x28

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # x = F.avg_pool2d(x, 2)
        x = F.max_pool2d(x, 2) # 14x14

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        # x = F.avg_pool2d(x, 2)
        x = F.max_pool2d(x, 2)  # 7x7

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1) # (1, 7x7x64)
        
        if(self.with_head):
            # Fully connected layer 1 + Dropout
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            
            # Fully connected layer 2 (final output)
            x = self.fc2(x)
        
        return x