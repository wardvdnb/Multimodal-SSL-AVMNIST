
from utils.reproducibility import set_seed
set_seed(1)
import torch
import os
import torch.nn as nn
import torch.optim as optim
from models.dino import SpectrogramEncoder
import torch.nn.functional as F
import lightning.pytorch as pl
from models.dino import ImageEncoder, SpectrogramEncoder, \
                        ProjectionHead

class InfoNCEModel(nn.Module):
    def __init__(self, output_dim=256, projection_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.projection_dim = projection_dim
        self.image_encoder = ImageEncoder(output_dim=output_dim)
        self.audio_encoder = SpectrogramEncoder(output_dim=output_dim)
        self.image_projection_head = ProjectionHead(output_dim, projection_dim)
        self.audio_projection_head = ProjectionHead(output_dim, projection_dim)
        
    def forward(self, batch):
        images, spectrograms, _ = batch # last is labels (not used)
        images = images.float()
        spectrograms = spectrograms.float()
    
        image_features = self.image_encoder(images, None)
        audio_features = self.audio_encoder(None, spectrograms)
        
        image_features = self.image_projection_head(image_features)
        audio_features = self.audio_projection_head(audio_features)

        return image_features, audio_features

# Unified Lightning module for any DINO model
class MultiModalInfoNCELightning(pl.LightningModule):
    def __init__(
        self,
        projection_dim=256,
        output_dim=256,
        learning_rate=0.0001,
        num_epochs=100,
        use_mixed_precision=True,
    ):
        """
        PyTorch Lightning module for MultiModal DINO model pretraining
        
        Args:
            encoder_class: Encoder class to use if dino_model is None
            encoder_kwargs: Kwargs for encoder if dino_model is None
            projection_dim: Dimension of projection features
            output_dim: Dimension of output features
            learning_rate: Learning rate for optimizer
            use_mixed_precision: Whether to use mixed precision training
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.projection_dim = projection_dim

        self.model = InfoNCEModel(output_dim=output_dim, projection_dim=projection_dim)
        
        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
        self.num_epochs = num_epochs
        
        # Save hyperparameters to be logged
        self.save_hyperparameters()
        
    def forward(self, batch):
       return self.model(batch) # image_features, audio_features
    
    def infoNCE_loss(self, image_outputs, audio_outputs, temperature=0.07):
        """
        Compute infoNCE loss for audio-visual alignment
        
        Args:
            image_outputs: Image features from projection head [batch_size, projection_dim]
            audio_outputs: Audio features from projection head [batch_size, projection_dim]
            temperature: Temperature parameter for softmax (default: 0.07)
        
        Returns:
            Loss value (scalar)
        """
        # Get batch size
        batch_size = image_outputs.size(0)
        
        # Normalize feature vectors for better similarity computation 
        # (Normalized vectors mean their dot product equals cosine similarity) 
        image_outputs = F.normalize(image_outputs, p=2, dim=1)
        audio_outputs = F.normalize(audio_outputs, p=2, dim=1)
        
        # Compute cosine similarity between all pairs
        # [batch_size, batch_size] matrix where sim[i, j] = cosine similarity between 
        # image[i] and audio[j]
        sim_matrix = torch.mm(image_outputs, audio_outputs.T) / temperature
        
        # Labels: diagonal elements are positive pairs (same class)
        # This assumes paired samples are at the same index in the batch
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # Compute loss from both directions (image→audio and audio→image)
        # Cross-entropy loss with one-hot encoded positive pairs
        loss_i2a = F.cross_entropy(sim_matrix, labels)
        loss_a2i = F.cross_entropy(sim_matrix.T, labels)
        
        # Symmetric loss
        loss = (loss_i2a + loss_a2i) / 2.0
        
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (images, spectrograms, labels)
            batch_idx: Index of the batch
        """
        # Forward pass through model
        image_out, audio_out = self(batch)
        
        # Calculate InfoNCE loss
        loss = self.infoNCE_loss(image_out, audio_out)
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }