import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl
import torch
import torch.optim as optim
from torch.nn import functional as F

# Vision Transformer implementation
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=28, patch_size=4, in_channels=1, embed_dim=192):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class AudioPatchEmbedding(nn.Module):
    def __init__(self, spectrogram_size=112, patch_size=8, in_channels=1, embed_dim=192):
        super().__init__()
        self.spectrogram_size = spectrogram_size
        self.patch_size = patch_size
        self.num_patches = (spectrogram_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class ViTEncoder(nn.Module):
    def __init__(
        self, 
        image_size=28, 
        patch_size=4, 
        in_channels=1, 
        embed_dim=192, 
        depth=4, 
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed  # (B, num_patches + 1, embed_dim)
        
        # Apply transformer
        x = self.transformer(x)  # (B, num_patches + 1, embed_dim)
        
        # Use class token for representation
        return x[:, 0]  # (B, embed_dim)

class AudioViTEncoder(nn.Module):
    def __init__(
        self, 
        spectrogram_size=112, 
        patch_size=8, 
        in_channels=1, 
        embed_dim=192, 
        depth=4, 
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        # Patch embedding
        self.patch_embed = AudioPatchEmbedding(
            spectrogram_size=spectrogram_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed  # (B, num_patches + 1, embed_dim)
        
        # Apply transformer
        x = self.transformer(x)  # (B, num_patches + 1, embed_dim)
        
        # Use class token for representation
        return x[:, 0]  # (B, embed_dim)

class MultiModalViTEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        
        # Image encoder for MNIST (28x28)
        self.image_encoder = ViTEncoder(
            image_size=28,
            patch_size=4,
            in_channels=1,
            embed_dim=192,
            depth=4,
            num_heads=3
        )
        
        # Audio encoder for FSDD spectrograms (112x112)
        self.audio_encoder = AudioViTEncoder(
            spectrogram_size=112,
            patch_size=8,
            in_channels=1,
            embed_dim=192,
            depth=4,
            num_heads=3
        )
        
        # Fusion and projection
        self.fusion = nn.Sequential(
            nn.Linear(384, 512),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, images, spectrograms):
        image_features = self.image_encoder(images)
        audio_features = self.audio_encoder(spectrograms)
        # Concatenate features from both modalities
        combined = torch.cat([image_features, audio_features], dim=1)
        return self.fusion(combined)

class MultiModalViTDINO(nn.Module):
    def __init__(self, projection_dim=256, output_dim=256, momentum=0.996,
                 center_momentum=0.9):
        super().__init__()
        
        self.student = MultiModalViTEncoder(output_dim)
        self.teacher = MultiModalViTEncoder(output_dim)
        self.teacher.load_state_dict(self.student.state_dict())
        
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, projection_dim)
        )
        
        self.teacher_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, projection_dim)
        )
        
        self.teacher_projection.load_state_dict(self.student_projection.state_dict())
        for param in self.teacher_projection.parameters():
            param.requires_grad = False
            
        self.momentum = momentum
        
        # Initialize center for teacher outputs
        self.register_buffer("center", torch.zeros(1, projection_dim))
        self.center_momentum = center_momentum
    
    @torch.no_grad()
    def update_teacher(self):
        """Update teacher network using momentum update"""
        for student_param, teacher_param in zip(self.student.parameters(), 
                                              self.teacher.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + \
                                (1 - self.momentum) * student_param.data
                                
        for student_param, teacher_param in zip(self.student_projection.parameters(),
                                              self.teacher_projection.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + \
                                (1 - self.momentum) * student_param.data
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center for teacher outputs using momentum update"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)
    
    def forward(self, batch):
        """
        Args:
            batch: tuple of (global_images, global_audios, local_images, local_audios)
            where:
                - global_images: tensor of shape [batch_size, channels, height, width]
                - global_audios: tensor of shape [batch_size, channels, height, width]
                - local_images: tensor of shape [batch_size, n_local_views, channels, height, width]
                - local_audios: tensor of shape [batch_size, n_local_views, channels, height, width]
        """
        global_images, global_audios, local_images, local_audios = batch
        device = next(self.student.parameters()).device
        
        # Move all tensors to device at once
        global_images = global_images.to(device)
        global_audios = global_audios.to(device)
        local_images = local_images.to(device)
        local_audios = local_audios.to(device)
        
        batch_size = global_images.shape[0]
        n_global_views = global_images.shape[1]
        n_local_views = local_images.shape[1]
        
        # Process student global views
        student_global_features = []
        for view_idx in range(n_global_views):
            current_global_images = global_images[:, view_idx, :, :, :]  # [batch_size, channels, height, width]
            current_global_audios = global_audios[:, view_idx, :, :, :]  # [batch_size, channels, height, width]
            features = self.student(current_global_images, current_global_audios)
            student_global_features.append(features)
        
        # Process student local views - reshape and process each view
        student_local_features = []
        for view_idx in range(n_local_views):
            # Extract the current view for all batch items
            current_local_images = local_images[:, view_idx, :, :, :]  # [batch_size, channels, height, width]
            current_local_audios = local_audios[:, view_idx, :, :, :]  # [batch_size, channels, height, width]
            # Process through the student encoder
            features = self.student(current_local_images, current_local_audios)
            student_local_features.append(features)
        
        # Combine all student features
        student_features = torch.cat(student_global_features + student_local_features)

       # Process teacher views (global only)
        with torch.no_grad():
            teacher_features = []
            for view_idx in range(n_global_views):
                current_global_images = global_images[:, view_idx, :, :, :]
                current_global_audios = global_audios[:, view_idx, :, :, :]
                features = self.teacher(current_global_images, current_global_audios)
                teacher_features.append(features)
            
            teacher_features = torch.cat(teacher_features)
        
        # Project features
        student_projs = self.student_projection(student_features)
        
        with torch.no_grad():
            teacher_projs = self.teacher_projection(teacher_features)
            center = self.center.to(teacher_projs.device)
            teacher_projs_c = teacher_projs - center
            
            # Update center using mean of teacher projections
            self.update_center(teacher_projs)
        
        student_outputs = student_projs.view(n_local_views + n_global_views, batch_size, -1)
        teacher_outputs = teacher_projs_c.view(n_global_views, batch_size, -1)
        return student_outputs, teacher_outputs

class MultiModalViTDINOLightning(pl.LightningModule):
    def __init__(
        self, 
        projection_dim=256,
        output_dim=256,
        momentum=0.996,
        center_momentum=0.9,
        student_temperature=0.1,
        teacher_temperature=0.04,
        learning_rate=0.0001,
        use_mixed_precision=True,
        # ViT image encoder parameters
        image_embed_dim=192,
        image_depth=4,
        image_num_heads=3,
        image_dropout=0.1,
        # ViT audio encoder parameters
        audio_embed_dim=192,
        audio_depth=4,
        audio_num_heads=3,
        audio_dropout=0.1
    ):
        """
        PyTorch Lightning module for MultiModal ViT DINO model pretraining
        
        Args:
            projection_dim: Dimension of projection features
            output_dim: Dimension of output features
            momentum: Momentum value for teacher update
            center_momentum: Momentum for center update
            student_temperature: Temperature parameter for student network
            teacher_temperature: Temperature parameter for teacher network
            learning_rate: Learning rate for optimizer
            use_mixed_precision: Whether to use mixed precision training
            image_embed_dim: Embedding dimension for image encoder
            image_depth: Number of transformer layers for image encoder
            image_num_heads: Number of attention heads for image encoder
            image_dropout: Dropout rate for image encoder
            audio_embed_dim: Embedding dimension for audio encoder
            audio_depth: Number of transformer layers for audio encoder
            audio_num_heads: Number of attention heads for audio encoder
            audio_dropout: Dropout rate for audio encoder
        """
        super().__init__()
        
        # Create a custom MultiModalViTEncoder with the provided parameters
        class CustomMultiModalViTEncoder(nn.Module):
            def __init__(self, projection_dim):
                super().__init__()
                
                self.projection_dim = projection_dim
                
                # Image encoder for MNIST (28x28)
                self.image_encoder = ViTEncoder(
                    image_size=28,
                    patch_size=4,
                    in_channels=1,
                    embed_dim=image_embed_dim,
                    depth=image_depth,
                    num_heads=image_num_heads,
                    dropout=image_dropout
                )
                
                # Audio encoder for FSDD spectrograms (112x112)
                self.audio_encoder = AudioViTEncoder(
                    spectrogram_size=112,
                    patch_size=8,
                    in_channels=1,
                    embed_dim=audio_embed_dim,
                    depth=audio_depth,
                    num_heads=audio_num_heads,
                    dropout=audio_dropout
                )
                
                # Calculate total input dimension for fusion
                total_embed_dim = image_embed_dim + audio_embed_dim
                
                # Fusion and projection
                self.fusion = nn.Sequential(
                    nn.Linear(total_embed_dim, 512),
                    nn.GELU(),
                    nn.Linear(512, projection_dim)
                )
                
            def forward(self, images, spectrograms):
                image_features = self.image_encoder(images)
                audio_features = self.audio_encoder(spectrograms)
                # Concatenate features from both modalities
                combined = torch.cat([image_features, audio_features], dim=1)
                return self.fusion(combined)
        
        # Create student and teacher encoders
        student_encoder = CustomMultiModalViTEncoder(output_dim)
        teacher_encoder = CustomMultiModalViTEncoder(output_dim)
        teacher_encoder.load_state_dict(student_encoder.state_dict())
        
        # Freeze teacher parameters
        for param in teacher_encoder.parameters():
            param.requires_grad = False
        
        # Create projections
        student_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, projection_dim)
        )
        
        teacher_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, projection_dim)
        )
        
        teacher_projection.load_state_dict(student_projection.state_dict())
        for param in teacher_projection.parameters():
            param.requires_grad = False
        
        # Create the full DINO model
        self.model = MultiModalViTDINO(
            projection_dim=projection_dim,
            output_dim=output_dim,
            momentum=momentum,
            center_momentum=center_momentum
        )
        
        # Replace the default encoders and projections with our customized ones
        self.model.student = student_encoder
        self.model.teacher = teacher_encoder
        self.model.student_projection = student_projection
        self.model.teacher_projection = teacher_projection
        
        # Store parameters
        self.learning_rate = learning_rate
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.use_mixed_precision = use_mixed_precision
        
        # Save hyperparameters to be logged
        self.save_hyperparameters()
        
    def forward(self, batch):
        """Forward pass (used for inference)"""
        return self.model(batch)
    
    def dino_loss(self, student_outputs, teacher_outputs):
        """
        Compute DINO loss between student and teacher outputs.
        
        for each pair of views: for every sample in batch, calculate the 
        loss based on values in the projection_dim, after summing these the shape is: [batch_size]
        we take the mean of these values to get the loss for this pair of views and continue for all pairs
        after that normalizing by the number of pairs (n_student_views * n_teacher_views)

        Args:
            student_outputs: Tensor of shape [n_views, batch_size, projection_dim]
            teacher_outputs: Tensor of shape [n_teacher_views, batch_size, projection_dim]
        """
        n_student_views = student_outputs.shape[0]
        n_teacher_views = teacher_outputs.shape[0]

        # Pre-compute all probabilities
        teacher_probs = F.softmax(teacher_outputs / self.teacher_temperature, dim=-1)
        student_probs = F.log_softmax(student_outputs / self.student_temperature, dim=-1)

        total_loss = 0
        # Compute loss for each student view against each teacher view
        for student_view_idx in range(n_student_views):
            for teacher_view_idx in range(n_teacher_views):
                view_loss = -(teacher_probs[teacher_view_idx] * student_probs[student_view_idx]).sum(dim=-1).mean()
                total_loss += view_loss
        
        return total_loss / (n_student_views * n_teacher_views)
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (global_images, global_audios, local_images, local_audios)
            batch_idx: Index of the batch
        """
        global_images, global_audios, local_images, local_audios = batch
        
        # Forward pass through model
        student_out, teacher_out = self.model((global_images, global_audios, local_images, local_audios))
        
        # Calculate loss
        loss = self.dino_loss(student_out, teacher_out)
        
        # Update teacher network using momentum encoder
        self.model.update_teacher()
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"  # Monitors train loss for LR scheduling
            }
        }

class DownstreamClassifier(nn.Module):
    def __init__(self, pretrained_dino, num_classes=10):
        super().__init__()
        self.encoder = pretrained_dino.student
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, images, spectrograms):
        with torch.no_grad():
            features = self.encoder(images, spectrograms)
        return self.classifier(features)

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_dino):
        super().__init__()
        self.encoder = pretrained_dino.student
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, images, spectrograms):
        with torch.no_grad():
            features = self.encoder(images, spectrograms)
        return features