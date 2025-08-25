import copy
import numpy as np
from utils.get_data import AVMNISTDataModule, get_dataloader_augmented
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import ASTModel
import lightning.pytorch as pl
import torch.optim as optim
from models.dino_vit import AudioViTEncoder, ViTEncoder
from models.unimodal import CentralUnimodalAudio, CentralUnimodalImage
from models.mini_resnet import MiniResNet
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torchvision.models.resnet import resnet18
from tqdm import tqdm

#------------------------------ PARTIAL ENCODERS ------------------------------ 

def image_encoder(output_dim):
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, output_dim),

        # alternatively:
        # nn.Flatten(),
        # nn.Linear(128 * 3 * 3, output_dim),
    )

def audio_encoder(output_dim):
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),                    # [B, 256]
        nn.Linear(256, output_dim),

        # Alternatively:
        # nn.Flatten(),
        # nn.Linear(256 * 7 * 7, output_dim),
    )

class LSTMImageEncoder(nn.Module):
    def __init__(self, output_dim=256, proj_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 32, 28, 28]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 14, 14]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 64, 7, 7]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [B, 128, 7, 7]
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.feature_proj = nn.Sequential(
            nn.Linear(128, proj_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=output_dim // 2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x = self.cnn(x)                            # [B, 128, 7, 7]
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)       # → [B, 49, 128]
        x = self.feature_proj(x)                   # → [B, 49, proj_dim]
        x, _ = self.lstm(x)                        # → [B, 49, output_dim]
        x = torch.mean(x, dim=1)  # Average over all timesteps:
                                  # HXW = 7x7=49, which is seen as a "sequence of tokens" 
                                  # and each is considered a timestep
        return x

class LSTMAudioEncoder(nn.Module):
    def __init__(self, output_dim=256, proj_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B, 32, 112, 112]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 32, 56, 56]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # [B, 64, 56, 56]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 64, 28, 28]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# [B, 128, 28, 28]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 128, 14, 14]
        )

        self.feature_proj = nn.Sequential(
            nn.Linear(128, proj_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=output_dim // 2,
            batch_first=True,
            bidirectional=True #NOTE: bidirectional=True means the real output size is hidden_size * 2
        )

    def forward(self, x):
        x = self.cnn(x)                            # [B, 128, 14, 14]
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)       # → [B, 196, 128]
        x = self.feature_proj(x)                   # → [B, 196, proj_dim]
        x, _ = self.lstm(x)                        # → [B, 196, output_dim]
        x = torch.mean(x, dim=1)  # Average over all timesteps
        return x

# # LSTM-based encoder for audio
    # def lstm_audio_encoder(output_dim=256):
    #     return nn.Sequential(
    #         # CNN layers for spatial feature extraction
    #         nn.Conv2d(1, 32, kernel_size=(3, 3)),  # Output: (32, 110, 110)
    #         nn.BatchNorm2d(32),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=(2, 2)),  # Output: (32, 55, 55)
            
    #         nn.Conv2d(32, 64, kernel_size=(3, 3)),  # Output: (64, 53, 53)
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=(2, 2)),  # Output: (64, 26, 26)
            
    #         nn.Conv2d(64, 128, kernel_size=(3, 3)),  # Output: (128, 24, 24)
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=(2, 2)),  # Output: (128, 12, 12)
            
    #         # Flatten and reshape for RNN
    #         nn.Flatten(start_dim=1, end_dim=2),  # Output: (128 * 12, 12)
    #         nn.Linear(12, 64),  # Reduce feature dimension for RNN input
    #         nn.ReLU(),
            
    #         # RNN (LSTM) for temporal feature extraction
    #         #NOTE: bidirectional=True means the real output size is hidden_size * 2
    #         nn.LSTM(input_size=64, hidden_size=output_dim // 2, batch_first=True, bidirectional=True),
    #     )

class MobileVitEncoder(nn.Module):
    def __init__(self, output_dim=256, pretrained=False):
        super().__init__()
        # MobileViT backbone (Modified MobileNetV3 for grayscale input)
        mobilenet = mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        # Modify first conv layer for single-channel input
        mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # Remove classifier head and replace with projection layer
        mobilenet.classifier = nn.Identity()  # Remove MobileNet's original classifier

        # Projection layer with dropout
        self.encoder = nn.Sequential(
            mobilenet,
            nn.Linear(576, 256),  # MobileNetV3 feature dim
            nn.ReLU(),
            nn.Linear(256, output_dim)  # Final embedding dim
        )
        
    def forward(self, x):
        x = self.encoder(x)
        return x

class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=256, pretrained=False):
        super().__init__()
        # ResNet backbone
        resnet = resnet18(weights="DEFAULT" if pretrained else None)
        # Modify first conv layer for single-channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove classifier head and replace with projection layer
        resnet.fc = nn.Identity()  # Remove ResNet's original classifier

        self.encoder = nn.Sequential(
            resnet,
            nn.Linear(512, 256),  # ResNet feature dim
            nn.ReLU(),
            nn.Linear(256, output_dim)  # Final embedding dim
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x
#----------------------------- MULTIMODAL ENCODERS ----------------------------

# Base class for all multimodal encoders
class BaseMultiModalEncoder(nn.Module):
    def __init__(self, output_dim=256, encoder_output_dim=512, fusion_dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.encoder_output_dim = encoder_output_dim
        self.fusion_dropout = fusion_dropout
        
    def forward(self, images, spectrograms):
        raise NotImplementedError("Subclasses must implement forward method")

# Basic concatenation encoder
class SimpleMultiModalEncoder(BaseMultiModalEncoder):
    def __init__(self, output_dim=256, encoder_output_dim=512, fusion_dropout=0.3):
        super().__init__(output_dim, encoder_output_dim, fusion_dropout)

        self.image_encoder = image_encoder(output_dim=encoder_output_dim)
        self.audio_encoder = audio_encoder(output_dim=encoder_output_dim)
        
        # Fusion and projection
        self.fusion = nn.Sequential(
            nn.Linear(2*encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),  # Add dropout
            nn.Linear(encoder_output_dim, output_dim)
        )
        
    def forward(self, images, spectrograms):
        image_features = self.image_encoder(images)
        audio_features = self.audio_encoder(spectrograms)
        # Concatenate features from both modalities
        combined = torch.cat([image_features, audio_features], dim=1)
        return self.fusion(combined)

# Gated encoder with simple CNN encoders
class GatedMultiModalEncoder(SimpleMultiModalEncoder):
    def __init__(self, output_dim=256, encoder_output_dim=512):
        super().__init__(output_dim, encoder_output_dim)
        
        # Learnable gates
        self.gate_image = nn.Parameter(torch.tensor(0.5))
        self.gate_audio = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, images, spectrograms):
        image_features = self.image_encoder(images)
        audio_features = self.audio_encoder(spectrograms)

        # Apply gating mechanism
        gate_image = torch.sigmoid(self.gate_image)
        gate_audio = torch.sigmoid(self.gate_audio)

        gated_image_features = gate_image * image_features
        gated_audio_features = gate_audio * audio_features

        # Concatenate gated features
        combined = torch.cat([gated_image_features, gated_audio_features], dim=1)

        return self.fusion(combined)

# LSTM-based multimodal encoder
class LSTMMultiModalEncoder(SimpleMultiModalEncoder):
    """
    Variant of SimpleMultiModalEncoder that uses a CNN + LSTM based encoder
    """
    def __init__(self, output_dim=256, encoder_output_dim=512):
        super().__init__(output_dim, encoder_output_dim)

        self.image_encoder = LSTMImageEncoder(output_dim=encoder_output_dim)
        self.audio_encoder = LSTMAudioEncoder(output_dim=encoder_output_dim)

# ViT-based multimodal encoder (using only audio ViT)
class ViTMultiModalEncoder(SimpleMultiModalEncoder):
    def __init__(self, output_dim=256, encoder_output_dim=512):
        super().__init__(output_dim, encoder_output_dim)
        
        # Audio encoder with ViT
        self.audio_encoder = AudioViTEncoder(
            spectrogram_size=112,
            patch_size=8,
            in_channels=1,
            embed_dim=encoder_output_dim,
            depth=4,
            num_heads=4
        )

# Dual ViT-based multimodal encoder (both image and audio are ViTs)
class DualViTMultiModalEncoder(GatedMultiModalEncoder):
    def __init__(self, 
        output_dim=256,
        encoder_output_dim=512,
        image_size=28,
        image_patch_size=4,
        # image_embed_dim=192, #TODO: remove this if never used
        image_depth=4,
        # image_num_heads=3,
        spectrogram_size=112,
        audio_patch_size=8,
        # audio_embed_dim=192,
        audio_depth=4,
        # audio_num_heads=3,
        dropout=0.1):
        super().__init__(output_dim)
        
        image_num_heads = max(1, encoder_output_dim // 64)
        audio_num_heads = max(1, encoder_output_dim // 64)

        # Learnable gates
        # self.gate_image = nn.Parameter(torch.tensor(0.5))
        # self.gate_audio = nn.Parameter(torch.tensor(0.5))
        
        # Image encoder with ViT
        self.image_encoder = ViTEncoder(
            image_size=image_size,
            patch_size=image_patch_size,
            in_channels=1,
            embed_dim=encoder_output_dim,
            depth=image_depth,
            num_heads=image_num_heads,
            dropout=dropout
        )
        
        # Audio encoder with ViT
        self.audio_encoder = AudioViTEncoder(
            spectrogram_size=spectrogram_size,
            patch_size=audio_patch_size,
            in_channels=1,
            embed_dim=encoder_output_dim,
            depth=audio_depth,
            num_heads=audio_num_heads,
            dropout=dropout
        )
        
        # Total dimension of concatenated features
        # total_dim = image_embed_dim + audio_embed_dim

        total_dim = 2 * encoder_output_dim

        # Fusion and projection
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    # def forward(self, images, spectrograms):
    #     image_features = self.image_encoder(images)
    #     audio_features = self.audio_encoder(spectrograms)

    #     # Apply gating mechanism
    #     # gate_image = torch.sigmoid(self.gate_image)
    #     # gate_audio = torch.sigmoid(self.gate_audio)

    #     # gated_image_features = gate_image * image_features
    #     # gated_audio_features = gate_audio * audio_features

    #     # Concatenate gated features
    #     combined = torch.cat([image_features, audio_features], dim=1)

    #     return self.fusion(combined)

class MobileViTMultiModalEncoder(SimpleMultiModalEncoder):

    def __init__(self, output_dim=256, encoder_output_dim=512, pretrained=False):
        super().__init__(output_dim, encoder_output_dim)

        # Image encoder with MobileViT
        self.image_encoder = MobileVitEncoder(output_dim=encoder_output_dim, pretrained=pretrained)
        
        # Audio encoder with MobileViT
        self.audio_encoder = MobileVitEncoder(output_dim=encoder_output_dim, pretrained=pretrained)
    
class ResNetMultiModalEncoder(GatedMultiModalEncoder):

    def __init__(self, output_dim=256, encoder_output_dim=512, pretrained=False):
        super().__init__(output_dim, encoder_output_dim)

        # Image encoder with ResNet
        self.image_encoder = ResNetEncoder(output_dim=encoder_output_dim, pretrained=pretrained)
        
        # Audio encoder with ResNet
        self.audio_encoder = ResNetEncoder(output_dim=encoder_output_dim, pretrained=pretrained)

# Cross-modal attention module (used in CrossAttentionMultiModalEncoder)
class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.scale = dim ** -0.5  # Fixed scaling factor

    def forward(self, x1, x2):
        # x1: modality 1 (e.g., MNIST features)
        # x2: modality 2 (e.g., spectrogram features)
        q = self.q_proj(x1)  # [B, D]
        k, v = self.kv_proj(x2).chunk(2, dim=-1)  # [B, D], [B, D]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, B]
        attn = attn.softmax(dim=-1)  # Normalize attention scores
        
        # Apply attention to values and add residual
        return x1 + attn @ v  # [B, D]
        
# Cross-modal attention encoder
class CrossAttentionMultiModalEncoder(SimpleMultiModalEncoder):
    def __init__(self, output_dim=256, encoder_output_dim=512, fusion_dropout=0.3):
        super().__init__(output_dim, encoder_output_dim, fusion_dropout)
        
        # Bidirectional cross-attention
        self.image_to_audio_attention = CrossModalAttention(dim=encoder_output_dim)
        self.audio_to_image_attention = CrossModalAttention(dim=encoder_output_dim)
        
        # Gated fusion
        # self.gate = nn.Sequential(
        #     nn.Linear(1024, 512),  # Concatenated features -> gate
        #     nn.ReLU(),
        #     nn.Linear(512, 2),  # Output 2 gates (one for each modality)
        #     nn.Softmax(dim=-1),  # Normalize gates to sum to 1
        # )
        
        # For storing alignment loss
        # self.loss_align = None  # Will be computed during forward pass

    def forward(self, images, spectrograms):
        # Encode images and spectrograms
        image_features = self.image_encoder(images)  # [B, 512]
        audio_features = self.audio_encoder(spectrograms)  # [B, 512]
        
        # Bidirectional cross-attention
        image_to_audio = self.image_to_audio_attention(image_features, audio_features)  # [B, 512]
        audio_to_image = self.audio_to_image_attention(audio_features, image_features)  # [B, 512]
        
        # Concatenate the two attention outputs
        combined = torch.cat([image_to_audio, audio_to_image], dim=1)  # [B, 1024]
        # fused_features = torch.cat([image_to_audio, audio_to_image], dim=1)  # [B, 1024]
        
        # Compute gated fusion
        # gates = self.gate(fused_features)  # [B, 2]
        # gate_image, gate_audio = gates[:, 0:1], gates[:, 1:2]  # Split gates
        
        # Apply gates to the attention outputs
        # gated_fused_features = gate_image * image_to_audio + gate_audio * audio_to_image  # [B, 512]
        
        # Project to final dimension
        projected_features = self.fusion(combined)  # [B, output_dim]
        
        # Compute alignment loss using cosine similarity
        # self.loss_align = 1 - F.cosine_similarity(image_features, audio_features).mean()
        
        return projected_features

class CentralMultiModalEncoder(SimpleMultiModalEncoder):
    def __init__(self, output_dim=256, encoder_output_dim=512):
        super().__init__(output_dim, encoder_output_dim)

        # Image encoder with CentralNet Unimodal encoder (LeNet-like)
        self.image_encoder = nn.Sequential(
            CentralUnimodalImage(),
            nn.Linear(64 * 5 * 5, encoder_output_dim)
        )
        
        # Audio encoder with CentralNet Unimodal encoder
        self.audio_encoder = nn.Sequential(
            CentralUnimodalAudio(),
            nn.Linear(64 * 7 * 7, encoder_output_dim)
        )

# # Cross-attention ViT-based encoder TODO: rework gate mechanism to use nn.Parameter
    # class CrossAttentionViTMultiModalEncoder(BaseMultiModalEncoder):
    #     def __init__(self, 
    #         output_dim=256,
    #         image_size=28,
    #         image_patch_size=4,
    #         image_embed_dim=192,
    #         image_depth=4,
    #         image_num_heads=3,
    #         spectrogram_size=112,
    #         audio_patch_size=8,
    #         audio_embed_dim=192,
    #         audio_depth=4,
    #         audio_num_heads=3,
    #         dropout=0.1):

    #         super().__init__(output_dim)

    #         # Image encoder with ViT
    #         self.image_encoder = ViTEncoder(
    #             image_size=image_size,
    #             patch_size=image_patch_size,
    #             in_channels=1,
    #             embed_dim=image_embed_dim,
    #             depth=image_depth,
    #             num_heads=image_num_heads,
    #             dropout=dropout
    #         )
            
    #         # Audio encoder with ViT
    #         self.audio_encoder = AudioViTEncoder(
    #             spectrogram_size=spectrogram_size,
    #             patch_size=audio_patch_size,
    #             in_channels=1,
    #             embed_dim=audio_embed_dim,
    #             depth=audio_depth,
    #             num_heads=audio_num_heads,
    #             dropout=dropout
    #         )
            
    #         # Bidirectional cross-attention
    #         self.image_to_audio_attention = CrossModalAttention(dim=image_embed_dim)
    #         self.audio_to_image_attention = CrossModalAttention(dim=audio_embed_dim)
            
    #         # Gated fusion
    #         self.gate = nn.Sequential(
    #             nn.Linear(image_embed_dim + audio_embed_dim, 512),  # Concatenated features -> gate
    #             nn.ReLU(),
    #             nn.Linear(512, 2),  # Output 2 gates (one for each modality)
    #             nn.Softmax(dim=-1),  # Normalize gates to sum to 1
    #         )
            
    #         # Fusion and projection - max of image and audio embeddings
    #         fusion_input_dim = max(image_embed_dim, audio_embed_dim)
            
    #         # Fusion and projection
    #         self.fusion = nn.Sequential(
    #             nn.Linear(fusion_input_dim, 512),
    #             nn.ReLU(),
    #             nn.Dropout(0.3),  # Add dropout
    #             nn.Linear(512, output_dim),
    #         )
            
    #         # For storing alignment loss
    #         self.loss_align = None  # Will be computed during forward pass

    #     def forward(self, images, spectrograms):
    #         # Encode images and spectrograms
    #         image_features = self.image_encoder(images) 
    #         audio_features = self.audio_encoder(spectrograms)
            
    #         # Bidirectional cross-attention
    #         image_to_audio = self.image_to_audio_attention(image_features, audio_features)
    #         audio_to_image = self.audio_to_image_attention(audio_features, image_features)
            
    #         # Concatenate the two attention outputs
    #         fused_features = torch.cat([image_to_audio, audio_to_image], dim=1)
            
    #         # Compute gated fusion
    #         gates = self.gate(fused_features)
    #         gate_image, gate_audio = gates[:, 0:1], gates[:, 1:2]  # Split gates
            
    #         # Need to handle the case where image and audio dimensions might be different
    #         # If dimensions are different, we need to project one to match the other
    #         if image_to_audio.shape[-1] == audio_to_image.shape[-1]:
    #             gated_fused_features = gate_image * image_to_audio + gate_audio * audio_to_image
    #         else:
    #             # In this implementation, we'll use the larger of the two as the base
    #             if image_to_audio.shape[-1] > audio_to_image.shape[-1]:
    #                 # Project audio to match image dimensions
    #                 audio_proj = nn.Linear(audio_to_image.shape[-1], image_to_audio.shape[-1]).to(audio_to_image.device)
    #                 audio_to_image_proj = audio_proj(audio_to_image)
    #                 gated_fused_features = gate_image * image_to_audio + gate_audio * audio_to_image_proj
    #             else:
    #                 # Project image to match audio dimensions
    #                 image_proj = nn.Linear(image_to_audio.shape[-1], audio_to_image.shape[-1]).to(image_to_audio.device)
    #                 image_to_audio_proj = image_proj(image_to_audio)
    #                 gated_fused_features = gate_image * image_to_audio_proj + gate_audio * audio_to_image
            
    #         # Project to final dimension
    #         projected_features = self.fusion(gated_fused_features)
            
    #         # Compute alignment loss using cosine similarity
    #         # self.loss_align = 1 - F.cosine_similarity(image_features, audio_features).mean()
            
    #         return projected_features
    #     

# ----------------------------- UNIMODAL ENCODERS ----------------------------- 

# Base class for all unimodal encoders
class BaseUniModalEncoder(nn.Module):
    def __init__(self, output_dim=256, modality='image'):
        super().__init__()
        self.output_dim = output_dim
        self.modality = modality  # 'image' or 'audio'
        
    def forward(self, images=None, spectrograms=None):
        raise NotImplementedError("Subclasses must implement forward method")

# Image-only encoder
class ImageEncoder(BaseUniModalEncoder):
    def __init__(self, output_dim=256):
        super().__init__(output_dim, modality='image')
        
        self.encoder = image_encoder(output_dim=512)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim)
        )
        
    def forward(self, images=None, spectrograms=None):
        # Only use images, ignore spectrograms
        if images is None:
            raise ValueError("ImageEncoder requires image input")
        features = self.encoder(images)
        return self.projection(features)

# Audio-only encoder
class SpectrogramEncoder(BaseUniModalEncoder):
    def __init__(self, output_dim=256):
        super().__init__(output_dim, modality='audio')
        
        self.encoder = audio_encoder(output_dim=output_dim)
        
    def forward(self, images=None, spectrograms=None):
        # Only use spectrograms, ignore images
        if spectrograms is None:
            raise ValueError("SpectrogramEncoder requires spectrogram input")
        features = self.encoder(spectrograms)
        return features

class SpectrogramEncoderCentral(SpectrogramEncoder):
    def __init__(self, output_dim=256):
        super().__init__(output_dim=output_dim)  # Use parent init
        
        # Override encoder with CentralUnimodalAudio
        self.encoder = nn.Sequential(
            CentralUnimodalAudio(),
            nn.Linear(64 * 7 * 7, output_dim)
        )

class SpectrogramEncoderLSTM(SpectrogramEncoder):
    def __init__(self, output_dim=256):
        super().__init__(output_dim=output_dim)  # Use parent init
        
        # Override encoder with CentralUnimodalAudio
        self.encoder = LSTMAudioEncoder(output_dim=output_dim)

class SpectrogramEncoderResidual(SpectrogramEncoder):
    def __init__(self, output_dim=256):
        super().__init__(output_dim=output_dim)

        self.encoder = nn.Sequential(
            MiniResNet(),
            nn.Linear(512, output_dim)
        )

class SpectrogramEncoderViT(SpectrogramEncoder):
    def __init__(self, output_dim=256):
        super().__init__(output_dim=output_dim)

        self.encoder = nn.Sequential(
           AudioViTEncoder(
            spectrogram_size=112,
            patch_size=8,
            in_channels=1,
            embed_dim=512,
            depth=4,
            num_heads=4
            ), 
            nn.Linear(512, output_dim)
        )

class SpectrogramEncoderMobileViT(SpectrogramEncoder):
    def __init__(self, output_dim=128, pretrained=False):
        super().__init__(output_dim=output_dim)

        # MobileViT backbone (Modified MobileNetV3 for grayscale input)
        mobilenet = mobilenet_v3_small(pretrained=pretrained)
        # Modify first conv layer for single-channel input
        mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # Remove classifier head and replace with projection layer
        mobilenet.classifier = nn.Identity()  # Remove MobileNet's original classifier

        # Projection layer with dropout
        self.encoder = nn.Sequential(
            mobilenet,
            nn.Linear(576, 256),  # MobileNetV3 feature dim
            nn.ReLU(),
            nn.Linear(256, output_dim)  # Final embedding dim
        )

# class SpectrogramEncoderAST(SpectrogramEncoder):
    #     def __init__(self, output_dim=128, pretrained=False):
    #         super().__init__(output_dim=output_dim)

            
    #         # Load the pretrained AST model from Hugging Face
    #         model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"  # Fine-tuned on AudioSet

    #         # # MobileViT backbone (Modified MobileNetV3 for grayscale input)
    #         # mobilenet = mobilenet_v3_small(pretrained=pretrained)
    #         # # Modify first conv layer for single-channel input
    #         # mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    #         # # Remove classifier head and replace with projection layer
    #         # mobilenet.classifier = nn.Identity()  # Remove MobileNet's original classifier

    #         # Projection layer with dropout
    #         self.encoder = ASTModel.from_pretrained(model_name)
    #         # self.encoder.eval()
        
    #     def forward(self, images=None, spectrograms=None):
    #         # Only use spectrograms, ignore images
    #         if spectrograms is None:
    #             raise ValueError("SpectrogramEncoder requires spectrogram input")
    #          # Resize to AST's expected input size (128x128) using interpolation
    #         spectrograms = torch.nn.functional.interpolate(spectrograms, size=(128, 1024), mode="bilinear", align_corners=False)
    #         # Remove the channel dimension (AST expects (B, time, freq))
    #         spectrograms = spectrograms.squeeze(1)  # Now (B, 128, 128)
    #         features = self.encoder(spectrograms)
            
    #         return features

class SpectrogramEncoderResNet(SpectrogramEncoder):
    def __init__(self, output_dim=256):
        super().__init__(output_dim=output_dim)

        # ResNet backbone
        resnet = resnet18(pretrained=False)
        # Modify first conv layer for single-channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove classifier head and replace with projection layer
        resnet.fc = nn.Identity()  # Remove ResNet's original classifier

        self.encoder = nn.Sequential(
            resnet,
            nn.Linear(512, 256),  # ResNet feature dim
            nn.ReLU(),
            nn.Linear(256, output_dim)  # Final embedding dim
        )

#--------------------------------- DINO MODELS --------------------------------

# Multimodal DINO model that works with any multimodal encoder
class MultiModalDINO(nn.Module):
    def __init__(
        self,
        encoder_class=SimpleMultiModalEncoder,
        encoder_kwargs=None,
        output_dim=256, # output dim of the encoder (after fusion)
        encoder_output_dim=512, # output dim of the encoder (before fusion)
        projection_dim=128, # output dim of the projection (before giving to dino loss)
        momentum=0.996,
        center_momentum=0.9,
        dropout=0.3
    ):
        super().__init__()
        
        self.projection_dim = projection_dim
        self.output_dim = output_dim
        self.encoder_output_dim = encoder_output_dim
        self.dropout = dropout
        self.momentum = momentum
        self.center_momentum = center_momentum

        # Default empty dict if no kwargs provided
        encoder_kwargs = encoder_kwargs or {}
        encoder_kwargs['output_dim'] = output_dim
        encoder_kwargs['encoder_output_dim'] = encoder_output_dim
        
        # Create student and teacher encoders
        self.student = encoder_class(**encoder_kwargs)
        self.teacher = encoder_class(**encoder_kwargs)
        self.teacher.load_state_dict(self.student.state_dict())
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Student and teacher projections
        self.student_projection = ProjectionHead(output_dim, projection_dim, dropout_rate=dropout)
        self.teacher_projection = ProjectionHead(output_dim, projection_dim)
        
        self.teacher_projection.load_state_dict(self.student_projection.state_dict())
        for param in self.teacher_projection.parameters():
            param.requires_grad = False

        # Initialize center for teacher outputs
        self.register_buffer("center", torch.zeros(1, projection_dim))
        
    
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
                - global_images: tensor of shape [batch_size, n_global_views, channels, height, width]
                - global_audios: tensor of shape [batch_size, n_global_views, channels, height, width]
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
            current_local_images = local_images[:, view_idx, :, :, :]  # [batch_size, channels, height, width]
            current_local_audios = local_audios[:, view_idx, :, :, :]  # [batch_size, channels, height, width]
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
        
        # Handle alignment loss for cross-attention models
        alignment_loss = None
        if hasattr(self.student, 'loss_align') and self.student.loss_align is not None:
            alignment_loss = self.student.loss_align
            
        return student_outputs, teacher_outputs, alignment_loss

# Unified Lightning module for any DINO model
class MultiModalDINOLightning(pl.LightningModule):
    def __init__(
        self,
        data_dir="data/avmnist", 
        data_augmentation="burst_noise",
        dino_model=None,
        encoder_class=SimpleMultiModalEncoder,
        encoder_kwargs=None,
        projection_dim=256,
        output_dim=256,
        encoder_output_dim=512,
        momentum=0.996,
        center_momentum=0.9,
        student_temperature=0.1,
        teacher_temperature=0.04,
        learning_rate=0.0001,
        use_mixed_precision=True,
        num_epochs=100,
        weight_decay=1e-6,
        dropout=0.3,
    ):
        """
        PyTorch Lightning module for MultiModal DINO model pretraining
        
        Args:
            dino_model: Optional pre-created DINO model
            encoder_class: Encoder class to use if dino_model is None
            encoder_kwargs: Kwargs for encoder if dino_model is None
            projection_dim: Dimension of projection features
            output_dim: Dimension of output features
            momentum: Momentum value for teacher update
            center_momentum: Momentum for center update
            student_temperature: Temperature parameter for student network
            teacher_temperature: Temperature parameter for teacher network
            learning_rate: Learning rate for optimizer
            use_mixed_precision: Whether to use mixed precision training
        """
        super().__init__()

        self.encoder_class=encoder_class
        self.encoder_kwargs=encoder_kwargs
        self.output_dim=output_dim
        self.encoder_output_dim=encoder_output_dim
        self.learning_rate = learning_rate
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.use_mixed_precision = use_mixed_precision
        self.num_epochs = num_epochs
        self.output_dim = output_dim
        self.projection_dim = projection_dim
        self.momentum = momentum
        self.center_momentum = center_momentum
        self.weight_decay = weight_decay
        self.dropout = dropout

        # Create model if not provided
        if dino_model is None:
            self.build_model()
        else:
            self.model = dino_model
        
        # Save hyperparameters to be logged
        self.save_hyperparameters(ignore=['dino_model', "traindata", "validdata", "testdata"])
        
        # Initialize dataloaders for validation
        avmnist_data = AVMNISTDataModule(
            data_dir=data_dir, 
            num_workers=0, 
            batch_size=128,
            type=data_augmentation,
        )
        avmnist_data.setup()
        self.traindata, self.validdata = avmnist_data.train_dataloader(), avmnist_data.val_dataloader()

    def build_model(self):
        """Build the DINO model"""
        self.model = MultiModalDINO(
            encoder_class=self.encoder_class,
            encoder_kwargs=self.encoder_kwargs,
            output_dim=self.output_dim,
            encoder_output_dim=self.encoder_output_dim,
            projection_dim=self.projection_dim,
            momentum=self.momentum,
            center_momentum=self.center_momentum,
            dropout=self.dropout,
        )
        return self.model

    def forward(self, batch):
        """Forward pass (used for inference)"""
        return self.model(batch)
    
    def dino_loss(self, student_outputs, teacher_outputs, alignment_loss=None):
        """
        Compute DINO loss between student and teacher outputs.
        
        Args:
            student_outputs: Tensor of shape [n_views, batch_size, projection_dim]
            teacher_outputs: Tensor of shape [n_teacher_views, batch_size, projection_dim]
            alignment_loss: Optional alignment loss to add
        """
        student_outputs = F.normalize(student_outputs, p=2, dim=-1)
        teacher_outputs = F.normalize(teacher_outputs, p=2, dim=-1)

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
        
        dino_loss_value = total_loss / (n_student_views * n_teacher_views)
        
        # # Add alignment loss if present
        # if alignment_loss is not None:
        #     return dino_loss_value + self.alignment_weight * alignment_loss
        
        return dino_loss_value
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (global_images, global_audios, local_images, local_audios)
            batch_idx: Index of the batch
        """
        # Forward pass through model
        student_out, teacher_out, alignment_loss = self.model(batch)
        
        # Calculate loss
        loss = self.dino_loss(student_out, teacher_out, alignment_loss)
        
        # Update teacher network using momentum encoder
        self.model.update_teacher()
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Train a simple MLP head for 1 epoch and log its accuracy."""
        model_downstream = DownstreamClassifier(self.model, trainable_encoder=False).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_downstream.classifier.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        scaler = torch.cuda.amp.GradScaler(enabled=True)  # For mixed precision training

        # TRAIN ON ONE EPOCH
        model_downstream.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.traindata:
            images, audios, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            
            images = images.to(dtype=torch.float32)
            audios = audios.to(dtype=torch.float32)

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model_downstream(images, audios)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        # Log just once per epoch
        self.log('val_loss', total_loss/num_batches)

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
                images, audios, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                
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

        _, mlp_acc, _, _, _ = evaluate(model_downstream, self.validdata)

        self.log("mlp_acc", mlp_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            }
        }

class MultiModalDINOSemiSupervised(MultiModalDINO):
    def __init__(self, *args, num_classes=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        encoder_output_dim = self.student.encoder_output_dim
        self.image_classifier = ProjectionHead(input_dim=encoder_output_dim, projection_dim=self.num_classes)
        self.audio_classifier = ProjectionHead(input_dim=encoder_output_dim, projection_dim=self.num_classes)

    def forward(self, batch):
        image, audio, views = batch

        """Forward pass"""
        student_out, teacher_out, _ = super().forward(views)
        image_logits = self.image_classifier(self.student.image_encoder(image))
        audio_logits = self.audio_classifier(self.student.audio_encoder(audio))
        
        return image_logits, audio_logits, student_out, teacher_out
    
class MultiModalDINOSemiSupervisedLightning(MultiModalDINOLightning):
    def __init__(self, *args, alpha=1, **kwargs): # alpha decides how much of the supervised loss is used
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def build_model(self):
        self.model = MultiModalDINOSemiSupervised(
            encoder_class=self.encoder_class,
            encoder_kwargs=self.encoder_kwargs,
            output_dim=self.output_dim,
            encoder_output_dim=self.encoder_output_dim,
            projection_dim=self.projection_dim,
            momentum=self.momentum,
            center_momentum=self.center_momentum,
            dropout=self.dropout,
        )
        return self.model

    def supervised_loss(self, image_logits, audio_logits, labels):
        """
        Compute supervised loss between image and audio features
        
        Args:
            image_features: Tensor of shape [batch_size, output_dim]
            audio_features: Tensor of shape [batch_size, output_dim]
            labels: Tensor of shape [batch_size, n_classes]
        """

        # Early fusion as opposed to ce_loss(image) + ce_loss(audio) (late fusion of gradients)
        # Combine logits (optional): you could average them or only use one modality
        # Pros:
        # - Forces the two modalities to agree before making a prediction.
        # - Encourages joint reasoning — the model must combine cues before the final decision.
        # - May lead to more robust predictions if both modalities are strong.
        # Cons:
        # - If one modality is weak, it can pull down the overall performance.
        # - No gradient goes back individually to the heads from their own prediction — just from the fused one.
        # combined_logits = (image_logits + audio_logits) / 2.0
        loss = self.ce_loss(image_logits, labels) + self.ce_loss(audio_logits, labels)

        # Cross-entropy loss
        # return self.ce_loss(combined_logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (image, audio, views)
            batch_idx: Index of the batch
        """
        image, audio, labels, views = batch
        
        # Forward pass through model
        image_features, audio_features, student_out, teacher_out = self.model((image, audio, views))
        
        # Calculate loss
        dino_loss = self.dino_loss(student_out, teacher_out)
        supervised_loss = self.supervised_loss(image_features, audio_features, labels)
        loss = dino_loss + self.alpha * supervised_loss
        
        # Update teacher network using momentum encoder
        self.model.update_teacher()
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

class MultiModalDINOWithINFONCE(MultiModalDINO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoder_output_dim = self.student.encoder_output_dim
        self.image_projection_head = ProjectionHead(input_dim=encoder_output_dim, projection_dim=self.projection_dim)
        self.audio_projection_head = ProjectionHead(input_dim=encoder_output_dim, projection_dim=self.projection_dim)

    def forward(self, batch):
        image, audio, views = batch

        """Forward pass"""
        student_out, teacher_out, _ = super().forward(views)
        image_features = self.image_projection_head(self.student.image_encoder(image))
        audio_features = self.audio_projection_head(self.student.audio_encoder(audio))
        
        return image_features, audio_features, student_out, teacher_out
    
class MultiModalDINOWithINFONCELightning(MultiModalDINOLightning):
    def __init__(self, *args, output_dim=256, encoder_output_dim=128, 
                 encoder_class=SimpleMultiModalEncoder, alpha=1, **kwargs): 
        
        super().__init__(*args, output_dim=output_dim, encoder_output_dim=encoder_output_dim, 
                         encoder_class=encoder_class,**kwargs)
        self.alpha = alpha # alpha decides how much of the supervised loss is used

    def build_model(self):
        self.model = MultiModalDINOWithINFONCE(
            encoder_class=self.encoder_class,
            encoder_kwargs=self.encoder_kwargs,
            output_dim=self.output_dim,
            encoder_output_dim=self.encoder_output_dim,
            projection_dim=self.projection_dim,
            momentum=self.momentum,
            center_momentum=self.center_momentum,
            dropout=self.dropout,
        )
        return self.model

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
            batch: Tuple of (image, audio, views)
            batch_idx: Index of the batch
        """
        image, audio, labels, views = batch
        
        # Forward pass through model
        image_features, audio_features, student_out, teacher_out = self.model((image, audio, views))
        
        # Calculate loss
        dino_loss = self.dino_loss(student_out, teacher_out)
        infoNCE_loss = self.infoNCE_loss(image_features, audio_features)
        loss = dino_loss + self.alpha * infoNCE_loss
        
        # Update teacher network using momentum encoder
        self.model.update_teacher()
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

class MultiModalDINOWithMSE(MultiModalDINO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoder_output_dim = self.student.encoder_output_dim
        self.image_projection_head = ProjectionHead(input_dim=encoder_output_dim, projection_dim=self.projection_dim)
        self.audio_projection_head = ProjectionHead(input_dim=encoder_output_dim, projection_dim=self.projection_dim)

    def forward(self, batch):
        image, audio, views = batch

        """Forward pass"""
        student_out, teacher_out, _ = super().forward(views)
        image_features = self.image_projection_head(self.student.image_encoder(image))
        audio_features = self.audio_projection_head(self.student.audio_encoder(audio))
        
        return image_features, audio_features, student_out, teacher_out
    
class MultiModalDINOWithMSELightning(MultiModalDINOLightning):
    def __init__(self, *args, output_dim=256, encoder_output_dim=128, 
                 encoder_class=SimpleMultiModalEncoder, alpha=1, **kwargs):
        super().__init__(*args, output_dim=output_dim, encoder_output_dim=encoder_output_dim, 
                         encoder_class=encoder_class,**kwargs)
        self.alpha = alpha

    def build_model(self):
        self.model = MultiModalDINOWithMSE(
            encoder_class=self.encoder_class,
            encoder_kwargs=self.encoder_kwargs,
            output_dim=self.output_dim,
            encoder_output_dim=self.encoder_output_dim,
            projection_dim=self.projection_dim,
            momentum=self.momentum,
            center_momentum=self.center_momentum,
            dropout=self.dropout,
        )
        return self.model

    def mse_loss(self, image_outputs, audio_outputs):
        """
        Compute MSE loss for audio-visual alignment.
        
        Args:
            image_outputs: Image features from projection head [batch_size, projection_dim]
            audio_outputs: Audio features from projection head [batch_size, projection_dim]
        
        Returns:
            Loss value (scalar)
        """
        # Normalize feature vectors to ensure consistency in feature space
        image_outputs = F.normalize(image_outputs, p=2, dim=1)
        audio_outputs = F.normalize(audio_outputs, p=2, dim=1)

        # Compute Mean Squared Error between corresponding (paired) features
        loss = F.mse_loss(image_outputs, audio_outputs, reduction='mean')

        return loss

    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (image, audio, views)
            batch_idx: Index of the batch
        """
        image, audio, labels, views = batch
        
        # Forward pass through model
        image_features, audio_features, student_out, teacher_out = self.model((image, audio, views))
        
        # Calculate loss
        dino_loss = self.dino_loss(student_out, teacher_out)
        mse_loss = self.mse_loss(image_features, audio_features)
        loss = dino_loss + self.alpha * mse_loss
        
        # Update teacher network using momentum encoder
        self.model.update_teacher()
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

class ProjectionHead(nn.Module):
        def __init__(self, input_dim, projection_dim=256, dropout_rate=0, hidden_dim=512):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),        # Add BatchNorm
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, projection_dim)
            )
            
        def forward(self, x):
            x = self.mlp(x)
            # x = F.normalize(x, p=2, dim=-1)  # L2 normalization -> moved to dino_loss
            return x

# Unified DINO model for unimodal training
class UniModalDINO(nn.Module):
    def __init__(
        self,
        encoder_class=ImageEncoder,  # Default to image encoder
        encoder_kwargs=None,
        output_dim=256, # output dim of the encoder
        projection_dim=128, # output dim of the projection (before giving to dino loss)
        momentum=0.996,
        center_momentum=0.9,
        dropout=0.3
    ):
        super().__init__()
        
        # Default empty dict if no kwargs provided
        encoder_kwargs = encoder_kwargs or {}
        encoder_kwargs['output_dim'] = output_dim
        # encoder_kwargs['dropout_rate'] = dropout
        
        # Create student and teacher encoders
        self.student = encoder_class(**encoder_kwargs)
        self.teacher = encoder_class(**encoder_kwargs)
        self.teacher.load_state_dict(self.student.state_dict())
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Student and teacher projections
        self.student_projection =  ProjectionHead(output_dim, projection_dim,  dropout_rate=dropout)
        
        self.teacher_projection =  ProjectionHead(output_dim, projection_dim)
        
        self.teacher_projection.load_state_dict(self.student_projection.state_dict())
        for param in self.teacher_projection.parameters():
            param.requires_grad = False
            
        self.momentum = momentum
        self.center_momentum = center_momentum

        # Initialize center for teacher outputs
        self.register_buffer("center", torch.zeros(1, projection_dim))

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
            batch: tuple of (global_views, global_audios, local_views, local_audios)
            where each is a tensor of appropriate shape
        """
        
        global_images, global_audios, local_images, local_audios = batch
        
        # Decide which views to use based on encoder type
        if self.student.modality == 'image':
            global_views, local_views = global_images, local_images
        else:  # audio
            global_views, local_views = global_audios, local_audios

        device = next(self.student.parameters()).device
        
        # Move all tensors to device at once
        global_views = global_views.to(device)
        local_views = local_views.to(device)
        
        batch_size = global_views.shape[0]
        n_global_views = global_views.shape[1]
        n_local_views = local_views.shape[1]
        
        # Process student global views
        student_global_features = []
        for view_idx in range(n_global_views):
            current_global_view = global_views[:, view_idx, :, :, :]  # [batch_size, n_global_views, channels, height, width]
            # Pass both modality arguments but the encoder will use only what it needs
            if self.student.modality == 'image':
                features = self.student(images=current_global_view, spectrograms=None)
            else:  # audio
                features = self.student(images=None, spectrograms=current_global_view)
            student_global_features.append(features)
        
        # Process student local views
        student_local_features = []
        for view_idx in range(n_local_views):
            current_local_view = local_views[:, view_idx, :, :, :]  # [batch_size, channels, height, width]
            # Pass both modality arguments but the encoder will use only what it needs
            if self.student.modality == 'image':
                features = self.student(images=current_local_view, spectrograms=None)
            else:  # audio
                features = self.student(images=None, spectrograms=current_local_view)
            student_local_features.append(features)
        
        # Combine all student features
        student_features = torch.cat(student_global_features + student_local_features) # [(n_global_views + n_local_views) * batch_size, projection_dim]

        # Process teacher views (global only)
        with torch.no_grad():
            teacher_features = []
            for view_idx in range(n_global_views):
                current_global_view = global_views[:, view_idx, :, :, :]
                # Pass both modality arguments but the encoder will use only what it needs
                if self.teacher.modality == 'image':
                    features = self.teacher(images=current_global_view, spectrograms=None)
                else:  # audio
                    features = self.teacher(images=None, spectrograms=current_global_view)
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
        embeddings = student_features.view(n_local_views + n_global_views, batch_size, -1)

        return student_outputs, teacher_outputs, embeddings

class UniModalDINOV2(nn.Module):
    """
    Version without center and sharpen
    """

    def __init__(
        self,
        encoder_class=ImageEncoder,
        encoder_kwargs=None,
        output_dim=256,
        projection_dim=128,
        momentum=0.996,
        dropout=0.3
    ):
        super().__init__()
        
        encoder_kwargs = encoder_kwargs or {}
        encoder_kwargs['output_dim'] = output_dim
        
        self.student = encoder_class(**encoder_kwargs)
        self.teacher = encoder_class(**encoder_kwargs)
        self.teacher.load_state_dict(self.student.state_dict())
        
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student_projection = ProjectionHead(output_dim, projection_dim, dropout_rate=dropout)
        self.teacher_projection = ProjectionHead(output_dim, projection_dim)
        
        self.teacher_projection.load_state_dict(self.student_projection.state_dict())
        for param in self.teacher_projection.parameters():
            param.requires_grad = False
        
        self.momentum = momentum
    
    @torch.no_grad()
    def update_teacher(self):
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + (1 - self.momentum) * student_param.data
                                
        for student_param, teacher_param in zip(self.student_projection.parameters(), self.teacher_projection.parameters()):
            teacher_param.data = self.momentum * teacher_param.data + (1 - self.momentum) * student_param.data
    
    def forward(self, batch):
        global_images, global_audios, local_images, local_audios = batch
        
        if self.student.modality == 'image':
            global_views, local_views = global_images, local_images
        else:
            global_views, local_views = global_audios, local_audios
        
        device = next(self.student.parameters()).device
        global_views, local_views = global_views.to(device), local_views.to(device)
        
        batch_size = global_views.shape[0]
        n_global_views = global_views.shape[1]
        n_local_views = local_views.shape[1]
        
        student_global_features = [
            self.student(images=current, spectrograms=None) if self.student.modality == 'image' 
            else self.student(images=None, spectrograms=current)
            for current in global_views.permute(1, 0, 2, 3, 4)
        ]
        
        student_local_features = [
            self.student(images=current, spectrograms=None) if self.student.modality == 'image' 
            else self.student(images=None, spectrograms=current)
            for current in local_views.permute(1, 0, 2, 3, 4)
        ]
        
        student_features = torch.cat(student_global_features + student_local_features)
        
        with torch.no_grad():
            teacher_features = [
                self.teacher(images=current, spectrograms=None) if self.teacher.modality == 'image' 
                else self.teacher(images=None, spectrograms=current)
                for current in global_views.permute(1, 0, 2, 3, 4)
            ]
            teacher_features = torch.cat(teacher_features)
        
        student_projs = self.student_projection(student_features)
        
        with torch.no_grad():
            teacher_projs = self.teacher_projection(teacher_features)
        
        student_outputs = student_projs.view(n_local_views + n_global_views, batch_size, -1)
        teacher_outputs = teacher_projs.view(n_global_views, batch_size, -1)
        embeddings = student_features.view(n_local_views + n_global_views, batch_size, -1)

        return student_outputs, teacher_outputs, embeddings

# PyTorch Lightning module for UniModal DINO training
class UniModalDINOLightning(pl.LightningModule):
    def __init__(
        self, 
        data_dir="data/avmnist",
        dino_model=None,
        encoder_class=ImageEncoder,
        encoder_kwargs=None,
        projection_dim=128,
        output_dim=256,
        momentum=0.996,
        center_momentum=0.9,
        student_temperature=0.1,
        teacher_temperature=0.04,
        learning_rate=0.0001,
        use_mixed_precision=True,
        weight_decay=1e-6,
        cosine_loss_alpha=0.3,
        dropout=0.3,
        num_epochs=10,
        data_augmentation="burst_noise",
        use_original_model = True,
    ):
        """
        PyTorch Lightning module for UniModal DINO model pretraining
        
        Args:
            dino_model: Optional pre-created DINO model
            encoder_class: Encoder class to use if dino_model is None
            encoder_kwargs: Kwargs for encoder if dino_model is None
            projection_dim: Dimension of projection features
            output_dim: Dimension of encoder output features
            momentum: Momentum value for teacher update
            center_momentum: Momentum for center update
            student_temperature: Temperature parameter for student network
            teacher_temperature: Temperature parameter for teacher network
            learning_rate: Learning rate for optimizer
            use_mixed_precision: Whether to use mixed precision training
        """
        super().__init__()
        
        # Create model if not provided
        if dino_model is None:
            if use_original_model:
                self.model = UniModalDINO(
                    encoder_class=encoder_class,
                    encoder_kwargs=encoder_kwargs,
                    projection_dim=projection_dim,
                    output_dim=output_dim,
                    momentum=momentum,
                    center_momentum=center_momentum,
                    dropout=dropout
                )
            else:
                self.model = UniModalDINOV2(
                    encoder_class=encoder_class,
                    encoder_kwargs=encoder_kwargs,
                    projection_dim=projection_dim,
                    output_dim=output_dim,
                    momentum=momentum,
                    dropout=dropout
                )
        else:
            self.model = dino_model
        
        self.learning_rate = learning_rate
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.use_mixed_precision = use_mixed_precision
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.cosine_loss_alpha = cosine_loss_alpha
        self.num_epochs = num_epochs
        
        # Save hyperparameters to be logged
        self.save_hyperparameters(ignore=['dino_model', "traindata", "validdata", "testdata"])
        
        # Initialize dataloaders for validation
        self.traindata, self.validdata, _ = get_dataloader_augmented(data_dir, type=data_augmentation, batch_size=128, num_workers=0)

    def forward(self, batch):
        """Forward pass (used for inference)"""
        return self.model(batch)
    
    def _cosine_consistency_loss(self, embeddings):
        num_views, batch_size, _ = embeddings.shape
        
        loss = 0.0
        count = 0
        
        # Add L2 normalization here
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        for i in range(num_views):
            for j in range(i + 1, num_views):
                # Calculate cosine similarity
                sim = torch.sum(embeddings[i] * embeddings[j], dim=-1)
                
                # Scale loss to emphasize smaller similarities
                # This keeps values in [0,1] range but makes smaller similarities contribute more to loss
                loss += (1 - sim).pow(2).mean()  # Squared difference makes small diffs more important
                count += 1
        
        return loss / count
    
    def dino_loss(self, student_outputs, teacher_outputs):
        """
        Compute DINO loss (with L2 normalization at the loss level)
        between student and teacher outputs.
        
        Args:
            student_outputs: Tensor of shape [n_views, batch_size, projection_dim]
            teacher_outputs: Tensor of shape [n_teacher_views, batch_size, projection_dim]
        """
        
        student_outputs = F.normalize(student_outputs, p=2, dim=-1)
        teacher_outputs = F.normalize(teacher_outputs, p=2, dim=-1)

        n_student_views = student_outputs.shape[0]
        n_teacher_views = teacher_outputs.shape[0]
        
        # Sharpen teacher outputs and add centering regularization
        center = torch.mean(teacher_outputs, dim=1, keepdim=True)
        teacher_centered = teacher_outputs - center
        
        # Convert to probabilities
        teacher_probs = F.softmax(teacher_centered / self.teacher_temperature, dim=-1)
        
        # # Add entropy regularization to avoid collapse
        # entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-10), dim=-1)
        # entropy_reg = 0.1 * torch.mean(entropy)  # Target a medium level of entropy
        
        # Student probabilities
        student_probs = F.log_softmax(student_outputs / self.student_temperature, dim=-1)

        total_loss = 0
        for student_view_idx in range(n_student_views):
            for teacher_view_idx in range(n_teacher_views):
                view_loss = -(teacher_probs[teacher_view_idx] * student_probs[student_view_idx]).sum(dim=-1).mean()
                total_loss += view_loss
        
        # Combine losses
        dino_loss_value = total_loss / (n_student_views * n_teacher_views) # - entropy_reg
        
        return dino_loss_value
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (global_views, local_views)
            batch_idx: Index of the batch
        """
         # Forward pass through model
        student_out, teacher_out, student_embeddings = self.model(batch)

        # Compute DINO loss
        dino_loss = self.dino_loss(student_out, teacher_out)

        if self.cosine_loss_alpha > 0:
            # Compute Cosine Consistency Loss
            cosine_loss = self._cosine_consistency_loss(student_embeddings)
            # Combine losses
            alpha = self.cosine_loss_alpha  # Set weight for cosine loss (tuneable)
            total_loss = dino_loss + alpha * cosine_loss
        else:
            total_loss = dino_loss

        # Update teacher network using momentum encoder
        self.model.update_teacher()

        # Log loss
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.cosine_loss_alpha > 0:
            self.log('cosine_loss', cosine_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss
    
    # def validation_step(self, batch, batch_idx):
    #     """Dummy validation step to allow validation_epoch_end to run."""
    #     pass  # We don't need per-batch validation; full evaluation happens in validation_epoch_end

    def on_train_epoch_end(self):
        """Train a simple MLP head for 1 epoch and log its accuracy."""
        model_downstream = DownstreamClassifier(self.model, trainable_encoder=False).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model_downstream.classifier.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        scaler = torch.cuda.amp.GradScaler(enabled=True)  # For mixed precision training

        # TRAIN ON ONE EPOCH
        model_downstream.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.traindata:
            images, audios, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            
            images = images.to(dtype=torch.float32)
            audios = audios.to(dtype=torch.float32)

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model_downstream(images, audios)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        # Log just once per epoch
        self.log('val_loss', total_loss/num_batches)

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
                images, audios, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                
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

        _, mlp_acc, _, _, _ = evaluate(model_downstream, self.validdata)

        self.log("mlp_acc", mlp_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }

#------------------------------ DOWNSTREAM MODELS -----------------------------

# Unified downstream classifier
class DownstreamClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes=10, trainable_encoder=False, is_dino_based=True):
        super().__init__()

        # Create a deep copy of the encoder instead of referencing it
        self.encoder = copy.deepcopy(pretrained_model.student if is_dino_based else pretrained_model)
         # Move the encoder to the same device as the original model
        device = next(pretrained_model.parameters()).device
        self.encoder = self.encoder.to(device)

        # Check if this is a unimodal or multimodal encoder
        self.is_unimodal = hasattr(self.encoder, 'modality')
        self.modality = getattr(self.encoder, 'modality', None)
        
        # Freeze or unfreeze the encoder based on parameter
        for param in self.encoder.parameters():
            param.requires_grad = trainable_encoder

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, images, spectrograms=None):
        """
        Forward pass that works with both unimodal and multimodal models
        
        Args:
            images: Image tensor (required for multimodal and image-only models)
            spectrograms: Spectrogram tensor (optional for image-only models)
        """
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                if self.is_unimodal:
                    if self.modality == 'image':
                        features = self.encoder(images=images, spectrograms=None)
                    else:  # audio
                        features = self.encoder(images=None, spectrograms=spectrograms)
                else:
                    features = self.encoder(images, spectrograms)
        else:
            if self.is_unimodal:
                if self.modality == 'image':
                    features = self.encoder(images=images, spectrograms=None)
                else:  # audio
                    features = self.encoder(images=None, spectrograms=spectrograms)
            else:
                features = self.encoder(images, spectrograms)
                
        return self.classifier(features)

# Feature extractor with frozen encoder
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model, is_dino_based=True):
        super().__init__()

        self.encoder = copy.deepcopy(pretrained_model.student if is_dino_based else pretrained_model)
        # Move the encoder to the same device as the original model
        device = next(pretrained_model.parameters()).device
        self.encoder = self.encoder.to(device)
        
        # Check if this is a unimodal or multimodal encoder
        self.is_unimodal = hasattr(self.encoder, 'modality')
        self.modality = getattr(self.encoder, 'modality', None)
        
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, images, spectrograms=None):
        """
        Forward pass that works with both unimodal and multimodal models
        
        Args:
            images: Image tensor (required for multimodal and image-only models)
            spectrograms: Spectrogram tensor (optional for image-only models)
        """
        with torch.no_grad():
            if self.is_unimodal:
                if self.modality == 'image':
                    features = self.encoder(images=images, spectrograms=None)
                else:  # audio
                    features = self.encoder(images=None, spectrograms=spectrograms)
            else:
                features = self.encoder(images, spectrograms)
        return features
