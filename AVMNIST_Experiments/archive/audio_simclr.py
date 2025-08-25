
from utils.reproducibility import set_seed
set_seed(1)
import torch
import torch.nn as nn
import torch.optim as optim
from models.dino import SpectrogramEncoder
import torch.nn.functional as F
import lightning.pytorch as pl
from models.dino import ProjectionHead

class AudioSimCLRModel(nn.Module):
    def __init__(self, audio_encoder_cls=SpectrogramEncoder, output_dim=256, projection_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.projection_dim = projection_dim
        self.audio_encoder = audio_encoder_cls(output_dim=output_dim)
        self.audio_projection_head = ProjectionHead(output_dim, projection_dim)

    def forward(self, batch):
        # Batch contains: (aug_img1, aug_spec1, aug_img2, aug_spec2)
        _, spec1, _, spec2 = batch
        spec1 = spec1.float()
        spec2 = spec2.float()

        z1 = self.audio_projection_head(self.audio_encoder(None, spec1)) # Images = None, Spectrograms = spec1
        z2 = self.audio_projection_head(self.audio_encoder(None, spec2))

        return z1, z2

class AudioSimCLRLightning(pl.LightningModule):
    def __init__(
        self,
        audio_encoder_cls=SpectrogramEncoder,
        projection_dim=256,
        output_dim=256,
        learning_rate=0.0001,
        num_epochs=100,
        use_mixed_precision=True,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.projection_dim = projection_dim

        self.model = AudioSimCLRModel(audio_encoder_cls=audio_encoder_cls, 
                                      output_dim=output_dim, projection_dim=projection_dim)

        self.learning_rate = learning_rate
        self.use_mixed_precision = use_mixed_precision
        self.num_epochs = num_epochs

        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch)

    def nt_xent_loss(self, reps, temperature=0.07):
        reps = F.normalize(reps, dim=1)
        batch_size = reps.shape[0] // 2

        similarity_matrix = torch.matmul(reps, reps.T) / temperature

        # Mask self-similarities
        mask = torch.eye(2 * batch_size, device=reps.device).bool()
        similarity_matrix.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i + batch_size)
        labels = torch.arange(batch_size, device=reps.device)
        labels = torch.cat([labels + batch_size, labels])

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def training_step(self, batch, batch_idx):
        z1, z2 = self(batch)

        # Combine into joint representation space
        reps = torch.cat([z1, z2], dim=0)  # View 1 + View 2

        # Compute NT-Xent Loss (SimCLR-style)
        loss = self.nt_xent_loss(reps)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }