"""Implements dataloaders for the AVMNIST dataset.

Here, the data is assumed to be in a folder titled "avmnist".
Code written by: Ward Van den Bossche (inspired by multibench get_data_loader code)
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchaudio.transforms as audio_transforms
import random
import lightning.pytorch as pl # IMPORTANT DONT MIX WITH pytorch_lightning, choose one of the two
import os
import torch.nn.functional as F
from PIL import Image

#----------------------------------------- Augmentations and Transformations -----------------------------------------#

class GaussianNoise(torch.nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        return x + torch.randn_like(x) * self.std

class TimeWarpWithStretch(torch.nn.Module):
    """
    Time-stretch module with size preservation
    note that this is a simplified version of the time warping mentioned in the SpecAugment paper 
    (since in that case interpolation is used to ensure a more smooth transition? TODO fact check this)
    """
    def __init__(self, min_factor=0.8, max_factor=1.2, target_length=112):
        super(TimeWarpWithStretch, self).__init__()
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.target_length = target_length
        self.time_stretch = audio_transforms.TimeStretch(n_freq=target_length)  # Keep size fixed

    def forward(self, spec):
        # Convert real-valued spectrogram to complex by adding a zero imaginary part
        spec_complex = torch.complex(spec, torch.zeros_like(spec))

        # Random time-stretching within a given range
        stretch_factor = random.uniform(self.min_factor, self.max_factor)
        stretched_spec = self.time_stretch(spec_complex, stretch_factor)

        # Size preservation: pad or trim to maintain target length
        current_length = stretched_spec.shape[-1]
        if current_length > self.target_length:
            stretched_spec = stretched_spec[..., :self.target_length]
        elif current_length < self.target_length:
            padding = self.target_length - current_length
            stretched_spec = F.pad(stretched_spec, (0, padding), "constant", 0)
        
        return torch.abs(stretched_spec)

class GroupedMasking(torch.nn.Module):
    """
    Randomly masks groups of patches in a 3D spectrogram.
    
    Args:
        spectrogram: Input spectrogram of shape (1, height, width).
        mask_ratio: Fraction of the spectrogram to mask.
        group_size: Size of each group (e.g., 4x4 patches).
    
    Returns:
        masked_spectrogram: Spectrogram with masked regions set to 0.
    """
    def __init__(self, mask_ratio=0.5, group_size=4):
        super(GroupedMasking, self).__init__()
        self.mask_ratio = mask_ratio
        self.group_size = group_size

    def forward(self, spectrogram):
        if spectrogram.ndim != 3 or spectrogram.shape[0] != 1:
            raise ValueError("Input spectrogram must have shape (1, height, width)")
        
        _, height, width = spectrogram.shape
        
        if height % self.group_size != 0 or width % self.group_size != 0:
            raise ValueError("Height and width must be divisible by group_size")

        # Compute number of groups
        num_groups_h = height // self.group_size
        num_groups_w = width // self.group_size
        num_groups = num_groups_h * num_groups_w
        num_masked_groups = int(self.mask_ratio * num_groups)

        # Reshape spectrogram into groups
        spectrogram = spectrogram.view(
            1, num_groups_h, self.group_size, num_groups_w, self.group_size
        ).permute(0, 1, 3, 2, 4).contiguous()

        # Create a binary mask for groups
        mask = torch.ones(num_groups, device=spectrogram.device)
        masked_indices = torch.randperm(num_groups)[:num_masked_groups]
        mask[masked_indices] = 0
        mask = mask.view(num_groups_h, num_groups_w, 1, 1).repeat(1, 1, self.group_size, self.group_size)

        # Apply mask and reshape back
        masked_spectrogram = spectrogram * mask
        masked_spectrogram = masked_spectrogram.permute(0, 1, 3, 2, 4).contiguous()
        masked_spectrogram = masked_spectrogram.view(1, height, width)
        
        return masked_spectrogram

class MultiModalAugmentation:
    def __init__(self, n_global_views=2, n_local_views=4, 
                 global_spec_size=112, local_spec_size=112, augment_values=None):
        
        self.n_local_views = n_local_views
        self.n_global_views = n_global_views
        self.global_spec_size = global_spec_size
        self.local_spec_size = local_spec_size

        self._initialize_transforms(augment_values)

    def _initialize_transforms(self, augment_values=None):
        global_image_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=28, scale=(0.75, 1.0), antialias=True),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        local_image_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=28, scale=(0.3, 0.75), antialias=True),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
        if augment_values is None:
            self.global_transforms = {
                'image': global_image_transforms,
                'audio': transforms.Compose([
                    # Keep original size but apply moderate augmentations
                    transforms.RandomApply(
                        [transforms.RandomResizedCrop(
                            size=(self.global_spec_size, self.global_spec_size), 
                            scale=(0.8, 1.0), 
                            antialias=True)
                        ], p=0.5),
                    transforms.RandomApply([
                        TimeWarpWithStretch(min_factor=0.9, max_factor=1.1, 
                                            target_length=self.global_spec_size)], p=0.3),  # Add time-stretch (global)
                    # Moderate frequency/time masking
                    transforms.RandomApply([audio_transforms.FrequencyMasking(freq_mask_param=15)], p=0.3),
                    transforms.RandomApply([audio_transforms.TimeMasking(time_mask_param=15)], p=0.3),
                    # Add gentle pitch shifting (vertical shift in spectrogram)
                    transforms.RandomApply(
                        [transforms.RandomAffine(degrees=0, 
                                                translate=(0, 0.1), 
                                                scale=(0.9, 1.1))
                        ], p=0.5),
                    # Grouped Masking
                    transforms.RandomApply([GroupedMasking(mask_ratio=0.15)], p=0.5),
                ]),
            }

            self.local_transforms = {
                'image': local_image_transforms,
                'audio': transforms.Compose([
                    # Apply crop to create local view but maintain overall dimensions
                    transforms.RandomApply(
                        [transforms.RandomResizedCrop(
                            size=(self.local_spec_size, self.local_spec_size), 
                            scale=(0.5, 0.9), 
                            antialias=True
                            )
                        ], p=0.7
                    ),
                    transforms.RandomApply(
                        [TimeWarpWithStretch(
                            min_factor=0.7, max_factor=1.3, 
                            target_length=self.local_spec_size)
                        ], p=0.7),  # Add time-stretch (local)
                        
                    # Much stronger frequency masking (vertical strips)
                    transforms.RandomApply([audio_transforms.FrequencyMasking(freq_mask_param=25)], p=0.7),
                    # Much stronger time masking (horizontal strips)
                    transforms.RandomApply([audio_transforms.TimeMasking(time_mask_param=25)], p=0.7),
                    # Add more aggressive affine transformations (simulates pitch shift in spectrogram)
                    transforms.RandomApply(
                        [transforms.RandomAffine(
                            degrees=0, translate=(0, 0.2), 
                            scale=(0.7, 1.3))], p=0.7),
                    # Add noise
                    transforms.RandomApply([GaussianNoise(std=0.1)], p=0.7),
                    # Grouped masking
                    transforms.RandomApply([GroupedMasking(mask_ratio=0.6)], p=0.9),
                ]),
            }
        else:
            aug_to_class = {
                "time_warp": TimeWarpWithStretch,
                "frequency_mask": audio_transforms.FrequencyMasking,
                "time_mask": audio_transforms.TimeMasking,
                "grouped_masking": GroupedMasking,
                "gaussian_noise": GaussianNoise,
                "random_affine": transforms.RandomAffine,
                "random_resized_crop": transforms.RandomResizedCrop,
            }

            augment_list = {"global_views": [], "local_views": []}
            for aug_type in ["global_views", "local_views"]:
                augmentations = augment_values['augmentations'][aug_type]
                augmentation_probabilities = augment_values['augmentation_probabilities'][aug_type]
                for aug in augmentations.keys():
                    # Process arguments to convert lists back into tuples if needed
                    aug_args = {}
                    for k, v in augmentations[aug].items():
                        if isinstance(v, list) or isinstance(v, tuple):
                            aug_args[k] = tuple(v)
                        else:
                            aug_args[k] = v

                    augment_list[aug_type].append(
                        transforms.RandomApply([aug_to_class[aug](**aug_args)], p=augmentation_probabilities[aug])
                    )
                    # augment_list[aug_type].append(transforms.RandomApply([aug_to_class[aug](**augmentations[aug])], p=augmentation_probabilities[aug]))

            self.global_transforms = {
                'image': global_image_transforms,
                'audio': transforms.Compose(augment_list['global_views']),
            }

            self.local_transforms = {
                'image': local_image_transforms,
                'audio': transforms.Compose(augment_list['local_views']),
            }

    @torch.no_grad()
    def __call__(self, images, audios):
        global_images = []
        global_audios = []

        for _ in range(self.n_global_views):
            # Process global views
            global_images.append(self.global_transforms['image'](images))
            global_audios.append(self.global_transforms['audio'](audios))
        
        global_images = torch.stack(global_images, dim=0)
        global_audios = torch.stack(global_audios, dim=0)

        # Generate multiple diverse local views
        local_images = []
        local_audios = []
        
        for _ in range(self.n_local_views):
            local_images.append(self.local_transforms['image'](images))
            local_audios.append(self.local_transforms['audio'](audios))
            
        local_images = torch.stack(local_images, dim=0)
        local_audios = torch.stack(local_audios, dim=0)
        
        return global_images, global_audios, local_images, local_audios
    
    def __str__(self):
        def transform_list(transform_obj):
            if isinstance(transform_obj, transforms.Compose):
                return [transform_to_string(t) for t in transform_obj.transforms]
            return [transform_to_string(transform_obj)]

        def transform_to_string(transform):
            cls_name = type(transform).__name__
            params = {
                k: v for k, v in vars(transform).items()
                if not k.startswith('_') and not callable(v)
            }
            param_str = ', '.join(f"{k}={v}" for k, v in params.items())
            return f"{cls_name}({param_str})"

        lines = [
            f"MultiModalAugmentation(",
            f"  n_global_views={self.n_global_views},",
            f"  n_local_views={self.n_local_views},",
            f"  global_spec_size={self.global_spec_size},",
            f"  local_spec_size={self.local_spec_size},",
            f"  global_transforms:",
            f"    image: ["
        ] + [f"      {t}" for t in transform_list(self.global_transforms['image'])] + [
            f"    ],",
            f"    audio: ["
        ] + [f"      {t}" for t in transform_list(self.global_transforms['audio'])] + [
            f"    ]",
            f"  local_transforms:",
            f"    image: ["
        ] + [f"      {t}" for t in transform_list(self.local_transforms['image'])] + [
            f"    ],",
            f"    audio: ["
        ] + [f"      {t}" for t in transform_list(self.local_transforms['audio'])] + [
            f"    ]",
            f")"
        ]

        return "\n".join(lines)

# class MultiModalAugmentation:
    #     def __init__(self, n_global_views=2, n_local_views=4):  # Increase local views
    #         self.n_local_views = n_local_views
    #         self.n_global_views = n_global_views
            
    #         # Global transforms - more conservative
    #         self.global_transforms = {
    #             'image': transforms.Compose([
    #                 transforms.RandomResizedCrop(size=28, scale=(0.75, 1.0), antialias=True),
    #                 # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    #                 transforms.RandomRotation(degrees=5),  # Add slight rotation
    #                 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
    #             ]),
    #             'audio': transforms.Compose([
    #                 transforms.RandomResizedCrop(size=112, scale=(0.75, 1.0), antialias=True),
    #                 transforms.RandomApply([audio_transforms.FrequencyMasking(freq_mask_param=10)], p=0.2),
    #                 transforms.RandomApply([audio_transforms.TimeMasking(time_mask_param=10)], p=0.2),
    #                 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
    #             ])
    #         }
            
    #         # Local transforms - more aggressive
    #         self.local_transforms = {
    #             'image': transforms.Compose([
    #                 transforms.RandomResizedCrop(size=28, scale=(0.3, 0.75), antialias=True),
    #                 # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.4),
    #                 transforms.RandomRotation(degrees=15),  # More rotation (not too much or it might confuse a 6 for a 9)
    #                 transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    #                 transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    #             ]),
    #             'audio': transforms.Compose([
    #                 transforms.RandomResizedCrop(size=112, scale=(0.3, 0.75), antialias=True),
    #                 transforms.RandomApply([audio_transforms.FrequencyMasking(freq_mask_param=30)], p=0.4),
    #                 transforms.RandomApply([audio_transforms.TimeMasking(time_mask_param=30)], p=0.4),
    #                 transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    #                 transforms.RandomApply([GaussianNoise(std=0.1)], p=0.3),
    #                 transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    #             ])
    #         }

    #     @torch.no_grad()
    #     def __call__(self, images, audios):

    #         global_images = []
    #         global_audios = []

    #         for _ in range(self.n_global_views):
    #             # Process global views
    #             global_images.append(self.global_transforms['image'](images))
    #             global_audios.append(self.global_transforms['audio'](audios))
            
    #         global_images = torch.stack(global_images, dim=0)
    #         global_audios = torch.stack(global_audios, dim=0)

    #         # Generate multiple diverse local views
    #         local_images = []
    #         local_audios = []
            
    #         for _ in range(self.n_local_views):
    #             local_images.append(self.local_transforms['image'](images))
    #             local_audios.append(self.local_transforms['audio'](audios))
                
    #         local_images = torch.stack(local_images, dim=0)
    #         local_audios = torch.stack(local_audios, dim=0)
            
    #         return global_images, global_audios, local_images, local_audios

class SimCLRMultiModalAugmentation:
    def __init__(self, image_size=28, spec_size=112, augment_values=None):
        """
        Args:
            image_size: Size of input images (assumed square).
            spec_size: Size of input spectrograms (assumed square).
            augment_values: Custom augmentation parameters (optional).
        """
        self.image_size = image_size
        self.spec_size = spec_size
        self._initialize_transforms(augment_values)

    def _initialize_transforms(self, augment_values=None):
        # Image augmentations
        image_transforms = transforms.Compose([
            # Moderate crop (avoid cutting digits completely)
            transforms.RandomResizedCrop(
                size=self.image_size, 
                scale=(0.5, 1.0),  # Less aggressive than SimCLR's 0.2
                ratio=(0.8, 1.2),  # Preserve aspect ratio
                antialias=True
            ),
            # Tiny rotation (±5° to avoid 6/9 confusion)
            transforms.RandomRotation(degrees=5),
            # Mild affine transforms (small shifts)
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Max 10% shift
            ),
            # Elastic deformations (simulate handwriting variance)
            transforms.RandomApply([
                transforms.ElasticTransform(
                    alpha=20.0,  # Magnitude
                    sigma=3.0    # Smoothness
                )
            ], p=0.3),
            # Mild blur (simulate focus variations)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),
        ])

        # Audio (spectrogram) augmentations
        spectrogram_transforms = transforms.Compose([
            # Crop with reasonable bounds (avoid losing all content)
            transforms.RandomResizedCrop(
                size=(self.spec_size, self.spec_size),
                scale=(0.5, 1.0),  # Less aggressive than SimCLR's 0.2
                antialias=True
            ),
            # Mild time warping (preserve temporal structure)
            transforms.RandomApply([
                TimeWarpWithStretch(
                    min_factor=0.9,  # Max 10% stretch/squeeze
                    max_factor=1.1,
                    target_length=self.spec_size
                )
            ], p=0.5),
            # Frequency masking (vertical bands)
            transforms.RandomApply([
                audio_transforms.FrequencyMasking(freq_mask_param=10)  # ~10% of freq bins
            ], p=0.5),
            # Time masking (horizontal bands)
            transforms.RandomApply([
                audio_transforms.TimeMasking(time_mask_param=10)  # ~10% of time steps
            ], p=0.5),
            # Additive noise (gentle)
            transforms.RandomApply([
                GaussianNoise(std=0.05)  # Reduced from 0.1
            ], p=0.3),
        ])

        if augment_values is not None:
            # Custom augmentations
            aug_to_class = {
                "time_warp": TimeWarpWithStretch,
                "frequency_mask": audio_transforms.FrequencyMasking,
                "time_mask": audio_transforms.TimeMasking,
                "grouped_masking": GroupedMasking,
                "gaussian_noise": GaussianNoise,
            }
            audio_augs = []
            for aug in augment_values['augmentations'].keys():
                if aug in aug_to_class:
                    audio_augs.append(
                        transforms.RandomApply([aug_to_class[aug](**augment_values['augmentations'][aug])], 
                        p=augment_values['augmentation_probabilities'][aug]
                    ))
                #TODO: Add image augmentations if needed

        self.image_transform = image_transforms
        self.spectrogram_transform = spectrogram_transforms

    @torch.no_grad()
    def __call__(self, images, audios):
        """
        Args:
            images: Input images (B, C, H, W)
            audios: Input spectrograms (B, C, H, W)
        Returns:
            Tuple: (aug_images1, aug_audios1, aug_images2, aug_audios2)
        """
        # Generate two augmented views per modality
        aug_images1 = self.image_transform(images)
        aug_images2 = self.image_transform(images)
        
        aug_audios1 = self.spectrogram_transform(audios)
        aug_audios2 = self.spectrogram_transform(audios)

        return aug_images1, aug_audios1, aug_images2, aug_audios2

#----------------------------------------------------- Datasets -----------------------------------------------------#

class BaseAVMNISTDataset(Dataset):
    def __init__(
        self, image_path, audio_path, labels_path, flatten_audio=False, flatten_image=False,
        unsqueeze_channel=True, normalize_image=True, normalize_audio=True, compute_stats=False
    ):
        self.image_path = image_path
        self.audio_path = audio_path
        self.labels_path = labels_path
        self.flatten_audio = flatten_audio
        self.flatten_image = flatten_image
        self.normalize_image = normalize_image
        self.normalize_audio = normalize_audio
        self.unsqueeze_channel = unsqueeze_channel

        # Load labels entirely since they are usually smaller
        self.labels = np.load(labels_path).astype(int)

        self.image_data = np.load(self.image_path, mmap_mode='r')
        # self.image_data = MemmapWrapper(self.image_path, shape=(len(self.labels), 28, 28), dtype="float64")
        # NOTE: Audio data was made using np.memmap with mode='w+', this means 
        # np.load with mmap_mode='r' will not work, and we need to use np.memmap
        self.audio_data = MemmapWrapper(self.audio_path, shape=(len(self.labels), 112, 112), dtype="uint8")

        # Get the dataset size from the labels
        self.dataset_size = len(self.labels)

        # Compute dataset-wide mean and std for spectrograms if needed
        if compute_stats and self.normalize_audio:
            self.audio_mean, self.audio_std = self.compute_audio_stats()
        else:
            self.audio_mean, self.audio_std = 0.0, 1.0  # Default values to prevent division by zero

    def compute_audio_stats(self):
        all_means = []
        all_stds = []
        for i in range(len(self)):
            audio = np.array(self.audio_data[i]) / 255.0
            all_means.append(audio.mean())
            all_stds.append(audio.std())
        return np.mean(all_means), np.mean(all_stds)

    def __len__(self):
        return self.dataset_size

    def _process_image_audio(self, idx):
        image = np.array(self.image_data[idx])
        audio = np.array(self.audio_data[idx])

        if self.flatten_audio:
            audio = audio.reshape(-1)
        if not self.flatten_image:
            image = image.reshape(28, 28)
        if self.normalize_image:
            image = image / 255.0
        if self.normalize_audio:
            audio = (audio / 255.0 - self.audio_mean) / self.audio_std
        if self.unsqueeze_channel:
            image = np.expand_dims(image, 0)
            audio = np.expand_dims(audio, 0)

        return image, audio

class AVMNISTDataset(BaseAVMNISTDataset):
    def __getitem__(self, idx):
        image, audio = self._process_image_audio(idx)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, audio, label

class AVMNISTSSLDataset(BaseAVMNISTDataset):
    def __init__(self, *args, transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        image, audio = self._process_image_audio(idx)
        image = torch.tensor(image, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        views = self.transform(image, audio)
        return views

class AVMNISTSSLDatasetExtended(AVMNISTSSLDataset):
    """
    A variation that returns original image/audio/label along with augmented views.
    Inherits all preprocessing from the base class and overrides __getitem__ for extended return values.
    """

    def __getitem__(self, idx):
        # Use base preprocessing logic
        image, audio = self._process_image_audio(idx)
        image = torch.tensor(image, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)

        # Apply transformations (returns global and local views)
        views = self.transform(image, audio)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, audio, label, views
    
#---------------------------------------------------- DataModules ---------------------------------------------------#

class BaseAVMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        num_workers: int = 6,
        type: str = "burst_noise",
        train_shuffle: bool = True,
        flatten_audio: bool = False,
        flatten_image: bool = False,
        unsqueeze_channel: bool = True,
        normalize_image: bool = True,
        normalize_audio: bool = True,
        train_size: int = 55000,
        val_size: int = 5000,
        test_size: int = 10000
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.type = type
        self.train_shuffle = train_shuffle
        self.flatten_audio = flatten_audio
        self.flatten_image = flatten_image
        self.unsqueeze_channel = unsqueeze_channel
        self.normalize_image = normalize_image
        self.normalize_audio = normalize_audio
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        
        # Define paths
        self.train_image_path = f"{data_dir}image/train_data.npy"
        self.train_audio_path = f"{data_dir}audio/train_data_augmented_{type}.npy"
        self.train_labels_path = f"{data_dir}train_labels.npy"
        self.test_image_path = f"{data_dir}image/test_data.npy"
        self.test_audio_path = f"{data_dir}audio/test_data_augmented_{type}.npy"
        self.test_labels_path = f"{data_dir}test_labels.npy"

    def prepare_data(self):
        # Check if files exist
        for path in [self.train_image_path, self.train_audio_path, self.train_labels_path,
                    self.test_image_path, self.test_audio_path, self.test_labels_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")

    def _get_dataset_kwargs(self):
        return {
            'flatten_audio': self.flatten_audio,
            'flatten_image': self.flatten_image,
            'unsqueeze_channel': self.unsqueeze_channel,
            'normalize_image': self.normalize_image,
            'normalize_audio': self.normalize_audio
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

class AVMNISTDataModule(BaseAVMNISTDataModule):
    def setup(self, stage: str = None):
        dataset_kwargs = self._get_dataset_kwargs()
        
        if stage == 'fit' or stage is None:
            full_train_dataset = AVMNISTDataset(
                self.train_image_path,
                self.train_audio_path,
                self.train_labels_path,
                **dataset_kwargs
            )
            
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, 
                [self.train_size, self.val_size]
            )

        if stage == 'test' or stage is None:
            test_dataset = AVMNISTDataset(
                self.test_image_path,
                self.test_audio_path,
                self.test_labels_path,
                **dataset_kwargs
            )
            # Split test dataset
            self.test_dataset, _ = random_split(test_dataset, 
                                              [self.test_size, 10000 - self.test_size])

class AVMNISTDinoDataModule(BaseAVMNISTDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, n_global_views=2, 
                 n_local_views=4, type: str = "burst_noise", augmentations=None):
        super().__init__(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, type=type)
        
        # Store view configuration as class attributes
        self.n_global_views = n_global_views
        self.n_local_views = n_local_views
    
        self.augmentations = MultiModalAugmentation(
            n_global_views=n_global_views, 
            n_local_views=n_local_views
        ) if (augmentations is None) else augmentations

    def setup(self, stage: str = None):
        dataset_kwargs = self._get_dataset_kwargs()
        
        if stage == 'fit' or stage is None:
            full_train_dataset = AVMNISTSSLDataset(
                self.train_image_path,
                self.train_audio_path,
                self.train_labels_path,
                transform=self.augmentations,
                **dataset_kwargs
            )
            
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, 
                [self.train_size, self.val_size]
            )

        if stage == 'test' or stage is None:
            # Note: Using regular AVMNISTDataset for test, not the Dino version
            test_dataset = AVMNISTDataset(
                self.test_image_path,
                self.test_audio_path,
                self.test_labels_path,
                **dataset_kwargs
            )
            # Split test dataset
            self.test_dataset, _ = random_split(test_dataset, 
                                             [self.test_size, 10000 - self.test_size])
    
    def get_view_config(self):
        """Return the current view configuration."""
        return {
            "n_global_views": self.n_global_views,
            "n_local_views": self.n_local_views
        }

class AVMNISTDinoDataModuleExtended(AVMNISTDinoDataModule):
    """
    A variation that returns original image/audio/label along with augmented views 
    """
    def __init__(self, data_dir, batch_size=32, num_workers=4, n_global_views=2, n_local_views=4, type: str = "burst_noise", augmentations=None):
        super().__init__(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, n_global_views=n_global_views, n_local_views=n_local_views, type=type, augmentations=augmentations)

    def setup(self, stage: str = None):
        dataset_kwargs = self._get_dataset_kwargs()
        
        if stage == 'fit' or stage is None:
            full_train_dataset = AVMNISTSSLDatasetExtended(
                self.train_image_path,
                self.train_audio_path,
                self.train_labels_path,
                transform=self.augmentations,
                **dataset_kwargs
            )
            
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, 
                [self.train_size, self.val_size]
            )

        if stage == 'test' or stage is None:
            # Note: Using regular AVMNISTDataset for test, not the Dino version
            test_dataset = AVMNISTDataset(
                self.test_image_path,
                self.test_audio_path,
                self.test_labels_path,
                **dataset_kwargs
            )
            # Split test dataset
            self.test_dataset, _ = random_split(test_dataset, 
                                             [self.test_size, 10000 - self.test_size])

class AVMNISTSimCLRDataModule(BaseAVMNISTDataModule):
    def __init__(self, data_dir, batch_size=128, num_workers=6, type: str = "burst_noise", augmentations=None):
        super().__init__(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, type=type)
    
        self.augmentations = SimCLRMultiModalAugmentation() if (augmentations is None) else augmentations

    def setup(self, stage: str = None):
        dataset_kwargs = self._get_dataset_kwargs()
        
        if stage == 'fit' or stage is None:
            full_train_dataset = AVMNISTSSLDataset(
                self.train_image_path,
                self.train_audio_path,
                self.train_labels_path,
                transform=self.augmentations,
                **dataset_kwargs
            )
            
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, 
                [self.train_size, self.val_size]
            )

        if stage == 'test' or stage is None:
            # Note: Using regular AVMNISTDataset for test
            test_dataset = AVMNISTDataset(
                self.test_image_path,
                self.test_audio_path,
                self.test_labels_path,
                **dataset_kwargs
            )
            # Split test dataset
            self.test_dataset, _ = random_split(test_dataset, 
                                             [self.test_size, 10000 - self.test_size])
            
#-------------------------------------------- Helper functions / Classes  -------------------------------------------#

class MemmapWrapper:
    # This wrapper class is needed for multiprocessing to work with np.memmap objects
    # (i.e. if we want to use num_workers > 0 in DataLoader)
    def __init__(self, path, shape, dtype):
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.memmap = np.memmap(path, mode='r', dtype=dtype, shape=shape)

    def __getitem__(self, idx):
        return self.memmap[idx]

    def __getstate__(self):
        # Ensure only necessary information is pickled
        return {'path': self.path, 'shape': self.shape, 'dtype': self.dtype}

    def __setstate__(self, state):
        # Recreate the memmap object upon unpickling
        self.__dict__.update(state)
        self.memmap = np.memmap(self.path, mode='r', dtype=self.dtype, shape=self.shape)

def get_dataloader_augmented(data_dir, batch_size=128, num_workers=6, type="burst_noise", train_shuffle=True, 
                    flatten_audio=False, flatten_image=False, unsqueeze_channel=True, 
                    generate_sample=False, normalize_image=True, normalize_audio=True):
    
    data_dir = data_dir.rstrip("/")  # Removes the trailing '/' only if it exists

    train_image_path = f"{data_dir}/image/train_data.npy"
    train_audio_path = f"{data_dir}/audio/train_data_augmented_{type}.npy"
    train_labels_path = f"{data_dir}/train_labels.npy"
    
    test_image_path = f"{data_dir}/image/test_data.npy"
    test_audio_path = f"{data_dir}/audio/test_data_augmented_{type}.npy"
    test_labels_path = f"{data_dir}/test_labels.npy"
    
    train_dataset = AVMNISTDataset(train_image_path, train_audio_path, train_labels_path,
                                   flatten_audio=flatten_audio, flatten_image=flatten_image, 
                                   unsqueeze_channel=unsqueeze_channel,
                                   normalize_image=normalize_image, normalize_audio=normalize_audio)
    test_dataset = AVMNISTDataset(test_image_path, test_audio_path, test_labels_path, 
                                    flatten_audio=flatten_audio, flatten_image=flatten_image, 
                                    unsqueeze_channel=unsqueeze_channel,
                                    normalize_image=normalize_image, normalize_audio=normalize_audio)
    
    train_size, val_size = 55000, 5000 # 55000, 5000 # 2430, 270 # for FSDD # 55000, 5000 for full data TODO: change to full data
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # test_dataset, _ = torch.utils.data.random_split(test_dataset, [1000, 5000])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def get_dataloader_dino(data_dir, batch_size=128, num_workers=6, type="burst_noise", train_shuffle=True, 
                    flatten_audio=False, flatten_image=False, unsqueeze_channel=True, 
                    generate_sample=False, normalize_image=True, normalize_audio=True):

    train_image_path = f"{data_dir}/image/train_data.npy"
    train_audio_path = f"{data_dir}/audio/train_data_augmented_{type}.npy"
    train_labels_path = f"{data_dir}/train_labels.npy"
    
    test_image_path = f"{data_dir}/image/test_data.npy"
    test_audio_path = f"{data_dir}/audio/test_data_augmented_{type}.npy"
    test_labels_path = f"{data_dir}/test_labels.npy"
    
    transform = MultiModalAugmentation()
    
    train_dataset = AVMNISTSSLDataset(train_image_path, train_audio_path, train_labels_path, transform=transform,
                                   flatten_audio=flatten_audio, flatten_image=flatten_image, 
                                   unsqueeze_channel=unsqueeze_channel,
                                   normalize_image=normalize_image, normalize_audio=normalize_audio)
    
    # NOTE: No augmentations for test set
    test_dataset = AVMNISTDataset(test_image_path, test_audio_path, test_labels_path, 
                                    flatten_audio=flatten_audio, flatten_image=flatten_image, 
                                    unsqueeze_channel=unsqueeze_channel,
                                    normalize_image=normalize_image, normalize_audio=normalize_audio)
    
    train_size, val_size = 55000, 5000 # 2430, 270 # for FSDD # 55000, 5000 for full data
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def load_results_from_csv(csv_file):
    """Load true labels and predicted probabilities from CSV"""
    df = pd.read_csv(csv_file)
    
    # Extract true labels and probabilities
    true_labels = df["true_label"].values
    predicted_probs = np.array([eval(x) for x in df["probabilities"]])  # Convert string to list

    return true_labels, predicted_probs