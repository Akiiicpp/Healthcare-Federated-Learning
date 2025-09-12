from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SyntheticMedicalDataset(Dataset):
    def __init__(self, num_samples: int = 1000, image_size: int = 224, in_channels: int = 1, pos_ratio: float = 0.3, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(num_samples, in_channels, image_size, image_size, generator=generator)
        # Simple signal for positive class: add a brighter center blob
        labels = torch.zeros(num_samples, dtype=torch.long)
        num_pos = int(num_samples * pos_ratio)
        labels[:num_pos] = 1
        perm = torch.randperm(num_samples, generator=generator)
        self.labels = labels[perm]
        if in_channels == 1:
            self.images[self.labels == 1, :, image_size//4:3*image_size//4, image_size//4:3*image_size//4] += 0.8
        else:
            self.images[self.labels == 1, :, image_size//4:3*image_size//4, image_size//4:3*image_size//4] += 0.8

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def get_dataloaders(batch_size: int = 32, val_split: float = 0.2, num_samples: int = 1000, image_size: int = 224, in_channels: int = 1, pos_ratio: float = 0.3, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    dataset = SyntheticMedicalDataset(num_samples=num_samples, image_size=image_size, in_channels=in_channels, pos_ratio=pos_ratio, seed=seed)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
