import gzip
import os
from pathlib import Path

import numpy as np

from core import DataLoader, LightningDataModule
from core.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32) / 255.0
        # MNIST images are (B, 28, 28), we need (B, 28, 28, 1) for Conv2d in MLX
        self.images = self.images[..., np.newaxis]
        self.labels = labels.astype(np.int64)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    @property
    def arrays(self):
        return (self.images, self.labels)


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir="~/Datasets/mnist", batch_size=64, **kwargs):
        super().__init__()
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.batch_size = batch_size

    def _load_mnist_file(self, filename):
        filepath = self.data_dir / "raw" / filename
        with gzip.open(filepath, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16 if "images" in filename else 8)
        if "images" in filename:
            return data.reshape(-1, 28, 28)
        else:
            return data

    def setup(self, stage=None):
        train_images = self._load_mnist_file("train-images-idx3-ubyte.gz")
        train_labels = self._load_mnist_file("train-labels-idx1-ubyte.gz")
        test_images = self._load_mnist_file("t10k-images-idx3-ubyte.gz")
        test_labels = self._load_mnist_file("t10k-labels-idx1-ubyte.gz")

        # Split train into train and val (e.g., 55000 / 5000)
        indices = np.random.permutation(len(train_images))
        train_idx, val_idx = indices[:55000], indices[55000:]

        self.train_dataset = MNISTDataset(train_images[train_idx], train_labels[train_idx])
        self.val_dataset = MNISTDataset(train_images[val_idx], train_labels[val_idx])
        self.test_dataset = MNISTDataset(test_images, test_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
