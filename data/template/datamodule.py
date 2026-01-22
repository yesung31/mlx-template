import warnings

import numpy as np

from core import DataLoader, LightningDataModule
from core.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays
        self.length = len(arrays[0])
        for arr in arrays:
            assert len(arr) == self.length, "All arrays must have same length"

    def __getitem__(self, index):
        return tuple(arr[index] for arr in self.arrays)

    def __len__(self):
        return self.length


class TemplateDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        warnings.warn(
            "TemplateDataModule is using dummy random data. "
            "Replace this with your actual data loading logic.",
            UserWarning,
        )

    def prepare_data(self):
        # Download data, etc.
        pass

    def setup(self, stage=None):
        # Create dummy data: 100 samples, 32 dimensions
        self.train_dataset = SimpleDataset(
            np.random.randn(100, 32).astype(np.float32), np.random.randint(0, 2, (100,))
        )
        self.val_dataset = SimpleDataset(
            np.random.randn(20, 32).astype(np.float32), np.random.randint(0, 2, (20,))
        )
        self.test_dataset = SimpleDataset(
            np.random.randn(20, 32).astype(np.float32), np.random.randint(0, 2, (20,))
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)
