import mlx.core as mx
import numpy as np


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

        for start in range(0, len(self.dataset), self.batch_size):
            end = start + self.batch_size
            batch_indexes = self.indexes[start:end]

            # Optimized slicing for SimpleDataset-like objects if they support it
            if hasattr(self.dataset, "arrays") and isinstance(self.dataset.arrays, tuple):
                batch_data = [arr[batch_indexes] for arr in self.dataset.arrays]
                batch_data = [mx.array(x) if isinstance(x, np.ndarray) else x for x in batch_data]
                yield tuple(batch_data)
            else:
                batch = [self.dataset[i] for i in batch_indexes]
                transposed = zip(*batch)
                stacked = []
                for item_list in transposed:
                    if isinstance(item_list[0], np.ndarray):
                        stacked.append(mx.array(np.stack(item_list)))
                    elif isinstance(item_list[0], mx.array):
                        stacked.append(mx.stack(item_list))
                    else:
                        stacked.append(mx.array(list(item_list)))
                yield tuple(stacked)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size