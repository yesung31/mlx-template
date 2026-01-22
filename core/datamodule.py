class LightningDataModule:
    def __init__(self):
        pass

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None
