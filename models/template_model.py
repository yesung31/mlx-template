import warnings

import mlx.nn as nn
import mlx.optimizers as optim

from core import LightningModule
from models.networks.template_network import TemplateNetwork


class TemplateModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # Initialize the dummy network
        self.net = TemplateNetwork()
        warnings.warn(
            "TemplateModel is using a dummy network. Replace this with your actual model logic.",
            UserWarning,
        )

    def __call__(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(learning_rate=0.001)
