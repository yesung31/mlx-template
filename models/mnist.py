import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from core import LightningModule
from models.networks.cnn import CNN


class MNISTModel(LightningModule):
    def __init__(self, learning_rate=0.001, **kwargs):
        super().__init__()
        self.net = CNN()
        self.lr = learning_rate

    def __call__(self, x):
        return self.net(x)

    def _accuracy(self, logits, y):
        return mx.mean(mx.argmax(logits, axis=-1) == y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        acc = self._accuracy(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        acc = self._accuracy(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        acc = self._accuracy(logits, y)

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(learning_rate=self.lr)
