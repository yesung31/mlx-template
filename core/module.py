import mlx.nn as nn


class MetricContext:
    def __init__(self):
        self.metrics = {}
        self.metadata = {}


class LightningModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._trainer = None
        self._metric_ctx = MetricContext()

    def training_step(self, batch, batch_idx):
        """
        Returns a scalar loss or a dictionary.
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        """
        Returns an optimizer or (optimizer, scheduler).
        """
        raise NotImplementedError

    def log(self, name, value, prog_bar=False, logger=True):
        self._metric_ctx.metrics[name] = value
        self._metric_ctx.metadata[name] = {"prog_bar": prog_bar, "logger": logger}
