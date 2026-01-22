from pathlib import Path

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

import wandb


class TensorBoardLogger:
    def __init__(self, save_dir, name="lightning_logs", version=None):
        if SummaryWriter is None:
            print("tensorboardX not installed, TensorBoardLogger will be disabled.")
            self.experiment = None
            return

        save_dir = Path(save_dir)
        if version is None:
            version = 0  # simplified versioning logic

        self.log_dir = save_dir / name / f"version_{version}"
        self.experiment = SummaryWriter(log_dir=str(self.log_dir))

    def log_metrics(self, metrics, step=None):
        if self.experiment:
            for k, v in metrics.items():
                self.experiment.add_scalar(k, v, step)


class WandbLogger:
    def __init__(
        self, save_dir, name=None, project=None, version=None, id=None, log_model=None, **kwargs
    ):
        self.save_dir = save_dir

        # Handle version/id precedence
        version = version or id

        # Handle resume logic
        # If passed in kwargs, use it. Otherwise, default to "allow" if version is present.
        resume = kwargs.pop("resume", "allow" if version else None)

        self.experiment = wandb.init(
            dir=save_dir, project=project, name=name, id=version, resume=resume, **kwargs
        )

    def log_metrics(self, metrics, step=None):
        if self.experiment:
            self.experiment.log(metrics, step=step)
