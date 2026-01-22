import mlx.core as mx
import numpy as np
from tqdm import tqdm

from .summary import ModelSummary


class Trainer:
    def __init__(
        self,
        max_epochs=10,
        accelerator="auto",  # ignored in MLX (auto-gpu usually)
        devices=1,  # ignored
        precision="fp32",  # ignored/auto
        logger=None,
        callbacks=None,
        log_every_n_steps=10,
        compile=False,
        **kwargs,
    ):
        self.max_epochs = max_epochs
        self.loggers = logger if isinstance(logger, list) else [logger]
        self.callbacks = callbacks or []
        self.log_every_n_steps = log_every_n_steps
        self.compile = compile
        self.current_epoch = 0
        self.global_step = 0
        self.logged_metrics = {}

    def log_metric(self, name, value):
        # Accumulate or print?
        # For simplicity, store and log at step
        if isinstance(value, mx.array):
            value = value.item()
        self.logged_metrics[name] = value

    def fit(self, model, datamodule, ckpt_path=None):
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        # Summary
        print("\n")
        ModelSummary(model).summarize()
        print("\n")

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Optimizer
        optimizer = model.configure_optimizers()

        # MLX State
        state = [model.state, optimizer.state]

        # Compile step function
        def loss_fn(model, batch):
            model._metric_ctx.metrics = {}
            model._metric_ctx.metadata = {}
            loss = model.training_step(batch, 0)
            return loss, (model._metric_ctx.metrics, model._metric_ctx.metadata)

        loss_and_grad = mx.value_and_grad(loss_fn)

        def step(batch):
            (loss, (metrics, metadata)), grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            return loss, metrics, metadata

        if self.compile:
            step = mx.compile(step)

        # Resume logic
        if ckpt_path:
            print(f"Resuming from {ckpt_path}")
            model.load_weights(ckpt_path)

        model.train()
        model._trainer = self

        # Initialize progress bar
        pbar = tqdm(total=len(train_loader), unit="step", leave=True)

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            pbar.reset()
            pbar.set_description(f"Epoch {epoch}")

            # Training Loop
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                loss, metrics, metadata = step(batch)

                # Evaluate loss and metrics
                mx.eval(state, loss, *metrics.values())

                self.global_step += 1

                # Update progress bar with only prog_bar=True metrics
                pbar_metrics = {}
                for k, v in metrics.items():
                    if metadata.get(k, {}).get("prog_bar", False):
                        pbar_metrics[k] = f"{v.item():.4f}"
                pbar.set_postfix(pbar_metrics)
                pbar.update(1)

                if batch_idx % self.log_every_n_steps == 0:
                    # Log
                    log_dict = {}
                    for k, v in metrics.items():
                        if metadata.get(k, {}).get("logger", True):
                            log_dict[k] = v.item()

                    log_dict["epoch"] = epoch
                    log_dict["step"] = self.global_step

                    for logger in self.loggers:
                        if logger:
                            logger.log_metrics(log_dict, step=self.global_step)

            # Validation Loop
            if val_loader:
                model.eval()
                val_metrics_accum = {}
                val_metadata = {}

                for batch in val_loader:
                    model._metric_ctx.metrics = {}
                    model._metric_ctx.metadata = {}
                    loss = model.validation_step(batch, 0)
                    
                    # Accumulate metrics
                    current_metrics = model._metric_ctx.metrics
                    val_metadata.update(model._metric_ctx.metadata)
                    
                    for k, v in current_metrics.items():
                        if k not in val_metrics_accum:
                            val_metrics_accum[k] = []
                        val_metrics_accum[k].append(v.item() if hasattr(v, "item") else v)

                if val_metrics_accum:
                    final_val_metrics = {k: np.mean(v) for k, v in val_metrics_accum.items()}
                    self.logged_metrics.update(final_val_metrics)

                    # Update training pbar with val metrics that have prog_bar=True
                    pbar_metrics = {}
                    # Include training metrics from last step if they were prog_bar
                    for k, v in metrics.items():
                        if metadata.get(k, {}).get("prog_bar", False):
                            pbar_metrics[k] = f"{v.item():.4f}"
                    
                    for k, v in final_val_metrics.items():
                        if val_metadata.get(k, {}).get("prog_bar", False):
                            pbar_metrics[f"val_{k}"] = f"{v:.4f}"
                    pbar.set_postfix(pbar_metrics)

                    # Log val metrics
                    log_dict = {f"val_{k}": v for k, v in final_val_metrics.items()}
                    log_dict["epoch"] = epoch
                    log_dict["step"] = self.global_step
                    for logger in self.loggers:
                        if logger:
                            logger.log_metrics(log_dict, step=self.global_step)

                # Callbacks (Checkpoint)
                for cb in self.callbacks:
                    cb.on_validation_end(self, model)

            # Reset metrics for next epoch start
            self.logged_metrics = {}
        
        pbar.close()

    def test(self, model, datamodule):
        datamodule.prepare_data()
        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()

        if not test_loader:
            print("No test dataloader found.")
            return

        model.eval()
        test_losses = []
        print("Starting testing...")

        for batch in tqdm(test_loader, desc="Testing"):
            loss = model.test_step(batch, 0)
            if loss is not None:
                if isinstance(loss, mx.array):
                    test_losses.append(loss.item())
                else:
                    test_losses.append(loss)

        if test_losses:
            mean_test_loss = np.mean(test_losses)
            print(f"Test Loss: {mean_test_loss:.4f}")
            return mean_test_loss
