import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
import numpy as np
from tqdm import tqdm
from .summary import ModelSummary

class Trainer:
    def __init__(
        self,
        max_epochs=10,
        accelerator="auto", # ignored in MLX (auto-gpu usually)
        devices=1, # ignored
        precision="fp32", # ignored/auto
        logger=None,
        callbacks=None,
        log_every_n_steps=10,
        compile=False,
        **kwargs
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
        # We need a closure for the loss function that captures the model structure but takes parameters
        def loss_fn(model, batch):
            model._metric_ctx.metrics = {} # Reset metrics capture
            loss = model.training_step(batch, 0) # batch_idx dummy
            return loss, model._metric_ctx.metrics

        loss_and_grad = mx.value_and_grad(loss_fn)

        def step(batch):
            (loss, metrics), grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            return loss, metrics

        if self.compile:
            step = mx.compile(step)

        # Resume logic (simplified)
        if ckpt_path:
            print(f"Resuming from {ckpt_path}")
            model.load_weights(ckpt_path)

        model.train()
        model._trainer = self

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # Training Loop
            model.train()
            t0 = time.time()
            
            # Progress bar for training
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", unit="step")
            
            for batch_idx, batch in pbar:
                loss, metrics = step(batch)
                
                # Evaluate loss and metrics
                mx.eval(state, loss, *metrics.values())

                self.global_step += 1
                
                # Update progress bar
                current_loss = loss.item()
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})
                
                if batch_idx % self.log_every_n_steps == 0:
                    # Log
                    log_dict = {k: v for k, v in self.logged_metrics.items()}
                    
                    for k, v in metrics.items():
                        log_dict[k] = v.item()

                    log_dict["epoch"] = epoch
                    log_dict["step"] = self.global_step
                    if "train_loss" not in log_dict:
                        log_dict["train_loss"] = current_loss
                    
                    for logger in self.loggers:
                        if logger:
                            logger.log_metrics(log_dict, step=self.global_step)
                    
                    self.logged_metrics = {} # Reset
            
            dt = time.time() - t0
            # print(f"Epoch {epoch} took {dt:.2f}s") # tqdm handles timing info mostly

            # Validation Loop
            if val_loader:
                model.eval()
                val_losses = []
                
                # Progress bar for validation
                val_pbar = tqdm(val_loader, desc="Validation", leave=False)
                
                for batch in val_pbar:
                    loss = model.validation_step(batch, 0)
                    if loss is not None:
                         if isinstance(loss, mx.array):
                            val_losses.append(loss.item())
                         else:
                            val_losses.append(loss)
                
                if val_losses:
                    mean_val_loss = np.mean(val_losses)
                    self.logged_metrics["val_loss"] = mean_val_loss
                    
                    # Update training pbar with val loss
                    pbar.set_postfix({"loss": f"{current_loss:.4f}", "val_loss": f"{mean_val_loss:.4f}"})
                    
                    # Log val metrics
                    for logger in self.loggers:
                        if logger:
                            logger.log_metrics(self.logged_metrics, step=self.global_step)

                # Callbacks (Checkpoint)
                for cb in self.callbacks:
                    cb.on_validation_end(self, model)
            
            # Reset metrics for next epoch start?
            self.logged_metrics = {}

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
