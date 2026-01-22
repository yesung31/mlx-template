import os
from pathlib import Path

import hydra
import wandb
import mlx.core as mx
from omegaconf import DictConfig, OmegaConf

import data
import models
from core import Trainer, ModelCheckpoint, TensorBoardLogger, WandbLogger
from utils.helpers import get_resume_info, instantiate, register_resolvers

register_resolvers()


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")

    mx.random.seed(cfg.seed)

    # Dynamic loading of Model and DataModule
    ModelClass = getattr(models, cfg.model.name)
    DataClass = getattr(data, cfg.data.name)

    print(f"Model: {ModelClass.__name__}")
    print(f"DataModule: {DataClass.__name__}")

    # Instantiate
    model = instantiate(ModelClass, cfg.model)
    dm = instantiate(DataClass, cfg.data)

    # Handle Resume
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_path, wandb_id = get_resume_info(log_dir) if cfg.resume else (None, None)

    # Logger
    tb_logger = TensorBoardLogger(save_dir=log_dir, name="", version="")
    wandb_logger = WandbLogger(
        save_dir=log_dir,
        project=Path.cwd().name,
        name=f"{cfg.data.name}/{cfg.model.name}",
        mode=cfg.wandb,
        id=wandb_id,
        resume="allow",
    )

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=[tb_logger, wandb_logger],
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        compile=cfg.compile,
    )

    trainer.fit(model, dm, ckpt_path=ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()
