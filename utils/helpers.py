from pathlib import Path

from omegaconf import OmegaConf


def register_resolvers():
    if not OmegaConf.has_resolver("resume_or_new"):
        OmegaConf.register_new_resolver("resume_or_new", lambda r, n: r if r else n)


def load_config_from_ckpt(ckpt_path):
    ckpt_path = Path(ckpt_path).resolve()
    for parent in ckpt_path.parents:
        cfg_path = parent / ".hydra" / "config.yaml"
        if cfg_path.exists():
            return OmegaConf.load(cfg_path)
    raise FileNotFoundError(f"Config not found for {ckpt_path}")


def get_resume_info(log_dir):
    log_dir = Path(log_dir).resolve()

    ckpt = log_dir / "checkpoints" / "last.npz"
    ckpt_path = str(ckpt) if ckpt.exists() else None
    if ckpt_path:
        print(f"Resuming: {ckpt_path}")

    wandb_id = None
    w_dir = log_dir / "wandb"
    if w_dir.exists():
        latest = w_dir / "latest-run"
        run = latest.resolve() if latest.exists() else None
        if not run:
            runs = sorted(
                [d for d in w_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
            )
            run = runs[-1] if runs else None
        if run:
            wandb_id = run.name.split("-")[-1]
            print(f"WandB ID: {wandb_id}")

    return ckpt_path, wandb_id


def instantiate(Class, cfg):
    kwargs = OmegaConf.to_container(cfg, resolve=True)

    # We pass the relevant sub-configs as kwargs, excluding 'name' which is used for class resolution.
    kwargs.pop("name", None)

    return Class(**kwargs)
