import os
import wandb
from omegaconf import DictConfig, OmegaConf

def wandb_init(args, mode, key, save_dir=None):
    if mode != "offline" and key:
        wandb.login(key=key)

    run_name = os.environ.get("WANDB_NAME", args.exp_name)
    wandb.init(
        project=args.exp_id,
        name=run_name,
        config=args,
        save_code=False
    )