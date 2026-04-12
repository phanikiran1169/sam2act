import os
import wandb
from omegaconf import DictConfig, OmegaConf

def wandb_init(args, mode, key, save_dir=None):
    if mode != "offline" and key:
        wandb.login(key=key)

    run_name = os.environ.get("WANDB_NAME", args.exp_name)
    project = os.environ.get("WANDB_PROJECT", args.exp_id)
    entity = os.environ.get("WANDB_ENTITY", None)
    wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        config=args,
        save_code=False
    )