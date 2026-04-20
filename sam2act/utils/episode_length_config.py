# episode_length_config.py: Loader for per-task episode_length used in the time-feature encoding.
# episode_length_config.py: Single source of truth — `configs/episode_length.yaml`.

import os
import yaml


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "episode_length.yaml",
)


def load_episode_length_config(config_path=None):
    """Load episode_length.yaml. Returns dict with `default` and `tasks` keys."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Episode length config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if "default" not in cfg:
        raise ValueError(f"{path} must define a `default` key")
    if not isinstance(cfg["default"], int) or cfg["default"] < 1:
        raise ValueError(f"`default` must be an int >= 1, got {cfg['default']}")
    tasks = cfg.get("tasks") or {}
    for task, value in tasks.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"tasks.{task} must be an int >= 1, got {value}")
    return {"default": cfg["default"], "tasks": tasks}


def get_episode_length(task, cfg=None, override=None):
    """Resolve episode_length for a task. Precedence: override > per-task > default."""
    if override is not None:
        if not isinstance(override, int) or override < 1:
            raise ValueError(f"override must be an int >= 1, got {override}")
        return override
    cfg = cfg or load_episode_length_config()
    return cfg["tasks"].get(task, cfg["default"])


def build_per_task_dict(tasks, cfg=None, override=None):
    """Build {task: episode_length} dict for a list of tasks."""
    cfg = cfg or load_episode_length_config()
    return {task: get_episode_length(task, cfg=cfg, override=override) for task in tasks}


def get_max_episode_length(tasks, cfg=None, override=None):
    """Return max episode_length across a list of tasks (safe upper bound for rollout budget)."""
    if override is not None:
        if not isinstance(override, int) or override < 1:
            raise ValueError(f"override must be an int >= 1, got {override}")
        return override
    cfg = cfg or load_episode_length_config()
    return max(get_episode_length(task, cfg=cfg) for task in tasks)
