"""Persist the environment contract a checkpoint was trained under, so eval can
reproduce it instead of guessing.

The policy is only comparable to itself when evaluated in the environment it was
trained in, but the settings that define that environment live in CLI flags
spread over three entrypoints (``train_sac``, ``train_bc_sac``/``bc_cli``,
``eval_sac``) whose defaults have drifted apart -- most damagingly
``--reactive-agents``, which BC-SAC training enables and eval did not, so the
reported collision / offroad / reaching rates came from a different world than
the one the policy saw.

Trainers call :func:`save_run_config`; ``eval_sac`` calls
:func:`apply_run_config_defaults`, which seeds the parser defaults from the saved
file. Explicit command-line flags still win, so an intentional cross-condition
eval ("how does this policy do against log-replay agents?") stays one flag away
-- it is only the *silent* mismatch that goes away.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUN_CONFIG_NAME = "run_config.json"

# The arguments that define the environment / reward / action / observation
# contract. Anything here changes what the policy sees or is scored on, so eval
# must inherit it. Training-process knobs (lr, batch size, wandb, ...) are
# deliberately excluded: they do not affect a rollout.
ENV_ARG_KEYS: tuple[str, ...] = (
    # Action space.
    "action_space",
    "delta_max_dx",
    "delta_max_dy",
    "delta_max_dyaw",
    # Episode / scene construction.
    "max_episode_steps",
    "max_num_objects",
    "reactive_agents",
    "idm_desired_vel",
    # Reward.
    "r_progress",
    "r_action_penalty",
    "r_collision",
    "r_offroad",
    "r_goal_bonus",
    "goal_threshold_m",
    "terminate_on_collision",
    "terminate_on_offroad",
    "route_reward",
    "r_lateral_penalty",
    "off_route_threshold_m",
    "r_off_route",
    "progression_indicator",
)

# Everything the encoder factory reads (``encoder``, ``encoder_dk``,
# ``encoder_depth``, ...) is captured by prefix, so adding a knob to
# ``add_encoder_args`` does not silently drop out of the saved config.
ENV_ARG_PREFIXES: tuple[str, ...] = ("encoder",)


def env_config_from_args(args) -> dict[str, Any]:
    """The subset of ``args`` that defines the environment contract."""
    keys = list(ENV_ARG_KEYS)
    keys += sorted(
        k for k in vars(args)
        if k.startswith(ENV_ARG_PREFIXES) and k not in ENV_ARG_KEYS
    )
    out: dict[str, Any] = {}
    for key in keys:
        if hasattr(args, key):
            value = getattr(args, key)
            # argparse gives us JSON-safe scalars already; guard anyway so a new
            # flag with an exotic type cannot make a training run fail at save.
            out[key] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
    return out


def save_run_config(save_dir: str | Path, args, extra: dict[str, Any] | None = None) -> Path:
    """Write ``<save_dir>/run_config.json``. Returns the path written."""
    path = Path(save_dir) / RUN_CONFIG_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(extra or {})
    payload["env"] = env_config_from_args(args)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def find_run_config(model_path: str | Path) -> Path | None:
    """Locate the run config next to a checkpoint, if one was saved."""
    model = Path(model_path).resolve()
    # The checkpoint may be <save_dir>/sac_waymax.zip or a checkpoints/ subdir
    # entry, so look beside it and one level up.
    for directory in (model.parent, model.parent.parent):
        candidate = directory / RUN_CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def apply_run_config_defaults(parser, config_path: str | Path) -> dict[str, Any]:
    """Seed ``parser`` defaults from a saved run config.

    Only keys the parser already defines are applied, so an older config missing
    newer flags (or carrying retired ones) degrades to the parser's own defaults
    instead of raising. Returns the settings actually applied.
    """
    with open(config_path, encoding="utf-8") as f:
        payload = json.load(f)
    env = payload.get("env")
    if not isinstance(env, dict):
        return {}

    known = {action.dest for action in parser._actions}
    applied = {k: v for k, v in env.items() if k in known}
    parser.set_defaults(**applied)
    return applied
