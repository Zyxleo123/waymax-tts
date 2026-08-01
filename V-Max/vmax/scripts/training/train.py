# Copyright 2025 Valeo.


"""Script to run the training process."""

import os
import pathlib
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from waymax import dynamics

from vmax import PATH_TO_APP, simulator
from vmax.agents import learning
from vmax.scripts.training import train_utils


OmegaConf.register_new_resolver("output_dir", train_utils.resolve_output_dir)


def _build_banker(config: dict, env_config: dict, run_config: dict, run_path: str):
    """Build the clean-rollout banker, or None when `bank_num_scenarios` is unset.

    Banking harvests a clean trajectory per scenario the first time one appears,
    instead of waiting for every scenario to be clean at the same checkpoint. It
    needs its own env and data pass: the training env auto-resets (so a slot's
    identity changes the moment a scenario terminates early) and the failures eval
    generator shuffles, so neither can be reused here.
    """
    if not config.get("bank_num_scenarios"):
        return None

    from vmax.agents.learning.reinforcement.sac.sac_factory import make_inference_fn, make_networks

    # `rl` lives in the parent repo, not in V-Max; running this script directly puts
    # only its own directory on sys.path, so point at the repo root explicitly.
    repo_root = str(pathlib.Path(__file__).resolve().parents[4])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from rl.bank import Banker, TrajectoryBank, make_bank_scenarios

    path = config.get("bank_dataset") or env_config.get("path_dataset_eval_failures") or env_config["path_dataset"]
    num_scenarios = int(config["bank_num_scenarios"])
    bank_dir = config.get("bank_dir") or os.path.join(run_path, "bank")

    bank_env = simulator.make_env_for_evaluation(
        max_num_objects=env_config["max_num_objects"],
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=env_config["sdc_paths_from_data"],
        observation_type=env_config["observation_type"],
        observation_config=env_config["observation_config"],
        reward_type=env_config["reward_type"],
        reward_config=env_config["reward_config"],
        termination_keys=env_config["termination_keys"],
    )
    chunks = list(
        make_bank_scenarios(
            path=path,
            max_num_objects=env_config["max_num_objects"],
            include_sdc_paths=env_config["sdc_paths_from_data"],
            num_records=num_scenarios,
            chunk_size=config.get("bank_chunk_size"),
            num_paths=env_config["num_sdc_paths"],
            num_points_per_path=env_config["num_points_per_sdc_path"],
        )
    )
    # Same network the trainer builds; `build_config_dicts` has already moved the
    # network spec out of config["algorithm"] and into run_config["network_config"].
    network = make_networks(
        observation_size=bank_env.observation_spec(),
        action_size=bank_env.action_spec().data.shape[0],
        unflatten_fn=bank_env.get_wrapper_attr("features_extractor").unflatten_features,
        learning_rate=run_config["learning_rate"],
        network_config=run_config["network_config"],
    )
    bank = TrajectoryBank(bank_dir, num_scenarios=num_scenarios)
    print(f"[bank] {num_scenarios} scenarios from {path} -> {bank_dir} "
          f"({len(chunks)} chunk(s), {bank.num_banked} already banked)")

    from rl.bank import MAX_YAW_RATE_RAD_S

    # Smoothness gate. The default (V-Max's own 0.95 rad/s comfort limit) refuses ~92% of
    # what the current policy produces, so it is only affordable once `yaw_rate_penalty`
    # has done its work -- set `bank_max_yaw_rate=inf` to harvest without it meanwhile.
    _gate = config.get("bank_max_yaw_rate")
    max_yaw_rate = MAX_YAW_RATE_RAD_S if _gate is None else float(_gate)

    return Banker(
        env=bank_env,
        policy_fn=make_inference_fn(network),
        bank=bank,
        scenario_chunks=chunks,
        scenario_length=config["scenario_length"],
        seed=config["seed"],
        max_yaw_rate_rad_s=max_yaw_rate,
    )


@hydra.main(version_base=None, config_name="base_config", config_path=PATH_TO_APP + "/config")
def run(cfg: DictConfig) -> None:
    """Run the training process with the provided configuration.

    Args:
        cfg: Configuration for the training process.

    """
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    train_utils.apply_xla_flags(config)
    train_utils.print_hyperparameters(config)
    train_utils.get_and_print_device_info()

    env_config, run_config = train_utils.build_config_dicts(config)

    # (num_devices, num_envs, num_episode_per_epoch)
    data_generator = simulator.make_data_generator(
        path=env_config["path_dataset"],
        max_num_objects=env_config["max_num_objects"],
        include_sdc_paths=env_config["sdc_paths_from_data"],
        batch_dims=(env_config["num_envs"], env_config["num_episode_per_epoch"]),
        seed=env_config["seed"],
        distributed=True,
        num_paths=env_config["num_sdc_paths"],
        num_points_per_path=env_config["num_points_per_sdc_path"],
    )

    if config["eval_freq"] > 0:
        eval_data_generator = simulator.make_data_generator(
            path=env_config["path_dataset_eval"],
            max_num_objects=env_config["max_num_objects"],
            include_sdc_paths=env_config["sdc_paths_from_data"],
            batch_dims=(8, config["num_scenario_per_eval"] // 8),
            seed=69,
            distributed=True,
            num_paths=env_config["num_sdc_paths"],
            num_points_per_path=env_config["num_points_per_sdc_path"],
        )
        eval_scenario = next(eval_data_generator)
        del eval_data_generator
    else:
        eval_scenario = None

    # Optional second eval set (e.g. failure cases), logged under `eval_failures/*`
    # alongside the regular `path_dataset_eval` -- joint validation for runs that
    # train/overfit on the same set. SAC only; ignored by other algorithms.
    eval_scenario_failures = None
    if (
        config["eval_freq"] > 0
        and config["algorithm"]["name"] == "SAC"
        and env_config.get("path_dataset_eval_failures")
    ):
        eval_data_generator_failures = simulator.make_data_generator(
            path=env_config["path_dataset_eval_failures"],
            max_num_objects=env_config["max_num_objects"],
            include_sdc_paths=env_config["sdc_paths_from_data"],
            batch_dims=(8, config["num_scenario_per_eval"] // 8),
            seed=69,
            distributed=True,
            num_paths=env_config["num_sdc_paths"],
            num_points_per_path=env_config["num_points_per_sdc_path"],
        )
        eval_scenario_failures = next(eval_data_generator_failures)
        del eval_data_generator_failures

    env = simulator.make_env_for_training(
        max_num_objects=env_config["max_num_objects"],
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=env_config["sdc_paths_from_data"],
        observation_type=env_config["observation_type"],
        observation_config=env_config["observation_config"],
        reward_type=env_config["reward_type"],
        reward_config=env_config["reward_config"],
        termination_keys=env_config["termination_keys"],
        goal_shaping_config=env_config["goal_shaping_config"],
    )

    absolute_run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    model_path = os.path.join(absolute_run_path, "model")
    os.makedirs(model_path, exist_ok=True)

    writer = train_utils.setup_tensorboard(absolute_run_path)
    use_wandb = bool(config.get("wandb"))
    if use_wandb:
        train_utils.setup_wandb(config, absolute_run_path)
    progress = train_utils.make_progress_fn(writer, use_wandb=use_wandb)

    ## TRAINING
    train_fn = learning.get_train_fn(config["algorithm"]["name"])

    extra_kwargs = {}
    if config["algorithm"]["name"] == "SAC":
        extra_kwargs["eval_scenario_failures"] = eval_scenario_failures
        extra_kwargs["banker"] = _build_banker(config, env_config, run_config, absolute_run_path)

    try:
        train_fn(
            env=env,
            data_generator=data_generator,
            eval_scenario=eval_scenario,
            **run_config,
            **extra_kwargs,
            progress_fn=progress,
            checkpoint_logdir=model_path,
            disable_tqdm=not sys.stdout.isatty(),
        )
    finally:
        train_utils.finish_wandb(use_wandb)


if __name__ == "__main__":
    run()
