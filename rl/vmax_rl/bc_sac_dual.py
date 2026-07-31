"""Dual-source BC_SAC trainer: BC on expert data, SAC RL on failure cases.

This is a thin adaptation of V-Max's ``bc_sac_trainer.train`` (Valeo, 2025) that
draws its two training modes from **two different** data generators:

* **imitation steps** sample from ``imitation_data_generator`` -- the non-failure
  WOMD *expert* data (behavior-cloning warm-up / regulariser); and
* **RL steps** sample from ``rl_data_generator`` -- the harvested *failure* cases
  (reward-driven SAC, no demonstrations).

The policy network is shared, so the BC signal keeps the SAC policy close to
human driving while RL solves the failure scenarios. Everything else (networks,
losses, replay buffer, pmap loop) is reused verbatim from V-Max.
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from functools import partial
from time import perf_counter

from rl.vmax_rl import compat  # noqa: F401

import jax
import jax.numpy as jnp
from tqdm import tqdm

from vmax.agents import datatypes, pipeline
from vmax.agents.learning.hybrid import bc_sac
from vmax.agents.learning.replay_buffer import ReplayBuffer
from vmax.agents.pipeline import inference, pmap
from vmax.scripts.training import train_utils
from vmax.simulator import metrics as _metrics


if typing.TYPE_CHECKING:
    from waymax import datatypes as waymax_datatypes
    from waymax import env as waymax_env


def _load_init_params(training_state, init_params, network, num_devices: int):
    """Replace the freshly-initialized actor/critic weights with ``init_params``.

    ``bc_sac.initialize`` only ever builds random networks (it takes a PRNG key
    and nothing else), so warm-starting from a pretrained checkpoint means
    swapping the params back out afterwards. Optimizer states are rebuilt from
    the loaded params rather than carried over: the pretrained run's Adam moments
    belong to a different task, and stale moments would blow the first few
    updates away.

    ``init_params`` may be a ``SACNetworkParams`` (what V-Max's SAC trainer
    saves) or a ``BCSACNetworkParams``; only policy/value/target_value are used.
    A SAC checkpoint's ``log_alpha`` is dropped — BC_SAC uses a fixed ``alpha``.
    """
    from vmax.agents.learning.hybrid.bc_sac.bc_sac_factory import BCSACNetworkParams

    def _field(name):
        if not hasattr(init_params, name):
            raise ValueError(
                f"init_params ({type(init_params).__name__}) has no '{name}' field; "
                "expected a SACNetworkParams/BCSACNetworkParams checkpoint."
            )
        return getattr(init_params, name)

    target = getattr(init_params, "target_value", None)
    loaded = BCSACNetworkParams(
        policy=_field("policy"),
        value=_field("value"),
        target_value=target if target is not None else _field("value"),
    )

    # The training state is already replicated across devices; compare against a
    # single-device copy so the shape check sees real per-device shapes.
    reference = pmap.unpmap(training_state).params
    ref_shapes = [x.shape for x in jax.tree_util.tree_leaves(reference)]
    new_shapes = [x.shape for x in jax.tree_util.tree_leaves(loaded)]
    if ref_shapes != new_shapes:
        raise ValueError(
            "init_params does not match the network built from this run's config.\n"
            f"  expected {len(ref_shapes)} arrays, first few: {ref_shapes[:4]}\n"
            f"  checkpoint {len(new_shapes)} arrays, first few: {new_shapes[:4]}\n"
            "The checkpoint was almost certainly trained with different "
            "network.policy/value layer_sizes or a different observation_config. "
            "Match them (e.g. --override algorithm.network.policy.layer_sizes=[256,256]) "
            "and retry."
        )

    single = pmap.unpmap(training_state)
    single = single.replace(
        params=loaded,
        rl_policy_optimizer_state=network.rl_policy_optimizer.init(loaded.policy),
        imitation_policy_optimizer_state=network.imitation_policy_optimizer.init(loaded.policy),
        value_optimizer_state=network.value_optimizer.init(loaded.value),
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(loaded))
    print(f"-> Warm-started actor+critic from checkpoint ({n_params:,} params); optimizer states reset.")

    return pmap.device_put_replicated(single, jax.local_devices()[:num_devices])


def train(
    env: "waymax_env.PlanningAgentEnvironment",
    rl_data_generator: typing.Iterator["waymax_datatypes.SimulatorState"],
    imitation_data_generator: typing.Iterator["waymax_datatypes.SimulatorState"],
    *,
    total_timesteps: int,
    num_envs: int,
    num_episode_per_epoch: int,
    scenario_length: int,
    log_freq: int,
    seed: int,
    learning_start: int,
    alpha: float,
    discount: float,
    tau: float,
    imitation_frequency: int,
    imitation_unroll_length: int,
    loss_type: str,
    save_freq: int,
    buffer_size: int,
    batch_size: int,
    rl_learning_rate: float,
    imitation_learning_rate: float,
    grad_updates_per_step: int,
    unroll_length: int,
    network_config: dict,
    eval_scenario: "waymax_datatypes.SimulatorState" = None,
    num_scenario_per_eval: int = 0,
    eval_freq: int = 0,
    progress_fn: Callable[[int, dict], None] = lambda *args: None,
    checkpoint_logdir: str = "",
    disable_tqdm: bool = False,
    init_params: Any = None,
) -> Any:  # noqa: F821 - returns final params pytree
    """Train BC_SAC with separate imitation (expert) and RL (failure) data sources."""
    print(" BC_SAC (dual-source) ".center(48, "="))

    rng = jax.random.PRNGKey(seed)
    num_devices = jax.local_device_count()

    do_save = save_freq > 1 and checkpoint_logdir is not None
    do_evaluation = eval_freq >= 1 and eval_scenario is not None
    # imitation_data_generator=None -> pure SAC (no behavior-cloning steps).
    do_imitation = imitation_data_generator is not None
    # Every iteration is BC when imitation_frequency==1 (skip RL buffer/prefill).
    bc_only = do_imitation and imitation_frequency == 1

    num_steps = num_episode_per_epoch * scenario_length
    env_steps_per_iter = num_steps * num_envs
    total_iters = (total_timesteps // env_steps_per_iter) + 1

    observation_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    rng, network_key = jax.random.split(rng)

    print("-> Initializing networks...")
    network, training_state, policy_fn = bc_sac.initialize(
        action_size,
        observation_size,
        env,
        rl_learning_rate,
        imitation_learning_rate,
        network_config,
        num_devices,
        network_key,
    )
    if init_params is not None:
        training_state = _load_init_params(training_state, init_params, network, num_devices)

    rl_learning_fn = bc_sac.make_rl_sgd_step(network, alpha, discount, tau)
    imitation_learning_fn = bc_sac.make_imitation_sgd_step(network, loss_type)

    rl_step_fn = partial(inference.policy_step, use_partial_transition=True)
    imitation_step_fn = partial(inference.expert_step, use_partial_transition=True)

    def _dummy_transition():
        return datatypes.RLPartialTransition(
            observation=jnp.zeros((observation_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            flag=0,
            done=0,
        )

    rl_replay_buffer = ReplayBuffer(
        buffer_size=buffer_size // num_devices,
        batch_size=batch_size * grad_updates_per_step // num_devices,
        samples_size=num_envs,
        dummy_data_sample=_dummy_transition(),
    )
    imitation_replay_buffer = ReplayBuffer(
        buffer_size=buffer_size // num_devices,
        batch_size=batch_size * grad_updates_per_step // num_devices,
        samples_size=num_envs,
        dummy_data_sample=_dummy_transition(),
    )
    print("-> Initializing networks... Done.")

    rl_unroll_fn = partial(inference.generate_unroll, unroll_length=unroll_length, env=env, step_fn=rl_step_fn)
    rl_run_training = jax.pmap(
        partial(
            pipeline.run_training_off_policy,
            replay_buffer=rl_replay_buffer,
            env=env,
            learning_fn=rl_learning_fn,
            policy_fn=policy_fn,
            unroll_fn=rl_unroll_fn,
            grad_updates_per_step=grad_updates_per_step,
            scan_length=num_steps // unroll_length,
        ),
        axis_name="batch",
    )

    imitation_unroll_fn = partial(
        inference.generate_unroll, unroll_length=imitation_unroll_length, env=env, step_fn=imitation_step_fn
    )
    imitation_run_training = jax.pmap(
        partial(
            pipeline.run_training_off_policy,
            replay_buffer=imitation_replay_buffer,
            env=env,
            learning_fn=imitation_learning_fn,
            policy_fn=policy_fn,
            unroll_fn=imitation_unroll_fn,
            grad_updates_per_step=grad_updates_per_step,
            scan_length=num_steps // imitation_unroll_length,
        ),
        axis_name="batch",
    )

    run_evaluation = None
    if do_evaluation:
        run_evaluation = jax.pmap(
            partial(
                pipeline.run_evaluation,
                env=env,
                policy_fn=policy_fn,
                step_fn=rl_step_fn,
                scan_length=scenario_length * num_scenario_per_eval,
            ),
            axis_name="batch",
        )

    print("-> Prefilling replay buffers...")
    rl_buffer_state = None
    if not bc_only:
        # RL prefill from failure cases.
        prefill_rl = jax.pmap(
            partial(
                pipeline.prefill_replay_buffer,
                env=env,
                replay_buffer=rl_replay_buffer,
                action_shape=(num_envs, action_size),
                learning_start=learning_start,
            ),
            axis_name="batch",
        )
        rng, rb_key = jax.random.split(rng)
        rl_buffer_state = jax.pmap(rl_replay_buffer.init)(jax.random.split(rb_key, num_devices))
        rng, prefill_key = jax.random.split(rng)
        rl_buffer_state = prefill_rl(next(rl_data_generator), rl_buffer_state, jax.random.split(prefill_key, num_devices))
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), rl_buffer_state)

    # Imitation prefill from expert data (skipped for pure SAC).
    imitation_buffer_state = None
    if do_imitation:
        prefill_il = jax.pmap(
            partial(
                pipeline.prefill_replay_buffer,
                env=env,
                replay_buffer=imitation_replay_buffer,
                action_shape=(num_envs, action_size),
                learning_start=learning_start,
            ),
            axis_name="batch",
        )
        rng, rb_key, prefill_key = jax.random.split(rng, 3)
        imitation_buffer_state = jax.pmap(imitation_replay_buffer.init)(jax.random.split(rb_key, num_devices))
        imitation_buffer_state = prefill_il(
            next(imitation_data_generator), imitation_buffer_state, jax.random.split(prefill_key, num_devices)
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), imitation_buffer_state)
    print("-> Prefilling replay buffers... Done.")

    time_training = perf_counter()
    current_step = 0

    print("-> Ground Control to Major Tom...")
    for it in tqdm(range(total_iters), desc="Training", total=total_iters, dynamic_ncols=True, disable=disable_tqdm):
        rng, iter_key = jax.random.split(rng)
        iter_keys = jax.random.split(iter_key, num_devices)

        is_imitation = bc_only or (do_imitation and (it % imitation_frequency) == 0)

        t = perf_counter()
        if is_imitation:
            batch_scenarios = next(imitation_data_generator)
            run_training = imitation_run_training
            buffer_state = imitation_buffer_state
        else:
            batch_scenarios = next(rl_data_generator)
            run_training = rl_run_training
            buffer_state = rl_buffer_state
        epoch_data_time = perf_counter() - t

        t = perf_counter()
        training_state, buffer_state, training_metrics = run_training(
            batch_scenarios, training_state, buffer_state, iter_keys
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
        epoch_training_time = perf_counter() - t

        if is_imitation:
            imitation_buffer_state = buffer_state
        else:
            rl_buffer_state = buffer_state

        t = perf_counter()
        training_metrics = pmap.flatten_tree(training_metrics)
        training_metrics = jax.device_get(training_metrics)
        training_metrics = _metrics.collect(training_metrics, "ep_len_mean")
        current_step = int(pmap.unpmap(training_state.env_steps))

        metrics = {
            "runtime/sps": int(env_steps_per_iter / max(epoch_training_time, 1e-6)),
            "train/mode": 1.0 if is_imitation else 0.0,
            **{f"{name}": value for name, value in training_metrics.items()},
        }
        # Tag rollout metrics by mode so W&B curves don't mix expert (+5) and
        # policy (−) rewards into one misleading ``ep_rew_mean``.
        mode_prefix = "il_" if is_imitation else "rl_"
        for rollout_key in ("ep_rew_mean", "ep_len_mean"):
            if rollout_key in metrics:
                metrics[f"{mode_prefix}{rollout_key}"] = metrics[rollout_key]
        for loss_key in ("imitation_loss", "policy_loss", "value_loss"):
            if loss_key in metrics:
                metrics[f"{mode_prefix}{loss_key}"] = metrics[loss_key]

        if do_save and (it % save_freq == 0):
            train_utils.save_params(f"{checkpoint_logdir}/model_{current_step}.pkl", pmap.unpmap(training_state.params))
        epoch_log_time = perf_counter() - t

        t = perf_counter()
        if run_evaluation is not None and (it % eval_freq == 0):
            eval_metrics = run_evaluation(eval_scenario, training_state)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), eval_metrics)
            eval_metrics = _metrics.collect(pmap.flatten_tree(eval_metrics), "ep_len_mean")
            progress_fn(current_step, eval_metrics)
        epoch_eval_time = perf_counter() - t

        if it % log_freq == 0:
            metrics["runtime/data_time"] = epoch_data_time
            metrics["runtime/training_time"] = epoch_training_time
            metrics["runtime/log_time"] = epoch_log_time
            metrics["runtime/eval_time"] = epoch_eval_time
            metrics["runtime/wall_time"] = perf_counter() - time_training
            metrics["train/rl_gradient_steps"] = int(pmap.unpmap(training_state.rl_gradient_steps))
            metrics["train/il_gradient_steps"] = int(pmap.unpmap(training_state.il_gradient_steps))
            metrics["train/env_steps"] = current_step
            progress_fn(current_step, metrics, total_timesteps)
            if disable_tqdm:
                print(f"-> Step {current_step}/{total_timesteps} - {(current_step / total_timesteps) * 100:.2f}%")

    print(f"-> Training took {perf_counter() - time_training:.2f}s")

    final_params = pmap.unpmap(training_state.params)
    if checkpoint_logdir:
        train_utils.save_params(f"{checkpoint_logdir}/model_final.pkl", final_params)

    return final_params
