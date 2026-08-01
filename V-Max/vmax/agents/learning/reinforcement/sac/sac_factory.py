# Copyright 2025 Valeo.

"""Factory functions for the Soft Actor-Critic (SAC) algorithm."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from vmax.agents import datatypes, networks
from vmax.agents.pipeline import pmap


@flax.struct.dataclass
class SACNetworkParams:
    """Parameters for SAC network."""

    policy: datatypes.Params
    value: datatypes.Params
    target_value: datatypes.Params
    log_alpha: jnp.ndarray


@flax.struct.dataclass
class SACNetworks:
    """SAC networks."""

    policy_network: Any
    value_network: Any
    parametric_action_distribution: Any
    policy_optimizer: Any
    value_optimizer: Any
    alpha_optimizer: Any


@flax.struct.dataclass
class SACTrainingState(datatypes.TrainingState):
    """Training state for SAC algorithm."""

    params: SACNetworkParams
    policy_optimizer_state: optax.OptState
    value_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    rl_gradient_steps: int


def initialize(
    action_size: int,
    observation_size: int,
    env: Any,
    learning_rate: float,
    network_config: dict,
    num_devices: int,
    key: jax.Array,
    init_checkpoint: str | None = None,
    alpha: float = 0.2,
) -> tuple[SACNetworks, SACTrainingState, datatypes.Policy]:
    """Initialize SAC components.

    Args:
        action_size: Size of the action space.
        observation_size: Size of the observation space.
        env: Environment instance with a features extractor.
        learning_rate: Learning rate for the optimizers.
        network_config: Network configuration dictionary.
        num_devices: Number of devices to use.
        key: Random key for initialization.
        init_checkpoint: Optional path to a prior run's ``model_*.pkl``. When given,
            policy/value/target_value params are loaded from it instead of a fresh
            random init (network_config must produce matching param shapes).
            Optimizer states still start fresh.
        alpha: Initial entropy coefficient. Constant when ``auto_alpha`` is off in
            ``make_sgd_step``; otherwise only the starting point for the tuned value.

    Returns:
        A tuple of (networks, training state, policy function).

    """
    network = make_networks(
        observation_size=observation_size,
        action_size=action_size,
        unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
        learning_rate=learning_rate,
        network_config=network_config,
    )

    policy_function = make_inference_fn(network)

    key_policy, key_value = jax.random.split(key)

    policy_params = network.policy_network.init(key_policy)
    policy_optimizer_state = network.policy_optimizer.init(policy_params)
    value_params = network.value_network.init(key_value)
    value_optimizer_state = network.value_optimizer.init(value_params)
    log_alpha = jnp.asarray(jnp.log(alpha), dtype=jnp.float32)

    init_params = SACNetworkParams(
        policy=policy_params,
        value=value_params,
        target_value=value_params,
        log_alpha=log_alpha,
    )

    if init_checkpoint:
        from vmax.scripts.evaluate import utils as eval_utils

        print(f"-> Warm-starting SAC params from {init_checkpoint} ...")
        restored = eval_utils.load_params(init_checkpoint)
        init_params = SACNetworkParams(
            policy=restored.policy,
            value=restored.value,
            target_value=restored.target_value,
            # Checkpoints predating auto entropy tuning have no log_alpha; fall back to
            # the configured initial alpha rather than failing the warm start.
            log_alpha=getattr(restored, "log_alpha", log_alpha),
        )

    alpha_optimizer_state = network.alpha_optimizer.init(init_params.log_alpha)

    training_state = SACTrainingState(
        params=init_params,
        policy_optimizer_state=policy_optimizer_state,
        value_optimizer_state=value_optimizer_state,
        alpha_optimizer_state=alpha_optimizer_state,
        env_steps=0,
        rl_gradient_steps=0,
    )

    training_state = pmap.device_put_replicated(training_state, jax.local_devices()[:num_devices])

    return network, training_state, policy_function


def make_inference_fn(sac_network: SACNetworks) -> datatypes.Policy:
    """Create the policy inference function for SAC.

    Args:
        sac_network: Instance of SACNetworks.

    Returns:
        A callable policy function.

    """

    def make_policy(params: datatypes.Params, deterministic: bool = False) -> datatypes.Policy:
        policy_network = sac_network.policy_network
        parametric_action_distribution = sac_network.parametric_action_distribution

        def policy(observations: jax.Array, key_sample: jax.Array = None) -> tuple[jax.Array, dict]:
            logits = policy_network.apply(params, observations)

            if deterministic:
                return parametric_action_distribution.mode(logits), {}

            return parametric_action_distribution.sample(logits, key_sample), {}

        return policy

    return make_policy


def make_networks(
    observation_size: int,
    action_size: int,
    unflatten_fn: callable,
    learning_rate: int,
    network_config: dict,
) -> SACNetworks:
    """Construct SAC networks.

    Args:
        observation_size: Size of the observation space.
        action_size: Size of the action space.
        unflatten_fn: Function to unflatten network inputs.
        learning_rate: Learning rate used for the optimizers.
        network_config: Network configuration dictionary.

    Returns:
        An instance of SACNetworks.

    """
    if "gaussian" in network_config["action_distribution"]:
        parametric_action_distribution = networks.NormalTanhDistribution(event_size=action_size)
    elif "beta" in network_config["action_distribution"]:
        parametric_action_distribution = networks.BetaDistribution(event_size=action_size)

    output_size = parametric_action_distribution.param_size

    policy_network = networks.make_policy_network(network_config, observation_size, output_size, unflatten_fn)
    value_network = networks.make_value_network(network_config, observation_size, action_size, unflatten_fn)

    policy_optimizer = optax.adam(learning_rate)
    value_optimizer = optax.adam(learning_rate)
    alpha_optimizer = optax.adam(learning_rate)

    return SACNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        alpha_optimizer=alpha_optimizer,
    )


def make_sgd_step(
    sac_network: SACNetworks,
    alpha: float,
    discount: float,
    tau: float,
    auto_alpha: bool = False,
    target_entropy: float | None = None,
    action_size: int | None = None,
    kl_coef: float = 0.0,
    reference_policy_params: datatypes.Params | None = None,
) -> datatypes.LearningFunction:
    """Create the SGD step function for SAC.

    Args:
        sac_network: The SAC networks.
        alpha: Entropy regularization coefficient (initial value when ``auto_alpha``).
        discount: Discount factor.
        tau: Coefficient for target network updates.
        auto_alpha: Whether to tune alpha against the target entropy.
        target_entropy: Target entropy. Defaults to ``-action_size`` when not given.
        action_size: Action dimensionality, used for the default target entropy.
        kl_coef: Weight on KL(pi || reference_policy_params). 0 disables the anchor
            (and skips the extra forward pass entirely).
        reference_policy_params: Frozen policy params to stay close to. Ignored when
            ``kl_coef`` is 0.

    Returns:
        A function that executes an SGD step.

    """
    if target_entropy is None:
        if action_size is None:
            raise ValueError("make_sgd_step needs either target_entropy or action_size to set the target entropy.")

        target_entropy = -float(action_size)

    value_loss, policy_loss, alpha_loss = _make_loss_fn(
        sac_network=sac_network,
        alpha=alpha,
        discount=discount,
        auto_alpha=auto_alpha,
        target_entropy=target_entropy,
        kl_coef=kl_coef,
        reference_policy_params=reference_policy_params,
    )

    policy_update = networks.gradient_update_fn(
        policy_loss, sac_network.policy_optimizer, pmap_axis_name="batch", has_aux=True
    )
    value_update = networks.gradient_update_fn(value_loss, sac_network.value_optimizer, pmap_axis_name="batch")
    alpha_update = networks.gradient_update_fn(alpha_loss, sac_network.alpha_optimizer, pmap_axis_name="batch")

    def sgd_step(
        carry: tuple[SACTrainingState, jax.Array],
        transitions: datatypes.RLTransition,
    ) -> tuple[tuple[SACTrainingState, jax.Array], datatypes.Metrics]:
        training_state, key = carry

        key, key_alpha, key_value, key_policy = jax.random.split(key, 4)

        log_alpha = training_state.params.log_alpha

        if auto_alpha:
            alpha_loss, log_alpha, alpha_optimizer_state = alpha_update(
                training_state.params.log_alpha,
                training_state.params.policy,
                transitions,
                key_alpha,
                optimizer_state=training_state.alpha_optimizer_state,
            )
        else:
            alpha_loss = jnp.zeros(())
            alpha_optimizer_state = training_state.alpha_optimizer_state

        value_loss, value_params, value_optimizer_state = value_update(
            training_state.params.value,
            training_state.params.policy,
            training_state.params.target_value,
            log_alpha,
            transitions,
            key_value,
            optimizer_state=training_state.value_optimizer_state,
        )
        (policy_loss, policy_aux), policy_params, policy_optimizer_state = policy_update(
            training_state.params.policy,
            training_state.params.value,
            log_alpha,
            transitions,
            key_policy,
            optimizer_state=training_state.policy_optimizer_state,
        )

        new_target_value_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.params.target_value,
            value_params,
        )

        sgd_metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(log_alpha) if auto_alpha else jnp.asarray(alpha, dtype=jnp.float32),
            "kl_to_ref": policy_aux["kl_to_ref"],
        }

        params = SACNetworkParams(
            policy=policy_params,
            value=value_params,
            target_value=new_target_value_params,
            log_alpha=log_alpha,
        )

        training_state = training_state.replace(
            params=params,
            policy_optimizer_state=policy_optimizer_state,
            value_optimizer_state=value_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
            rl_gradient_steps=training_state.rl_gradient_steps + 1,
        )

        return (training_state, key), sgd_metrics

    return sgd_step


def _make_loss_fn(
    sac_network: SACNetworks,
    alpha: float,
    discount: float,
    auto_alpha: bool,
    target_entropy: float,
    kl_coef: float = 0.0,
    reference_policy_params: datatypes.Params | None = None,
) -> tuple[callable, callable, callable]:
    """Define the loss functions for SAC.

    Args:
        sac_network: The SAC networks.
        alpha: Entropy regularization coefficient. Used as a constant when ``auto_alpha``
            is False; otherwise it is only the initial value of the tuned coefficient.
        discount: Discount factor.
        auto_alpha: Whether to tune alpha against ``target_entropy`` (Haarnoja et al.,
            2018). A fixed alpha keeps injecting the same action noise no matter how
            confident the policy becomes, which over an 80-step episode quietly costs
            displacement -- exactly what the goal-reaching metric charges for.
        target_entropy: Entropy the tuned policy is driven towards. Conventionally
            ``-action_size``.

    Returns:
        A tuple containing the value, policy and alpha loss functions.

    """
    policy_network = sac_network.policy_network
    value_network = sac_network.value_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def get_alpha(log_alpha: jnp.ndarray) -> jax.Array:
        return jnp.exp(log_alpha) if auto_alpha else jnp.asarray(alpha, dtype=jnp.float32)

    def compute_value_loss(
        value_params: datatypes.Params,
        policy_params: datatypes.Params,
        target_value_params: datatypes.Params,
        log_alpha: jnp.ndarray,
        transitions: datatypes.RLTransition,
        key: jax.Array,
    ) -> jax.Array:
        value_old_action = value_network.apply(value_params, transitions.observation, transitions.action)
        next_dist_params = policy_network.apply(policy_params, transitions.next_observation)

        next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
        next_log_prob = parametric_action_distribution.log_prob(next_dist_params, next_action)
        next_action = parametric_action_distribution.postprocess(next_action)

        next_value = value_network.apply(target_value_params, transitions.next_observation, next_action)
        next_v = jnp.min(next_value, axis=-1) - get_alpha(log_alpha) * next_log_prob

        target_value = jax.lax.stop_gradient(transitions.reward + transitions.flag * discount * next_v)
        value_error = value_old_action - jnp.expand_dims(target_value, -1)
        value_loss = 0.5 * jnp.mean(jnp.square(value_error))

        return value_loss

    def compute_policy_loss(
        policy_params: datatypes.Params,
        value_params: datatypes.Params,
        log_alpha: jnp.ndarray,
        transitions: datatypes.RLTransition,
        key: jax.Array,
    ) -> tuple[jax.Array, datatypes.Metrics]:
        dist_params = policy_network.apply(policy_params, transitions.observation)

        raw_action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, raw_action)
        action = parametric_action_distribution.postprocess(raw_action)

        value_action = value_network.apply(value_params, transitions.observation, action)
        min_value = jnp.min(value_action, axis=-1)
        policy_loss = jax.lax.stop_gradient(get_alpha(log_alpha)) * log_prob - min_value

        # Anchor to a frozen reference policy (typically the pretrained checkpoint
        # this run warm-started from). SAC's own objective has nothing holding it
        # near the pretrained driver, so on a tiny overfit set it is free to drift
        # into whatever reaches the goal. This penalises KL(pi || pi_ref), keeping
        # rollouts recognisable as the reference's driving.
        #
        # Single-sample estimator: KL = E_{a~pi}[log pi(a|s) - log pi_ref(a|s)],
        # evaluated on the reparameterised sample so it is differentiable. Both
        # log-probs carry the same tanh log-det correction, so it cancels in the
        # difference and this is effectively the KL of the pre-squash Gaussians.
        if kl_coef > 0.0 and reference_policy_params is not None:
            ref_dist_params = policy_network.apply(reference_policy_params, transitions.observation)
            ref_log_prob = parametric_action_distribution.log_prob(ref_dist_params, raw_action)
            kl_to_ref = log_prob - ref_log_prob
            policy_loss = policy_loss + kl_coef * kl_to_ref
        else:
            kl_to_ref = jnp.zeros_like(log_prob)

        return jnp.mean(policy_loss), {"kl_to_ref": jnp.mean(kl_to_ref)}

    def compute_alpha_loss(
        log_alpha: jnp.ndarray,
        policy_params: datatypes.Params,
        transitions: datatypes.RLTransition,
        key: jax.Array,
    ) -> jax.Array:
        dist_params = policy_network.apply(policy_params, transitions.observation)

        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)

        # Drives the policy's entropy towards target_entropy: alpha grows while the
        # policy is more deterministic than the target, and shrinks once it is noisier.
        alpha_loss = -jnp.exp(log_alpha) * jax.lax.stop_gradient(log_prob + target_entropy)

        return jnp.mean(alpha_loss)

    return compute_value_loss, compute_policy_loss, compute_alpha_loss
