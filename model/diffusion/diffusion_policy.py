from __future__ import annotations

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from model.diffusion.diffusion import GaussianDiffusion, UNet1DConditioned
from model.diffusion.modules.scene_tokenizer import SceneTokenizer
from model.diffusion.modules.mlp import MLP


@struct.dataclass
class PolicyFeatures:
    ego_state: jnp.ndarray
    goal_xy: jnp.ndarray
    subgoal_xy: jnp.ndarray
    remaining_timesteps: jnp.ndarray
    other_states: jnp.ndarray
    other_valid: jnp.ndarray
    map_features: jnp.ndarray
    map_valid: jnp.ndarray
    traffic_light_features: jnp.ndarray
    traffic_light_valid: jnp.ndarray
    inst_features: Optional[jnp.ndarray] = None
    inst_valid: Optional[jnp.ndarray] = None
    subgoal_valid: Optional[jnp.ndarray] = None
    ego_trajectory: Optional[jnp.ndarray] = None

    @staticmethod
    def from_mapping(features: Mapping[str, jnp.ndarray]) -> "PolicyFeatures":
        required = (
            "ego_state",
            "goal_xy",
            "subgoal_xy",
            "subgoal_valid",
            "remaining_timesteps",
            "other_states",
            "other_valid",
            "map_features",
            "map_valid",
            "traffic_light_features",
            "traffic_light_valid",
            "inst_features",
            "inst_valid",
        )
        missing = [k for k in required if k not in features]
        if missing:
            raise KeyError(f"Missing required feature keys: {missing}")
        return PolicyFeatures(
            ego_state=features["ego_state"],
            goal_xy=features["goal_xy"],
            subgoal_xy=features["subgoal_xy"],
            subgoal_valid=features["subgoal_valid"],
            remaining_timesteps=features["remaining_timesteps"],
            other_states=features["other_states"],
            other_valid=features["other_valid"],
            map_features=features["map_features"],
            map_valid=features["map_valid"],
            traffic_light_features=features["traffic_light_features"],
            traffic_light_valid=features["traffic_light_valid"],
            inst_features=features.get("inst_features"),
            inst_valid=features.get("inst_valid"),
            ego_trajectory=features.get("ego_trajectory"),
        )


class DiffusionPolicy(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        target_dim: int,
        hidden_dim: int,
        cond_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        inst_attr_dim: int,
        ego_dim: int = 5,
        other_dim: int = 15,
        predict_type: str = "v",
        predict_horizon: int = 25,
        subgoal_conditioned: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.target_dim = target_dim
        self.predict_horizon = predict_horizon
        # Whether the subgoal encoder is part of the graph. The base pretrain is
        # "without_subgoal" (subgoal_conditioned=False) and was saved WITHOUT the
        # subgoal_encoder's 12 param leaves; always building it here makes the
        # module incompatible with that checkpoint. Gate it so both the with- and
        # without-subgoal checkpoints load. (Ported from the es_baseline vendored
        # copy, which is byte-identical to the checkpoint's training code.)
        self.subgoal_conditioned = subgoal_conditioned

        self.scene_tokenizer = SceneTokenizer(
            ego_dim=ego_dim,
            other_dim=other_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            token_num=32,
            rngs=rngs,
        )
        self.instruction_encoder = MLP([inst_attr_dim, hidden_dim, hidden_dim, cond_dim], rngs=rngs)
        if self.subgoal_conditioned:
            self.subgoal_encoder = MLP([2, hidden_dim, hidden_dim, cond_dim], rngs=rngs)

        denoise_fn = UNet1DConditioned(
            in_ch=target_dim,
            cond_dim=cond_dim,
            cond2_dim=cond_dim,
            time_emb_dim=cond_dim,
            horizon=predict_horizon,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )
        self.diffusion = GaussianDiffusion(
            denoise_fn=denoise_fn,
            timesteps=100,
            predict_type=predict_type,
        )

        self._mesh: Optional[Mesh] = None
        self._cond_sharding: Optional[NamedSharding] = None
        self._mask_sharding: Optional[NamedSharding] = None
        self._traj_sharding: Optional[NamedSharding] = None
        self._init_data_parallel_sharding()

    def _init_data_parallel_sharding(self) -> None:
        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        if len(gpu_devices) <= 1:
            return
        self._mesh = Mesh(np.asarray(gpu_devices), ("data",))
        self._cond_sharding = NamedSharding(self._mesh, P("data", None))
        self._mask_sharding = NamedSharding(self._mesh, P("data"))
        self._traj_sharding = NamedSharding(self._mesh, P("data", None, None))

    @staticmethod
    def _as_features(features: PolicyFeatures | Mapping[str, jnp.ndarray]) -> PolicyFeatures:
        if isinstance(features, PolicyFeatures):
            return features
        return PolicyFeatures.from_mapping(features)

    def supports_data_parallel(self) -> bool:
        return self._mesh is not None

    def shard_condition(self, cond: jnp.ndarray) -> jnp.ndarray:
        if self._cond_sharding is None:
            return cond
        return jax.device_put(cond, self._cond_sharding)

    def shard_mask(self, mask: jnp.ndarray) -> jnp.ndarray:
        if self._mask_sharding is None:
            return mask
        return jax.device_put(mask, self._mask_sharding)

    def shard_trajectory(self, traj: jnp.ndarray) -> jnp.ndarray:
        if self._traj_sharding is None:
            return traj
        return jax.device_put(traj, self._traj_sharding)

    def _condition_impl(self, features: PolicyFeatures) -> jnp.ndarray:
        return self.scene_tokenizer._tokenize_scene_features(
            input_features=features.__dict__,
            deterministic=True,
        )
    
    def _instruction_condition_impl(self, features: PolicyFeatures) -> jnp.ndarray:
        x_inst = self.instruction_encoder(features.inst_features, deterministic=True)
        if self.subgoal_conditioned:
            x_subgoal = self.subgoal_encoder(features.subgoal_xy, deterministic=True)
            x_subgoal = x_subgoal * features.subgoal_valid[..., None]
        else:
            x_subgoal = 0
        return x_inst + x_subgoal


    def _sample_from_condition_impl(self, cond: jnp.ndarray, inst_cond: jnp.ndarray, inst_cond_mask: jnp.ndarray, rng: jax.Array, eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform") -> jnp.ndarray:
        batch_size = cond.shape[0]
        pred = self.diffusion.sample(
            shape=(batch_size, self.target_dim, self.predict_horizon),
            cond1=cond,
            cond2=inst_cond,
            cond2_mask=inst_cond_mask,
            rng=rng,
            eta=eta,
            sampling_temp=sampling_temp,
            temp_mode=temp_mode,
        )
        return jnp.transpose(pred, (0, 2, 1))

    def _loss_from_condition_impl(self, target_btd: jnp.ndarray, cond: jnp.ndarray, inst_cond: jnp.ndarray, inst_cond_mask: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        return self.diffusion.loss(jnp.transpose(target_btd, (0, 2, 1)), cond1=cond, cond2=inst_cond, cond2_mask=inst_cond_mask, rng=rng)

    # -- RL / DPPO condition-space wrappers (transpose [B,T,D] <-> [B,D,T]) --- #
    def loss_per_example_from_condition(self, target_btd: jnp.ndarray, cond: jnp.ndarray, inst_cond: jnp.ndarray, inst_cond_mask: jnp.ndarray, *, rng: jax.Array) -> jnp.ndarray:
        """Per-example diffusion loss ``[B]`` given a precomputed condition."""
        return self.diffusion.loss_per_example(
            jnp.transpose(target_btd, (0, 2, 1)), cond1=cond, cond2=inst_cond, cond2_mask=inst_cond_mask, rng=rng,
        )

    def sample_with_trace_from_condition(
        self, cond: jnp.ndarray, inst_cond: jnp.ndarray, inst_cond_mask: jnp.ndarray, *, rng: jax.Array,
        k_trainable: int, trainable_denoise_fn=None, rl_mode: bool = False,
        sigma_sample_floor: float = 0.0, sigma_logprob_floor: float = 0.1, prefix_len: int | None = None,
        eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform",
    ):
        """Sample + record last ``k_trainable`` transitions.

        Returns ``(trajectory_btd, trace)``: the trajectory is ``[B, T, D]`` (like
        :meth:`sample_from_condition`); the trace arrays (``x_t``/``x_prev``/
        ``mean``/``std``) stay in the diffusion's native ``[K, B, D, T]`` layout,
        to be fed back to :meth:`transition_log_prob_from_condition` unchanged.
        """
        batch_size = cond.shape[0]
        traj, trace = self.diffusion.sample_with_trace(
            shape=(batch_size, self.target_dim, self.predict_horizon),
            cond1=cond, cond2=inst_cond, cond2_mask=inst_cond_mask, rng=rng,
            k_trainable=k_trainable, trainable_denoise_fn=trainable_denoise_fn, rl_mode=rl_mode,
            sigma_sample_floor=sigma_sample_floor, sigma_logprob_floor=sigma_logprob_floor,
            prefix_len=prefix_len, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode,
        )
        return jnp.transpose(traj, (0, 2, 1)), trace

    def transition_log_prob_from_condition(
        self, x_t: jnp.ndarray, x_prev: jnp.ndarray, t: jnp.ndarray,
        cond: jnp.ndarray, inst_cond: jnp.ndarray, inst_cond_mask: jnp.ndarray, *,
        sigma_floor: float = 0.1, prefix_len: int | None = None, trainable_denoise_fn=None,
    ) -> jnp.ndarray:
        """Recompute the transition log-prob ``[B]`` for a stored transition.

        ``x_t``/``x_prev`` are in the diffusion's native ``[B, D, T]`` layout (as
        returned in the trace), so no transpose is needed here.
        """
        return self.diffusion.transition_log_prob(
            x_t, x_prev, t, cond1=cond, cond2=inst_cond, cond2_mask=inst_cond_mask,
            sigma_floor=sigma_floor, prefix_len=prefix_len, denoise_fn=trainable_denoise_fn,
        )

    def _resample_from_condition_impl(
        self, cond: jnp.ndarray, inst_cond: jnp.ndarray, inst_cond_mask: jnp.ndarray, proposals_btd: jnp.ndarray, n_timesteps: int, rng: jax.Array, noise_scale: float = 1.0, eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform"
    ) -> jnp.ndarray:
        x = self.diffusion.resample(jnp.transpose(proposals_btd, (0, 2, 1)), cond1=cond, cond2=inst_cond, cond2_mask=inst_cond_mask, n_timesteps=n_timesteps, rng=rng, noise_scale=noise_scale, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)
        return jnp.transpose(x, (0, 2, 1))

    def compute_condition(self, input_features: PolicyFeatures | Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        return self._condition_impl(self._as_features(input_features)), self._instruction_condition_impl(self._as_features(input_features))

    def input_features_projection(self, input_features: PolicyFeatures | Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        return self.compute_condition(input_features)

    def sample_from_condition(self, cond: jnp.ndarray, inst_cond: jnp.ndarray, inst_cond_mask: jnp.ndarray, *, rng: jax.Array, data_parallel: bool = False, eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform") -> jnp.ndarray:
        cond_in = self.shard_condition(cond) if data_parallel else cond
        inst_cond_in = self.shard_condition(inst_cond) if data_parallel else inst_cond
        inst_cond_mask_in = self.shard_mask(inst_cond_mask) if data_parallel else inst_cond_mask
        return self._sample_from_condition_impl(cond_in, inst_cond_in, inst_cond_mask_in, rng, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)

    def loss_from_condition(
        self,
        target_btd: jnp.ndarray,
        cond: jnp.ndarray,
        inst_cond: jnp.ndarray,
        inst_cond_mask: jnp.ndarray,
        *,
        rng: jax.Array,
        data_parallel: bool = False,
        pre_sharded: bool = False,
    ) -> jnp.ndarray:
        if data_parallel:
            traj = target_btd if pre_sharded else self.shard_trajectory(target_btd)
            cond_in = cond if pre_sharded else self.shard_condition(cond)
            inst_cond_in = inst_cond if pre_sharded else self.shard_condition(inst_cond)
            inst_cond_mask_in = inst_cond_mask if pre_sharded else self.shard_mask(inst_cond_mask)
            return self._loss_from_condition_impl(traj, cond_in, inst_cond_in, inst_cond_mask_in, rng)
        return self._loss_from_condition_impl(target_btd, cond, inst_cond, inst_cond_mask, rng)

    def resample_from_condition(
        self,
        cond: jnp.ndarray,
        inst_cond: jnp.ndarray,
        inst_cond_mask: jnp.ndarray,
        proposals_btd: jnp.ndarray,
        n_timesteps: int,
        *,
        rng: jax.Array,
        data_parallel: bool = False,
        noise_scale: float = 2.0,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        if data_parallel:
            cond = self.shard_condition(cond)
            inst_cond = self.shard_condition(inst_cond)
            inst_cond_mask = self.shard_mask(inst_cond_mask)
            proposals_btd = self.shard_trajectory(proposals_btd)
        return self._resample_from_condition_impl(cond, inst_cond, inst_cond_mask, proposals_btd, n_timesteps, rng, noise_scale=noise_scale, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)

    def forward(self, input_features: PolicyFeatures | Mapping[str, jnp.ndarray], rng: jax.Array, eta: float = 1.0, sampling_temp: float = 1.0, temp_mode: str = "uniform") -> jnp.ndarray:
        return self.sample(input_features, rng=rng, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)

    def sample(
        self,
        input_features: PolicyFeatures | Mapping[str, jnp.ndarray],
        *,
        rng: jax.Array,
        data_parallel: bool = False,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        cond, inst_cond = self.compute_condition(input_features)
        inst_cond_mask = input_features.inst_valid if isinstance(input_features, PolicyFeatures) else input_features["inst_valid"]
        inst_cond_mask = jnp.array(inst_cond_mask, dtype=bool)
        return self.sample_from_condition(cond, inst_cond, inst_cond_mask, rng=rng, data_parallel=data_parallel, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)

    def loss(
        self,
        input_features: PolicyFeatures | Mapping[str, jnp.ndarray],
        *,
        rng: jax.Array,
        data_parallel: bool = False,
    ) -> jnp.ndarray:
        features = self._as_features(input_features)
        if features.ego_trajectory is None:
            raise ValueError("ego_trajectory is required for loss()")
        target = features.ego_trajectory[:, : self.predict_horizon, :]
        cond, inst_cond = self.compute_condition(features)
        inst_cond_mask = features.inst_valid if isinstance(input_features, PolicyFeatures) else input_features["inst_valid"]
        inst_cond_mask = jnp.array(inst_cond_mask, dtype=bool)
        return self.loss_from_condition(target, cond, inst_cond, inst_cond_mask, rng=rng, data_parallel=data_parallel)

    def resample(
        self,
        input_features: PolicyFeatures | Mapping[str, jnp.ndarray],
        proposals: jnp.ndarray,
        n_timesteps: int,
        *,
        rng: jax.Array,
        data_parallel: bool = False,
        eta: float = 1.0,
        sampling_temp: float = 1.0,
        temp_mode: str = "uniform",
    ) -> jnp.ndarray:
        cond, inst_cond = self.compute_condition(input_features)
        inst_cond_mask = input_features.inst_valid if isinstance(input_features, PolicyFeatures) else input_features["inst_valid"]
        inst_cond_mask = jnp.array(inst_cond_mask, dtype=bool)
        return self.resample_from_condition(cond, inst_cond, inst_cond_mask, proposals, n_timesteps, rng=rng, data_parallel=data_parallel, eta=eta, sampling_temp=sampling_temp, temp_mode=temp_mode)