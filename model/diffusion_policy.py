from __future__ import annotations

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .diffusion import GaussianDiffusion, UNet1DConditioned
from .mlp import MLP
from .point_net import PointNet


@struct.dataclass
class PolicyFeatures:
    ego_state: jnp.ndarray
    other_states: jnp.ndarray
    other_valid: jnp.ndarray
    map_features: jnp.ndarray
    map_valid: jnp.ndarray
    traffic_light_features: jnp.ndarray
    traffic_light_valid: jnp.ndarray
    ego_trajectory: Optional[jnp.ndarray] = None

    @staticmethod
    def from_mapping(features: Mapping[str, jnp.ndarray]) -> "PolicyFeatures":
        required = (
            "ego_state",
            "other_states",
            "other_valid",
            "map_features",
            "map_valid",
            "traffic_light_features",
            "traffic_light_valid",
        )
        missing = [k for k in required if k not in features]
        if missing:
            raise KeyError(f"Missing required feature keys: {missing}")
        return PolicyFeatures(
            ego_state=features["ego_state"],
            other_states=features["other_states"],
            other_valid=features["other_valid"],
            map_features=features["map_features"],
            map_valid=features["map_valid"],
            traffic_light_features=features["traffic_light_features"],
            traffic_light_valid=features["traffic_light_valid"],
            ego_trajectory=features.get("ego_trajectory"),
        )


class DiffusionPolicy(nnx.Module):
    def __init__(
        self,
        target_dim: int,
        hidden_dim: int,
        cond_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        rngs: nnx.Rngs,
        ego_dim: int = 5,
        other_dim: int = 7,
        predict_type: str = "v",
        predict_horizon: int = 16,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.target_dim = target_dim
        self.predict_horizon = predict_horizon

        self.input_projections = {
            "ego": MLP([ego_dim, hidden_dim, hidden_dim, hidden_dim], rngs=rngs),
            "other": PointNet(other_dim, hidden_dim, rngs=rngs),
            "map": PointNet(map_attr_dim, hidden_dim, rngs=rngs),
            "tl": PointNet(tl_attr_dim, hidden_dim, rngs=rngs),
        }
        self.cond_projection = MLP([hidden_dim * 4, hidden_dim, cond_dim], rngs=rngs)

        denoise_fn = UNet1DConditioned(
            in_ch=target_dim,
            base_ch=hidden_dim,
            cond_dim=cond_dim,
            time_emb_dim=hidden_dim,
            rngs=rngs,
        )
        self.diffusion = GaussianDiffusion(
            denoise_fn=denoise_fn,
            timesteps=100,
            predict_type=predict_type,
        )

        self._jit_condition = nnx.jit(self._condition_impl)
        self._jit_sample_from_condition = nnx.jit(self._sample_from_condition_impl)
        self._jit_loss_from_condition = nnx.jit(self._loss_from_condition_impl)
        self._jit_resample_from_condition = nnx.jit(self._resample_from_condition_impl, static_argnums=(2,))

        self._mesh: Optional[Mesh] = None
        self._cond_sharding: Optional[NamedSharding] = None
        self._traj_sharding: Optional[NamedSharding] = None
        self._replicated_sharding: Optional[NamedSharding] = None
        self._jit_sample_from_condition_sharded = None
        self._jit_loss_from_condition_sharded = None
        self._init_data_parallel_jits()

    def _init_data_parallel_jits(self) -> None:
        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        if len(gpu_devices) <= 1:
            return
        self._mesh = Mesh(np.asarray(gpu_devices), ("data",))
        self._cond_sharding = NamedSharding(self._mesh, P("data", None))
        self._traj_sharding = NamedSharding(self._mesh, P("data", None, None))
        self._replicated_sharding = NamedSharding(self._mesh, P())
        self._jit_sample_from_condition_sharded = jax.jit(
            self._sample_from_condition_impl,
            in_shardings=(self._cond_sharding, self._replicated_sharding),
            out_shardings=self._traj_sharding,
        )
        self._jit_loss_from_condition_sharded = jax.jit(
            self._loss_from_condition_impl,
            in_shardings=(self._traj_sharding, self._cond_sharding, self._replicated_sharding),
            out_shardings=self._replicated_sharding,
        )

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

    def shard_trajectory(self, traj: jnp.ndarray) -> jnp.ndarray:
        if self._traj_sharding is None:
            return traj
        return jax.device_put(traj, self._traj_sharding)

    def _condition_impl(self, features: PolicyFeatures) -> jnp.ndarray:
        ego_feature = self.input_projections["ego"](features.ego_state, deterministic=True)
        other_feature = self.input_projections["other"](
            features.other_states,
            features.other_valid,
            deterministic=True,
        )
        map_features = self.input_projections["map"](
            features.map_features,
            features.map_valid,
            deterministic=True,
        )
        traffic_light_features = self.input_projections["tl"](
            features.traffic_light_features,
            features.traffic_light_valid,
            deterministic=True,
        )
        cond = jnp.concatenate([ego_feature, other_feature, map_features, traffic_light_features], axis=-1)
        return self.cond_projection(cond, deterministic=True)

    def _sample_from_condition_impl(self, cond: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        batch_size = cond.shape[0]
        pred = self.diffusion.sample(
            shape=(batch_size, self.target_dim, self.predict_horizon),
            cond=cond,
            rng=rng,
        )
        return jnp.transpose(pred, (0, 2, 1))

    def _loss_from_condition_impl(self, target_btd: jnp.ndarray, cond: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        return self.diffusion.loss(jnp.transpose(target_btd, (0, 2, 1)), cond=cond, rng=rng)

    def _resample_from_condition_impl(
        self, cond: jnp.ndarray, proposals_btd: jnp.ndarray, n_timesteps: int, rng: jax.Array
    ) -> jnp.ndarray:
        x = self.diffusion.resample(jnp.transpose(proposals_btd, (0, 2, 1)), cond, n_timesteps=n_timesteps, rng=rng)
        return jnp.transpose(x, (0, 2, 1))

    def compute_condition(self, input_features: PolicyFeatures | Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        return self._jit_condition(self._as_features(input_features))

    def input_features_projection(self, input_features: PolicyFeatures | Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        return self.compute_condition(input_features)

    def sample_from_condition(self, cond: jnp.ndarray, *, rng: jax.Array, data_parallel: bool = False) -> jnp.ndarray:
        if data_parallel and self._jit_sample_from_condition_sharded is not None:
            return self._jit_sample_from_condition_sharded(self.shard_condition(cond), rng)
        return self._jit_sample_from_condition(cond, rng)

    def loss_from_condition(
        self, target_btd: jnp.ndarray, cond: jnp.ndarray, *, rng: jax.Array, data_parallel: bool = False
    ) -> jnp.ndarray:
        if data_parallel and self._jit_loss_from_condition_sharded is not None:
            return self._jit_loss_from_condition_sharded(self.shard_trajectory(target_btd), self.shard_condition(cond), rng)
        return self._jit_loss_from_condition(target_btd, cond, rng)

    def resample_from_condition(
        self,
        cond: jnp.ndarray,
        proposals_btd: jnp.ndarray,
        n_timesteps: int,
        *,
        rng: jax.Array,
        data_parallel: bool = False,
    ) -> jnp.ndarray:
        if data_parallel:
            cond = self.shard_condition(cond)
            proposals_btd = self.shard_trajectory(proposals_btd)
        return self._jit_resample_from_condition(cond, proposals_btd, n_timesteps, rng)

    def forward(self, input_features: PolicyFeatures | Mapping[str, jnp.ndarray], rng: jax.Array) -> jnp.ndarray:
        return self.sample(input_features, rng=rng)

    def sample(
        self,
        input_features: PolicyFeatures | Mapping[str, jnp.ndarray],
        *,
        rng: jax.Array,
        data_parallel: bool = False,
    ) -> jnp.ndarray:
        cond = self.compute_condition(input_features)
        return self.sample_from_condition(cond, rng=rng, data_parallel=data_parallel)

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
        cond = self.compute_condition(features)
        return self.loss_from_condition(target, cond, rng=rng, data_parallel=data_parallel)

    def resample(
        self,
        input_features: PolicyFeatures | Mapping[str, jnp.ndarray],
        proposals: jnp.ndarray,
        n_timesteps: int,
        *,
        rng: jax.Array,
        data_parallel: bool = False,
    ) -> jnp.ndarray:
        cond = self.compute_condition(input_features)
        return self.resample_from_condition(cond, proposals, n_timesteps, rng=rng, data_parallel=data_parallel)
