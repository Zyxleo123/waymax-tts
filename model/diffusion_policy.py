from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from .diffusion import GaussianDiffusion, UNet1DConditioned
from .mlp import MLP
from .point_net import PointNet


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
            timesteps=50,
            predict_type=predict_type,
        )

        self._jit_input_features_projection = nnx.jit(self._input_features_projection_impl)
        self._jit_loss_with_noise_t = nnx.jit(self._loss_with_noise_t_impl)
        self._jit_sample_with_fixed_noises = nnx.jit(self._sample_with_fixed_noises_impl)
        self._jit_resample_with_fixed_noises = nnx.jit(self._resample_with_fixed_noises_impl, static_argnums=(2,))

    def _input_features_projection_impl(self, input_features: Dict[str, jnp.ndarray], deterministic: bool = True) -> jnp.ndarray:
        ego_feature = self.input_projections["ego"](input_features["ego_state"], deterministic=deterministic)
        other_feature = self.input_projections["other"](
            input_features["other_states"],
            input_features["other_valid"],
            deterministic=deterministic,
        )
        map_features = self.input_projections["map"](
            input_features["map_features"],
            input_features["map_valid"],
            deterministic=deterministic,
        )
        traffic_light_features = self.input_projections["tl"](
            input_features["traffic_light_features"],
            input_features["traffic_light_valid"],
            deterministic=deterministic,
        )

        cond = jnp.concatenate([ego_feature, other_feature, map_features, traffic_light_features], axis=-1)
        return self.cond_projection(cond, deterministic=deterministic)

    def input_features_projection(self, input_features: Dict[str, jnp.ndarray], deterministic: bool = True) -> jnp.ndarray:
        return self._jit_input_features_projection(input_features, deterministic)

    def _sample_with_fixed_noises_impl(
        self,
        input_features: Dict[str, jnp.ndarray],
        x_init: jnp.ndarray,
        step_noises: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size = input_features["ego_state"].shape[0]
        cond = self._input_features_projection_impl(input_features, deterministic=deterministic)
        pred = self.diffusion.sample(
            shape=(batch_size, self.target_dim, self.predict_horizon),
            cond=cond,
            x_init=x_init,
            step_noises=step_noises,
        )
        return jnp.transpose(pred, (0, 2, 1))

    def _loss_with_noise_t_impl(
        self,
        input_features: Dict[str, jnp.ndarray],
        noise: jnp.ndarray,
        t: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        target = input_features["ego_trajectory"][:, : self.predict_horizon, :]
        cond = self._input_features_projection_impl(input_features, deterministic=deterministic)
        return self.diffusion._loss_with_noise_t_impl(jnp.transpose(target, (0, 2, 1)), cond=cond, noise=noise, t=t)

    def _resample_with_fixed_noises_impl(
        self,
        input_features: Dict[str, jnp.ndarray],
        proposals: jnp.ndarray,
        n_timesteps: int,
        q_noise: jnp.ndarray,
        step_noises: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size = input_features["ego_state"].shape[0]
        if int(step_noises.shape[0]) != int(n_timesteps):
            raise ValueError("step_noises length must equal n_timesteps")

        cond = self._input_features_projection_impl(input_features, deterministic=deterministic)
        t0 = jnp.full((batch_size,), n_timesteps - 1, dtype=jnp.int32)
        x = self.diffusion.q_sample(jnp.transpose(proposals, (0, 2, 1)), t0, noise=q_noise)

        def body(i: int, x_curr: jnp.ndarray) -> jnp.ndarray:
            timestep = n_timesteps - 1 - i
            t = jnp.full((batch_size,), timestep, dtype=jnp.int32)
            return self.diffusion.p_sample(x_curr, t, cond, step_noises[timestep])

        x = jax.lax.fori_loop(0, n_timesteps, body, x)
        return jnp.transpose(x, (0, 2, 1))

    def forward(self, input_features: Dict[str, jnp.ndarray], rng: jax.Array, deterministic: bool = True) -> jnp.ndarray:
        return self.sample(input_features, rng=rng, deterministic=deterministic)

    def sample(
        self,
        input_features: Dict[str, jnp.ndarray],
        *,
        rng: Optional[jax.Array] = None,
        x_init: Optional[jnp.ndarray] = None,
        step_noises: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        if x_init is not None and step_noises is not None:
            return self._jit_sample_with_fixed_noises(input_features, x_init, step_noises, deterministic)
        if rng is None:
            raise ValueError("Provide either (x_init and step_noises) or rng")
        batch_size = input_features["ego_state"].shape[0]
        key_x, key_steps = jax.random.split(rng)
        x_init = jax.random.normal(key_x, (batch_size, self.target_dim, self.predict_horizon), dtype=jnp.float32)
        step_noises = jax.random.normal(
            key_steps,
            (self.diffusion.timesteps, batch_size, self.target_dim, self.predict_horizon),
            dtype=jnp.float32,
        )
        return self._jit_sample_with_fixed_noises(input_features, x_init, step_noises, deterministic)

    def loss(
        self,
        input_features: Dict[str, jnp.ndarray],
        *,
        rng: Optional[jax.Array] = None,
        noise: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        if noise is not None and t is not None:
            return self._loss_with_noise_t_impl(input_features, noise, t, deterministic)
        if rng is None:
            raise ValueError("Provide either (noise and t) or rng")
        batch_size = input_features["ego_state"].shape[0]
        key_t, key_noise = jax.random.split(rng)
        t = jax.random.randint(key_t, (batch_size,), 0, self.diffusion.timesteps, dtype=jnp.int32)
        noise = jax.random.normal(key_noise, (batch_size, self.target_dim, self.predict_horizon), dtype=jnp.float32)
        return self._jit_loss_with_noise_t(input_features, noise, t, deterministic)

    def resample(
        self,
        input_features: Dict[str, jnp.ndarray],
        proposals: jnp.ndarray,
        n_timesteps: int,
        *,
        rng: Optional[jax.Array] = None,
        q_noise: Optional[jnp.ndarray] = None,
        step_noises: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        if q_noise is not None and step_noises is not None:
            return self._jit_resample_with_fixed_noises(input_features, proposals, n_timesteps, q_noise, step_noises, deterministic)
        if rng is None:
            raise ValueError("Provide either (q_noise and step_noises) or rng")
        batch_size = input_features["ego_state"].shape[0]
        key_q, key_steps = jax.random.split(rng)
        q_noise = jax.random.normal(key_q, (batch_size, self.target_dim, self.predict_horizon), dtype=jnp.float32)
        step_noises = jax.random.normal(
            key_steps,
            (n_timesteps, batch_size, self.target_dim, self.predict_horizon),
            dtype=jnp.float32,
        )
        return self._jit_resample_with_fixed_noises(input_features, proposals, n_timesteps, q_noise, step_noises, deterministic)
