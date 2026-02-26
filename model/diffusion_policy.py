from typing import Dict

import jax
import jax.numpy as jnp
from flax import nnx

from .diffusion import GaussianDiffusion, UNet1DConditioned
from .mlp import MLP
from .point_net import PointNet


class GRUCell(nnx.Module):
    def __init__(self, input_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        self.input_size = input_size
        self.hidden_size = hidden_size
        k1 = rngs()
        k2 = rngs()
        k3 = rngs()
        k4 = rngs()
        self.weight_ih = nnx.Param(jax.random.normal(k1, (3 * hidden_size, input_size), dtype=jnp.float32) * 0.02)
        self.weight_hh = nnx.Param(jax.random.normal(k2, (3 * hidden_size, hidden_size), dtype=jnp.float32) * 0.02)
        self.bias_ih = nnx.Param(jax.random.normal(k3, (3 * hidden_size,), dtype=jnp.float32) * 0.02)
        self.bias_hh = nnx.Param(jax.random.normal(k4, (3 * hidden_size,), dtype=jnp.float32) * 0.02)

    def __call__(self, carry: jnp.ndarray, x: jnp.ndarray):
        gi = x @ self.weight_ih.value.T + self.bias_ih.value
        gh = carry @ self.weight_hh.value.T + self.bias_hh.value
        i_r, i_z, i_n = jnp.split(gi, 3, axis=-1)
        h_r, h_z, h_n = jnp.split(gh, 3, axis=-1)
        r = jax.nn.sigmoid(i_r + h_r)
        z = jax.nn.sigmoid(i_z + h_z)
        n = jnp.tanh(i_n + r * h_n)
        h_new = (1.0 - z) * n + z * carry
        return h_new, h_new


class StackedGRU(nnx.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2, *, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = [GRUCell(hidden_size, hidden_size, rngs=rngs) for _ in range(num_layers)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, T, H]
        b, t, _ = x.shape
        carry = [jnp.zeros((b, self.hidden_size), dtype=x.dtype) for _ in self.cells]
        outputs = []
        for i in range(t):
            h = x[:, i, :]
            next_carry = []
            for li, cell in enumerate(self.cells):
                c, h = cell(carry[li], h)
                next_carry.append(c)
            carry = next_carry
            outputs.append(h)
        return jnp.stack(outputs, axis=1)


class DiffusionPolicy(nnx.Module):
    def __init__(
        self,
        target_dim: int,
        hidden_dim: int,
        cond_dim: int,
        lidar_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        goal_dim: int = 2,
        predict_type: str = "v",
        predict_horizon: int = 16,
        debug_nan_checks: bool = False,
        compute_dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.target_dim = target_dim
        self.predict_horizon = predict_horizon
        self.debug_nan_checks = debug_nan_checks
        self.compute_dtype = compute_dtype

        self.lidar_projection = PointNet(input_dim=lidar_attr_dim, hidden_dim=hidden_dim, rngs=rngs)
        self.map_projection = PointNet(input_dim=map_attr_dim, hidden_dim=hidden_dim, rngs=rngs)
        self.tl_projection = PointNet(input_dim=tl_attr_dim, hidden_dim=hidden_dim, rngs=rngs)
        self.goal_projection = MLP(fc_dims=[goal_dim, hidden_dim, hidden_dim, hidden_dim], rngs=rngs)

        self.lidar_history_projection = StackedGRU(hidden_size=hidden_dim, num_layers=2, rngs=rngs)
        self.cond_projection = MLP(fc_dims=[hidden_dim * 4, hidden_dim, cond_dim], rngs=rngs)

        self.diffusion = GaussianDiffusion(
            denoise_fn=UNet1DConditioned(
                in_ch=target_dim,
                base_ch=hidden_dim,
                cond_dim=cond_dim,
                time_emb_dim=hidden_dim,
                compute_dtype=compute_dtype,
                rngs=rngs,
            ),
            timesteps=100,
            predict_type=predict_type,
        )
        self._forward_jit_fn = nnx.jit(lambda m, feats, k: m(feats, key=k))
        self._loss_jit_fn = nnx.jit(lambda m, feats, k: m.loss(feats, key=k))

    def _cast_features(self, input_features: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        out = {}
        for k, v in input_features.items():
            if jnp.issubdtype(v.dtype, jnp.bool_):
                out[k] = v
            else:
                out[k] = v.astype(self.compute_dtype)
        return out

    def input_features_projection(self, input_features: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        input_features = self._cast_features(input_features)
        _, history_len, n_points, d_lidar = input_features["lidar_points"].shape
        lidar_features = input_features["lidar_points"].reshape(-1, n_points, d_lidar)
        lidar_valid = input_features["lidar_valid"].reshape(-1, n_points)
        lidar_feature = self.lidar_projection(lidar_features, lidar_valid)
        lidar_feature = lidar_feature.reshape(-1, history_len, lidar_feature.shape[-1])
        lidar_feature = self.lidar_history_projection(lidar_feature)
        lidar_feature = lidar_feature[:, -1, :]

        map_features = self.map_projection(input_features["map_features"], input_features["map_valid"])
        traffic_light_features = self.tl_projection(input_features["traffic_light_features"], input_features["traffic_light_valid"])
        goal_features = self.goal_projection(input_features["goal_position"])

        cond = jnp.concatenate([lidar_feature, map_features, traffic_light_features, goal_features], axis=-1)
        return self.cond_projection(cond)

    def __call__(self, input_features: Dict[str, jnp.ndarray], *, key: jax.Array) -> jnp.ndarray:
        batch_size = input_features["lidar_points"].shape[0]
        cond = self.input_features_projection(input_features)
        pred = self.diffusion.sample(shape=(batch_size, self.target_dim, self.predict_horizon), cond=cond, key=key)
        return jnp.transpose(pred, (0, 2, 1))

    def forward_jit(self, input_features: Dict[str, jnp.ndarray], *, key: jax.Array) -> jnp.ndarray:
        return self._forward_jit_fn(self, input_features, key)

    def loss(self, input_features: Dict[str, jnp.ndarray], *, key: jax.Array) -> jnp.ndarray:
        input_features = self._cast_features(input_features)
        target = input_features["ego_trajectory"][:, : self.predict_horizon, :]
        cond = self.input_features_projection(input_features)
        return self.diffusion.loss(jnp.transpose(target, (0, 2, 1)), cond=cond, key=key)

    def loss_jit(self, input_features: Dict[str, jnp.ndarray], *, key: jax.Array) -> jnp.ndarray:
        return self._loss_jit_fn(self, input_features, key)
