from typing import Dict

import jax
import jax.numpy as jnp
from flax import linen as nn

from .diffusion import GaussianDiffusion, UNet1DConditioned
from .mlp import MLP
from .point_net import PointNet


class StackedGRU(nn.Module):
    hidden_size: int
    num_layers: int = 2

    def setup(self) -> None:
        self.cells = [nn.GRUCell(features=self.hidden_size, name=f"gru_{i}") for i in range(self.num_layers)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, T, H]
        b, t, _ = x.shape
        carry = [jnp.zeros((b, self.hidden_size), dtype=x.dtype) for _ in range(self.num_layers)]
        outputs = []
        for i in range(t):
            h = x[:, i, :]
            next_carry = []
            for li, cell in enumerate(self.cells):
                c, h = cell(carry[li], h)
                next_carry.append(c)
            carry = next_carry
            outputs.append(h)
        return jnp.stack(outputs, axis=1)  # [B, T, H]


class DiffusionPolicy(nn.Module):
    target_dim: int
    hidden_dim: int
    cond_dim: int
    lidar_attr_dim: int
    map_attr_dim: int
    tl_attr_dim: int
    goal_dim: int = 2
    predict_type: str = "v"
    predict_horizon: int = 16
    debug_nan_checks: bool = False

    def setup(self) -> None:
        self.lidar_projection = PointNet(input_dim=self.lidar_attr_dim, hidden_dim=self.hidden_dim, name="lidar")
        self.map_projection = PointNet(input_dim=self.map_attr_dim, hidden_dim=self.hidden_dim, name="map")
        self.tl_projection = PointNet(input_dim=self.tl_attr_dim, hidden_dim=self.hidden_dim, name="tl")
        self.goal_projection = MLP(fc_dims=[self.goal_dim, self.hidden_dim, self.hidden_dim, self.hidden_dim], name="goal")

        self.lidar_history_projection = StackedGRU(hidden_size=self.hidden_dim, num_layers=2)
        self.cond_projection = MLP(fc_dims=[self.hidden_dim * 4, self.hidden_dim, self.cond_dim])

        self.diffusion = GaussianDiffusion(
            denoise_fn=UNet1DConditioned(
                in_ch=self.target_dim,
                base_ch=self.hidden_dim,
                cond_dim=self.cond_dim,
                time_emb_dim=self.hidden_dim,
            ),
            timesteps=50,
            predict_type=self.predict_type,
        )

    def input_features_projection(self, input_features: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        _, history_len, n_points, d_lidar = input_features["lidar_points"].shape
        lidar_features = input_features["lidar_points"].reshape(-1, n_points, d_lidar)
        lidar_valid = input_features["lidar_valid"].reshape(-1, n_points)
        lidar_feature = self.lidar_projection(lidar_features, lidar_valid)
        lidar_feature = lidar_feature.reshape(-1, history_len, lidar_feature.shape[-1])
        lidar_feature = self.lidar_history_projection(lidar_feature)
        lidar_feature = lidar_feature[:, -1, :]

        map_features = self.map_projection(input_features["map_features"], input_features["map_valid"])
        traffic_light_features = self.tl_projection(
            input_features["traffic_light_features"], input_features["traffic_light_valid"]
        )
        goal_features = self.goal_projection(input_features["goal_position"])

        if self.debug_nan_checks:
            debug_vals = [goal_features, lidar_feature, map_features, traffic_light_features]
            _ = [jax.debug.print("NaN detected") for v in debug_vals if bool(jnp.isnan(v).any())]

        cond = jnp.concatenate([lidar_feature, map_features, traffic_light_features, goal_features], axis=-1)
        return self.cond_projection(cond)

    def __call__(self, input_features: Dict[str, jnp.ndarray], *, key: jax.Array) -> jnp.ndarray:
        batch_size = input_features["lidar_points"].shape[0]
        cond = self.input_features_projection(input_features)
        pred = self.diffusion.sample(shape=(batch_size, self.target_dim, self.predict_horizon), cond=cond, key=key)
        return jnp.transpose(pred, (0, 2, 1))

    def loss(self, input_features: Dict[str, jnp.ndarray], *, key: jax.Array) -> jnp.ndarray:
        target = input_features["ego_trajectory"][:, : self.predict_horizon, :]
        cond = self.input_features_projection(input_features)
        return self.diffusion.loss(jnp.transpose(target, (0, 2, 1)), cond=cond, key=key)
