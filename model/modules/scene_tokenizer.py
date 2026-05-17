from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx, struct

from model.modules.mlp import MLP
from model.modules.point_net import PointNet
from model.modules.attention import CrossAttentionLayers


class SceneTokenizer(nnx.Module):
    def __init__(
        self,
        ego_dim: int = 5,
        other_dim: int = 15,
        map_attr_dim: int = 25,
        tl_attr_dim: int = 9,
        hidden_dim: int = 1024,
        cond_dim: int = 256,
        token_num: int = 32,
        rngs: nnx.Rngs = None,
    ) -> None:
        super().__init__()
        # `token_dim` is the output embedding size for scene tokens (cond_dim)
        self.token_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_other_agents = 128
        self.other_pos_embedding = nnx.Param(jnp.zeros((1, self.max_other_agents, self.token_dim), dtype=jnp.float32))
        
        self.ego_tokenizer = MLP([ego_dim, hidden_dim, hidden_dim, cond_dim], rngs=rngs)
        self.goal_tokenizer = MLP([3, hidden_dim, hidden_dim, cond_dim], rngs=rngs)
        self.other_tokenizer = MLP([other_dim, hidden_dim, hidden_dim, cond_dim], rngs=rngs)
        self.tl_tokenizer = MLP([tl_attr_dim, hidden_dim, hidden_dim, cond_dim], rngs=rngs)
        self.map_tokenizer = PointNet(map_attr_dim, cond_dim, rngs=rngs)
        self.cond_tokens = nnx.Param(jnp.zeros((1, token_num, cond_dim), dtype=jnp.float32))
        self.attention = CrossAttentionLayers(cond_dim, cond_dim, num_heads=4, num_layers=4, rngs=rngs)

    def _tokenize_scene_features(
        self,
        input_features: dict[str, jnp.ndarray],
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        ego_state = input_features["ego_state"]
        goal_xy = input_features["goal_xy"]
        remaining_timesteps = input_features["remaining_timesteps"]
        other_states = input_features["other_states"]
        other_valid = input_features["other_valid"]
        map_features = input_features["map_features"]
        map_valid = input_features["map_valid"]
        traffic_light_features = input_features["traffic_light_features"]
        traffic_light_valid = input_features["traffic_light_valid"]

        goal_input = jnp.concatenate([goal_xy, remaining_timesteps], axis=-1)

        batch_size, num_others, other_dim = other_states.shape
        _, num_map_segments, num_points_per_segment, map_dim = map_features.shape
        _, num_tl, tl_dim = traffic_light_features.shape

        ego_token = self.ego_tokenizer(ego_state, deterministic=deterministic).reshape(batch_size, 1, self.token_dim)
        goal_token = self.goal_tokenizer(goal_input, deterministic=deterministic).reshape(batch_size, 1, self.token_dim)

        other_token = self.other_tokenizer(
            other_states.reshape(-1, other_dim),
            deterministic=deterministic,
        ).reshape(batch_size, num_others, self.token_dim)
        other_token = other_token + self.other_pos_embedding.value[:, :num_others, :]

        map_token = self.map_tokenizer(
            map_features.reshape(-1, num_points_per_segment, map_dim),
            map_valid.reshape(-1, num_points_per_segment),
            deterministic=deterministic,
        ).reshape(batch_size, num_map_segments, self.token_dim)

        tl_token = self.tl_tokenizer(
            traffic_light_features.reshape(-1, tl_dim),
            deterministic=deterministic,
        ).reshape(batch_size, num_tl, self.token_dim)

        scene_tokens = jnp.concatenate([ego_token, goal_token, other_token, map_token, tl_token], axis=1)
        scene_valid = jnp.concatenate(
            [
                jnp.ones((batch_size, 2), dtype=bool),
                other_valid,
                jnp.any(map_valid, axis=2),
                traffic_light_valid,
            ],
            axis=1,
        )

        scene_tokens = self.attention(
            query_btc=self.cond_tokens,
            context_bnc=scene_tokens,
            context_mask_bn=scene_valid,
        )

        return scene_tokens

    def __call__(self, input_features: dict[str, jnp.ndarray], *, deterministic: bool = True) -> jnp.ndarray:
        scene_tokens = self._tokenize_scene_features(input_features, deterministic=deterministic)
        return scene_tokens