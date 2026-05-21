from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.torch_modules.mlp import MLP


class PointNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layer: int = 4,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        end_layer_activation: bool = True,
        dropout_p: Optional[float] = None,
        pool_mode: str = "max",
    ) -> None:
        super().__init__()
        if pool_mode != "max":
            raise ValueError(f"pool_mode {pool_mode} not supported")
        if n_layer < 2:
            raise ValueError("n_layer must be at least 2")

        self.pool_mode = pool_mode
        self.hidden_dim = hidden_dim
        self.input_mlp = MLP(
            [input_dim, hidden_dim, hidden_dim],
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
        )
        self.mlp_layers = nn.ModuleList([
            MLP(
                [hidden_dim, hidden_dim // 2],
                dropout_p=dropout_p,
                use_layernorm=use_layernorm,
                use_batchnorm=use_batchnorm,
            )
            for _ in range(n_layer - 1)
        ])
        self.mlp_out = MLP(
            [hidden_dim, hidden_dim],
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
            end_layer_activation=end_layer_activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Input x should be 3D tensor, got {tuple(x.shape)}")

        batch_size, num_points, _ = x.shape
        if valid is None:
            valid = torch.ones((batch_size, num_points), dtype=torch.bool, device=x.device)

        has_any = valid.any(dim=1)
        neg_inf = torch.full_like(x[..., 0], -1e9)

        h = self.input_mlp(x, deterministic=deterministic)

        for mlp in self.mlp_layers:
            feature_encoded = mlp(h, deterministic=deterministic)
            feature_pooled = torch.where(valid[..., None], feature_encoded, neg_inf[..., None]).amax(dim=1, keepdim=True)
            feature_pooled = torch.where(has_any[:, None, None], feature_pooled, torch.zeros_like(feature_pooled))
            h = torch.cat([feature_encoded, feature_pooled.expand(batch_size, num_points, -1)], dim=-1)

        h = torch.where(valid[..., None], h, neg_inf[..., None]).amax(dim=1, keepdim=False)
        h = torch.where(has_any[:, None], h, torch.zeros_like(h))
        return self.mlp_out(h, deterministic=deterministic)


__all__ = ["PointNet"]
