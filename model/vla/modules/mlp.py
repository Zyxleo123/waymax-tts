from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F



def _get_activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    raise RuntimeError(f"activation {name} not implemented")


class MLP(nn.Module):
    def __init__(
        self,
        fc_dims: Sequence[int],
        dropout_p: Optional[float] = None,
        use_layernorm: bool = True,
        activation: str = "gelu",
        end_layer_activation: bool = True,
        init_weight_norm: bool = False,
        init_bias: Optional[float] = None,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if len(fc_dims) < 2:
            raise ValueError("fc_dims should include input and output dimensions")
        if use_layernorm and use_batchnorm:
            raise ValueError("use_layernorm and use_batchnorm are mutually exclusive")

        self.input_dim = int(fc_dims[0])
        self.output_dim = int(fc_dims[-1])
        self.dropout_p = dropout_p
        self.use_layernorm = use_layernorm
        self.use_batchnorm = use_batchnorm
        self.end_layer_activation = end_layer_activation
        self.init_weight_norm = init_weight_norm
        self.init_bias = init_bias
        self.activation = _get_activation(activation)

        fc_layers: list[nn.Module] = []
        norm_layers: list[nn.Module] = []
        dropout_layers: list[nn.Module] = []

        for i in range(len(fc_dims) - 1):
            in_dim = int(fc_dims[i])
            out_dim = int(fc_dims[i + 1])
            dense = nn.Linear(in_dim, out_dim)

            if init_weight_norm:
                with torch.no_grad():
                    weight = dense.weight.data
                    norm = torch.linalg.norm(weight, dim=1, keepdim=True)
                    dense.weight.data = weight / torch.clamp(norm, min=1e-8)
            if init_bias is not None and i == len(fc_dims) - 2:
                with torch.no_grad():
                    dense.bias.data.fill_(float(init_bias))

            fc_layers.append(dense)

            is_last = i == len(fc_dims) - 2
            if (not is_last) or end_layer_activation:
                if use_layernorm:
                    norm_layers.append(nn.LayerNorm(out_dim))
                elif use_batchnorm:
                    norm_layers.append(nn.BatchNorm1d(out_dim))
                else:
                    norm_layers.append(nn.Identity())

                if dropout_p is not None:
                    dropout_layers.append(nn.Dropout(p=float(dropout_p)))
                else:
                    dropout_layers.append(nn.Identity())

        self.fc_layers = nn.ModuleList(fc_layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.dropout_layers = nn.ModuleList(dropout_layers)

    def _apply_batchnorm(self, norm_layer: nn.BatchNorm1d, h: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return F.batch_norm(
            h,
            norm_layer.running_mean,
            norm_layer.running_var,
            norm_layer.weight,
            norm_layer.bias,
            training=not deterministic,
            momentum=norm_layer.momentum,
            eps=norm_layer.eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        fill_invalid: float = 0.0,
        *,
        deterministic: bool = True,
    ) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        h = x.reshape(-1, x.shape[-1])

        norm_dropout_idx = 0
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            is_last = i == len(self.fc_layers) - 1

            if (not is_last) or self.end_layer_activation:
                norm_layer = self.norm_layers[norm_dropout_idx]
                if self.use_batchnorm:
                    h = self._apply_batchnorm(norm_layer, h, deterministic)
                else:
                    h = norm_layer(h)

                dropout_layer = self.dropout_layers[norm_dropout_idx]
                if isinstance(dropout_layer, nn.Dropout):
                    h = F.dropout(h, p=dropout_layer.p, training=not deterministic)
                else:
                    h = dropout_layer(h)

                if not is_last:
                    h = self.activation(h)
                norm_dropout_idx += 1

        h = h.reshape(*leading_shape, self.output_dim)
        if valid_mask is not None:
            h = torch.where(valid_mask[..., None], h, torch.tensor(fill_invalid, dtype=h.dtype, device=h.device))
        if self.end_layer_activation:
            h = self.activation(h)
        return h


__all__ = ["MLP"]
