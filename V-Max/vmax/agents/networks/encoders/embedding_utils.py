# Copyright 2025 Valeo.


"""Module containing utility functions for constructing MLP embeddings."""

import jax.numpy as jnp
from flax import linen as nn


def build_goal_token(goal_features, output_size, hidden_sizes, activation_fn):
    """Embed the goal feature block as a single attention token.

    The goal block is one vector per scene (not a set of elements like agents or
    roadgraph points), so it becomes exactly one always-valid token.

    Args:
        goal_features: Goal features of shape [B, goal_size]. A goal_size of 0 means
            goal features are disabled in the observation config.
        output_size: The token dimension (dk).
        hidden_sizes: A sequence of hidden layer sizes.
        activation_fn: Activation function to use between layers.

    Returns:
        A tuple of (encoding [B, 1, dk], mask [B, 1]), or (None, None) when the goal
        block is empty.

    """
    if goal_features.shape[-1] == 0:
        return None, None

    encoding = build_mlp_embedding(goal_features, output_size, hidden_sizes, activation_fn, "goal_enc")
    encoding = jnp.expand_dims(encoding, axis=1)
    mask = jnp.ones(encoding.shape[:-1], dtype=bool)

    return encoding, mask


def build_mlp_embedding(input_features, output_size, hidden_sizes, activation_fn, name_prefix):
    """Build an MLP embedding network.

    Args:
        input_features: The input tensor.
        output_size: The final output size.
        hidden_sizes: A sequence of hidden layer sizes.
        activation_fn: Activation function to use between layers.
        name_prefix: Prefix for naming layers.

    Returns:
        The output tensor after applying the MLP.

    """
    x = input_features
    for i, hidden_size in enumerate(hidden_sizes):
        x = nn.Dense(hidden_size, name=f"{name_prefix}_layer_{i}")(x)
        if activation_fn:
            x = activation_fn(x)

    output = nn.Dense(output_size, name=f"{name_prefix}_output")(x)

    return output
