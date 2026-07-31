from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import jax
from jax import numpy as jnp


def polygon_corners(trajectory: jax.Array) -> jax.Array:
    '''
    trajectory: [num_scenarios, num_trajectories, num_time_steps, 5 (x, y, heading, length, width)]
    returns: [num_scenarios, num_trajectories, num_time_steps, 4 (corners), 2 (x, y)]
    '''
    x = trajectory[..., 0]
    y = trajectory[..., 1]
    heading = trajectory[..., 2]
    length = trajectory[..., 3]
    width = trajectory[..., 4]

    cos_h = jnp.cos(heading)
    sin_h = jnp.sin(heading)

    dx = length / 2 * cos_h
    dy = length / 2 * sin_h
    wx = width / 2 * sin_h
    wy = width / 2 * cos_h

    front_left_x = x + dx - wx
    front_left_y = y + dy + wy
    front_right_x = x + dx + wx
    front_right_y = y + dy - wy
    rear_right_x = x - dx + wx
    rear_right_y = y - dy - wy
    rear_left_x = x - dx - wx
    rear_left_y = y - dy + wy

    corners_x = jnp.stack([front_left_x, front_right_x, rear_right_x, rear_left_x], axis=-1)
    corners_y = jnp.stack([front_left_y, front_right_y, rear_right_y, rear_left_y], axis=-1)

    return jnp.stack([corners_x, corners_y], axis=-1)


def polygon_axes(polygon_xy: jax.Array) -> jax.Array:
    closed = jnp.concatenate([polygon_xy, polygon_xy[..., :1, :]], axis=-2)
    edges = closed[..., 1:, :] - closed[..., :-1, :]
    axes = jnp.stack([-edges[..., 1], edges[..., 0]], axis=-1)
    norm = jnp.maximum(jnp.linalg.norm(axes, axis=-1, keepdims=True), 1e-6)
    return axes / norm


def project_polygon(polygon_xy: jax.Array, axis_xy: jax.Array) -> tuple[jax.Array, jax.Array]:
    proj = jnp.sum(polygon_xy * axis_xy[..., None, :], axis=-1)
    return jnp.min(proj, axis=-1), jnp.max(proj, axis=-1)


def intersects_convex_polygons(poly_a_xy: jax.Array, poly_b_xy: jax.Array) -> jax.Array:
    axes_a = polygon_axes(poly_a_xy)
    axes_b = polygon_axes(poly_b_xy)
    axes = jnp.concatenate([axes_a, axes_b], axis=0)

    def _axis_overlap(axis_xy: jax.Array) -> jax.Array:
        a_min, a_max = project_polygon(poly_a_xy, axis_xy)
        b_min, b_max = project_polygon(poly_b_xy, axis_xy)
        return (a_max >= b_min) & (b_max >= a_min)

    return jnp.all(jax.vmap(_axis_overlap)(axes))


def pairwise_intersections(ego_polygons_xy: jax.Array, target_polygons_xy: jax.Array) -> jax.Array:
    def _for_ego(ego_poly_xy: jax.Array) -> jax.Array:
        return jax.vmap(
            lambda target_poly_xy: intersects_convex_polygons(ego_poly_xy, target_poly_xy)
        )(target_polygons_xy)

    return jax.vmap(_for_ego)(ego_polygons_xy)

