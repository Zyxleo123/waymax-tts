"""Compatibility shims so the vendored V-Max runs on the ``waymax`` conda env.

V-Max targets ``jax[cuda12]<0.6`` but our ``waymax`` env ships ``jax 0.10``,
which removed a handful of APIs. The V-Max source under ``V-Max/`` has been
patched directly, but importing this module *also* re-installs the shims at
runtime so the integration keeps working even if the vendored tree is replaced
by a pristine upstream checkout.

Import side effects only; safe to import multiple times.
"""

from __future__ import annotations

import os

# Keep JAX from grabbing the whole GPU so the Waymax sim + any torch leave room.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax
import jax.numpy as jnp


def _device_put_replicated(tree, devices):
    """Drop-in for the removed ``jax.device_put_replicated``.

    Adds a leading axis of size ``len(devices)`` to every leaf so the result can
    be consumed directly by ``jax.pmap`` (which shards axis 0 across devices).
    """
    n = len(devices)

    def _replicate(x):
        x = jnp.asarray(x)
        return jnp.broadcast_to(x[None], (n,) + x.shape)

    return jax.tree_util.tree_map(_replicate, tree)


if not hasattr(jax, "device_put_replicated"):
    jax.device_put_replicated = _device_put_replicated  # type: ignore[attr-defined]
