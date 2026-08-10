"""Load the pretrained diffusion policy from checkpoint *metadata*.

Phase 1, item 1: "Load architecture from checkpoint metadata, not fresh config
defaults. The config class and argument-parser defaults currently disagree on
fields such as ``inst_dim`` and ``max_range``."

The authoritative contract lives in ``<checkpoint>/metadata.json`` (written by
``train_diffusion/utils/checkpoints.py``). For the base checkpoint
``pretrain_diffusion_without_subgoal_.../latest`` the argparse defaults won:
``inst_dim=768``, ``max_range=30.0``, ``map_attr_dim=25``, ``tl_attr_dim=11`` --
none of which match the ``DiffusionPretrainConfig`` dataclass. We therefore never
touch the config; we read the metadata the checkpoint shipped with.

This wraps ``train_diffusion.utils.infer`` (the canonical inference loader) but
restores the on-disk state *once* and exposes both parameter sets:

* ``params_state``     -- the raw trained parameters. DPPO and every policy-
  gradient method collect rollouts and compute likelihoods with these.
* ``ema_params_state`` -- the EMA parameters. Evaluation (deterministic N=1)
  uses these, matching how the pretrained policy is scored today.

Reconstitute a live ``DiffusionPolicy`` with :meth:`DiffusionCheckpoint.merge`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
from flax import nnx

from data.types import PreprocessConfig
from model.diffusion.diffusion_policy import DiffusionPolicy
from train_diffusion.utils.checkpoints import restore_checkpoint
from train_diffusion.utils.infer import (
    _build_model_from_metadata,
    _build_preprocess_cfg,
    _load_checkpoint_metadata,
)
from train_diffusion.utils.utils import coerce_tree_like


_ORBAX_PATCH_APPLIED = False


def _apply_orbax_sharding_patch() -> None:
    """Make arg-less orbax restore work when the metadata carries no sharding.

    ``restore_checkpoint`` restores via ``PyTreeCheckpointer().restore(path)`` with
    no target. orbax (the ``waymax_rs`` env version) rejects that on some
    topologies (notably CPU) with "sharding passed to deserialization should be
    ... Got None". This monkeypatch retries, only when the original call raises,
    with restore_args pinning every leaf to a single-device (== replicated)
    sharding -- value-preserving, and inert where the arg-less restore already
    works. Mirrors ``es_baseline/experiments/orbax_patch.py`` so this package
    stays self-contained.
    """
    global _ORBAX_PATCH_APPLIED
    if _ORBAX_PATCH_APPLIED:
        return
    from jax.sharding import SingleDeviceSharding
    import orbax.checkpoint as ocp

    _orig_restore = ocp.PyTreeCheckpointer.restore

    def _restore(self, directory, *args, **kwargs):
        # Always pin every leaf to the local default device. The checkpoint ships
        # a sharding file naming the GPU it was saved on (cuda:0); on CPU orbax
        # would happily restore *onto* that nonexistent device WITHOUT raising, and
        # then every downstream op logs "Device cuda:0 was not found" and thrashes.
        # Restoring single-device (== replicated) is value-preserving, so we do it
        # unconditionally on every platform; only fall back to the plain restore if
        # metadata is unavailable.
        if not args and not kwargs:
            try:
                meta = self.metadata(directory).item_metadata
                sharding = SingleDeviceSharding(jax.devices()[0])
                sharding_tree = jax.tree_util.tree_map(lambda _: sharding, meta)
                restore_args = ocp.checkpoint_utils.construct_restore_args(
                    meta, sharding_tree=sharding_tree
                )
                return _orig_restore(
                    self, directory, args=ocp.args.PyTreeRestore(restore_args=restore_args)
                )
            except (ValueError, TypeError, KeyError):
                pass
        return _orig_restore(self, directory, *args, **kwargs)

    ocp.PyTreeCheckpointer.restore = _restore
    _ORBAX_PATCH_APPLIED = True


def _leaf_count(tree: Any) -> int:
    return len(jax.tree_util.tree_leaves(tree))


def _deep_overlay(base: Any, over: Any) -> Any:
    """Return ``base`` with ``over``'s leaves substituted wherever their paths
    exist in ``over``.

    The checkpoint's ``ema_params`` can track a *subset* of the parameters (the
    without-subgoal pretrain did not EMA every module), so it has fewer leaves
    than the full ``params_state``. The EMA of an un-tracked parameter is just
    that parameter's raw value, so we start from the complete raw ``params_state``
    (pure dict) and overlay the EMA values where present. This yields a full,
    mergeable parameter set whether ``ema_params`` is a subset or complete.
    """
    if isinstance(base, dict) and isinstance(over, dict):
        return {
            k: (_deep_overlay(base[k], over[k]) if k in over else base[k])
            for k in base
        }
    if isinstance(base, (list, tuple)) and isinstance(over, (list, tuple)) and len(base) == len(over):
        return type(base)(_deep_overlay(b, o) for b, o in zip(base, over))
    return over  # leaf (or structural mismatch): prefer the EMA value


@dataclass
class DiffusionCheckpoint:
    """A restored diffusion policy plus everything needed to run/train it.

    ``graphdef`` + ``nonparam_state`` + (``params_state`` | ``ema_params_state``)
    reconstitute a :class:`DiffusionPolicy` via :meth:`merge`.
    """

    graphdef: Any
    nonparam_state: Any
    params_state: Any          # raw trained params (collection / likelihoods)
    ema_params_state: Any       # EMA params (evaluation)
    preprocess_cfg: PreprocessConfig
    metadata: dict[str, Any]
    checkpoint_path: str

    @property
    def target_dim(self) -> int:
        return int(self.metadata["target_dim"])

    @property
    def predict_horizon(self) -> int:
        return int(self.metadata["predict_horizon"])

    @property
    def predict_type(self) -> str:
        return str(self.metadata["predict_type"])

    @property
    def model_dt(self) -> float:
        return float(self.metadata["model_dt"])

    def merge(self, *, use_ema: bool = False) -> DiffusionPolicy:
        """Reconstitute a live :class:`DiffusionPolicy`.

        ``use_ema=False`` (default) returns the raw-parameter policy used for
        rollout collection and likelihoods; ``use_ema=True`` returns the EMA
        policy used for deterministic evaluation.
        """
        params = self.ema_params_state if use_ema else self.params_state
        return nnx.merge(self.graphdef, params, self.nonparam_state)


def load_diffusion_checkpoint(
    checkpoint_path: str,
    *,
    metadata_path: str | None = None,
    seed: int = 0,
) -> DiffusionCheckpoint:
    """Restore the pretrained diffusion policy from ``checkpoint_path``.

    ``checkpoint_path`` is an orbax tag directory (e.g. ``.../latest``) that
    contains both the PyTree state and a sibling ``metadata.json``. The
    architecture is built entirely from that metadata.
    """
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    _apply_orbax_sharding_patch()
    metadata = _load_checkpoint_metadata(checkpoint_path, metadata_path)
    preprocess_cfg = _build_preprocess_cfg(metadata)

    # Build a template module purely to obtain the graphdef and the pytree
    # structure of Param vs non-Param state; the template's random init values
    # are immediately overwritten by the restored checkpoint below.
    model = _build_model_from_metadata(metadata, seed)
    graphdef, template_params_state, template_nonparam_state = nnx.split(
        model, nnx.Param, ...
    )

    restored = restore_checkpoint(checkpoint_path)
    for key in ("params_state", "ema_params"):
        if key not in restored:
            raise KeyError(
                f"Checkpoint {checkpoint_path!r} is missing '{key}'. "
                f"Present keys: {sorted(restored.keys())}."
            )

    params_state = coerce_tree_like(template_params_state, restored["params_state"])
    # ema_params may cover only a subset of parameters; overlay onto the full raw
    # params so the EMA policy still has every leaf the graphdef expects.
    ema_full = _deep_overlay(restored["params_state"], restored["ema_params"])
    ema_params_state = coerce_tree_like(template_params_state, ema_full)
    nonparam_state = coerce_tree_like(
        template_nonparam_state,
        restored.get("nonparam_state", template_nonparam_state),
    )

    # The checkpoint was saved on GPU, so its arrays are committed to a device
    # that may not exist here (e.g. cuda:0 during a CPU smoke run), which makes
    # every op on them fail/retry. Pin them to the local default device. On GPU
    # this is a no-op (devices()[0] is the GPU already).
    local_dev = jax.devices()[0]
    _to_local = lambda t: jax.tree_util.tree_map(lambda x: jax.device_put(x, local_dev), t)
    params_state = _to_local(params_state)
    ema_params_state = _to_local(ema_params_state)
    nonparam_state = _to_local(nonparam_state)

    n_raw = _leaf_count(restored["params_state"])
    n_ema = _leaf_count(restored["ema_params"])
    n_tmpl_p = _leaf_count(template_params_state)
    n_tmpl_np = _leaf_count(template_nonparam_state)
    print(
        f"[diffusion_ft.checkpoint] leaves: template_params={n_tmpl_p} "
        f"template_nonparam={n_tmpl_np} restored_params={n_raw} restored_ema={n_ema} "
        f"ema_overlaid={_leaf_count(ema_params_state)}"
    )

    return DiffusionCheckpoint(
        graphdef=graphdef,
        nonparam_state=nonparam_state,
        params_state=params_state,
        ema_params_state=ema_params_state,
        preprocess_cfg=preprocess_cfg,
        metadata=metadata,
        checkpoint_path=str(checkpoint_path),
    )


# The base pretrained checkpoint this framework fine-tunes from. Kept here as a
# single source of truth (see also ``reference_es_checkpoint`` /
# ``reference_diffusion_ckpt_contract`` in project memory).
DEFAULT_CHECKPOINT_PATH = (
    "/zfsauton/scratch/mineuih/waymax_rs/vla/pretrain_diffusion/"
    "pretrain_diffusion_without_subgoal_20260616_013642/latest"
)
