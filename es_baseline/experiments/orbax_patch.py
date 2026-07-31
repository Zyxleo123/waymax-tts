"""Runtime fix for orbax arg-less checkpoint restore, applied WITHOUT editing any
vendored file (keeps VENDOR_MANIFEST byte-identity intact).

The vendored `train_diffusion/utils/checkpoints.py` restores via
`ocp.PyTreeCheckpointer().restore(path)` with no target. orbax 0.11.39 (the
version in the `waymax_rs` env) rejects that when the array metadata carries no
concrete sharding:

    ValueError: sharding passed to deserialization should be specified,
                concrete and an instance of `jax.sharding.Sharding`. Got None

`apply()` monkeypatches `PyTreeCheckpointer.restore` so that, only if the original
arg-less call raises, it retries with restore_args that pin every leaf to a
single-device sharding. This loads the *identical* saved weights (single-device =
replicated), so it is value-preserving; environments where the arg-less restore
already works are untouched (the retry never fires).
"""
from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    import jax
    from jax.sharding import SingleDeviceSharding
    import orbax.checkpoint as ocp

    _orig_restore = ocp.PyTreeCheckpointer.restore

    def _restore(self, directory, *args, **kwargs):
        try:
            return _orig_restore(self, directory, *args, **kwargs)
        except (ValueError, TypeError):
            # only reachable when the arg-less restore lacked a concrete sharding
            meta = self.metadata(directory).item_metadata
            sharding = SingleDeviceSharding(jax.devices()[0])
            sharding_tree = jax.tree_util.tree_map(lambda _: sharding, meta)
            restore_args = ocp.checkpoint_utils.construct_restore_args(
                meta, sharding_tree=sharding_tree
            )
            return _orig_restore(
                self, directory, args=ocp.args.PyTreeRestore(restore_args=restore_args)
            )

    ocp.PyTreeCheckpointer.restore = _restore
    _APPLIED = True
