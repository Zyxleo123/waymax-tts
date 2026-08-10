"""Unified training entrypoint for diffusion simulator-reward fine-tuning.

    python -m rl.diffusion_ft.train --method dppo --indices 0,1,2,3,4,5,6,7

Methods:
* ``dppo``  -- direct policy gradient (DPPO) on the late denoising steps
  (:mod:`rl.diffusion_ft.train_dppo`). ``--expert_weight > 0`` adds the expert
  anchor (DPPO + expert anchor). Implemented.

Planned (Phase 3, land incrementally behind this same CLI): ``filtered``, ``drwr``,
``dawr``, ``preference``. Each reuses the shared env/reward/actor here.
"""

from __future__ import annotations

import sys

from rl.diffusion_ft import train_dppo

_METHODS = {"dppo"}


def main():
    argv = sys.argv[1:]
    method = "dppo"
    if "--method" in argv:
        i = argv.index("--method")
        method = argv[i + 1]
        del argv[i : i + 2]
    if method not in _METHODS:
        raise SystemExit(
            f"Unknown/not-yet-implemented method {method!r}. Available: {sorted(_METHODS)}."
        )
    args = train_dppo.build_argparser().parse_args(argv)
    if args.max_replans <= 0:
        args.max_replans = None
    train_dppo.DPPOTrainer(args).train()


if __name__ == "__main__":
    main()
