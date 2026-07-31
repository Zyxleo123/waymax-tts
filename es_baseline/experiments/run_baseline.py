"""Patched entrypoint for the vendored ES baseline (`simulation.run_simulation`).

Identical to running `python simulation/run_simulation.py ...` except it applies
the orbax restore fix first (see experiments/orbax_patch.py). No vendored file is
modified. Accepts exactly the same CLI args as run_simulation.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import orbax_patch
orbax_patch.apply()

from simulation.run_simulation import main, _parse_args

if __name__ == "__main__":
    main(_parse_args())
