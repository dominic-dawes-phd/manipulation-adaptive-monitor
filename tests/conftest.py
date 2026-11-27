"""Test-session setup that must happen before any test touches CUDA."""

import os

# cuBLAS reads this once, when it first initializes. Tests that enable deterministic algorithms
# (amon.policy.act.enable_determinism) fail if an earlier test initialized cuBLAS without it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
