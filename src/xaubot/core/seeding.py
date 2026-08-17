"""Deterministic seeding.

Reproducibility is a hard requirement (spec section 31). A result that cannot be
regenerated bit-for-bit cannot be trusted, and single-seed neural-network
results are not results at all -- every reported NN number is the median of
several seeds.

Torch is imported lazily so that the data pipeline (Phases 0-1) runs without a
torch install.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

from xaubot.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SeedState:
    """Record of what was seeded, written into every run artifact."""

    seed: int
    deterministic_torch: bool
    torch_available: bool


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> SeedState:
    """Seed every RNG this process can reach.

    Args:
        seed: Master seed.
        deterministic_torch: If True, force cuDNN determinism and error on
            non-deterministic kernels. Costs some speed; worth it.

    Returns:
        A :class:`SeedState` describing what was actually seeded.
    """
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must fit in uint32, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Seeding the legacy global RNG is intentional and not replaceable by a
    # Generator here: scikit-learn, LightGBM, and SHAP all fall back to
    # numpy's global state when no explicit random_state is threaded through.
    # Code we own uses new_rng() below instead.
    np.random.seed(seed)  # noqa: NPY002

    torch_available = False
    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")
    else:
        torch_available = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)

    logger.debug("Seeded everything with %d (torch=%s)", seed, torch_available)
    return SeedState(seed=seed, deterministic_torch=deterministic_torch, torch_available=torch_available)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initialiser that keeps per-worker RNGs deterministic."""
    base = int(os.environ.get("PYTHONHASHSEED", "0"))
    seed = (base + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - see seed_everything


def new_rng(seed: int) -> np.random.Generator:
    """Return an isolated numpy Generator.

    Prefer this over the global numpy RNG anywhere reproducibility matters --
    global state is shared and therefore order-dependent.
    """
    return np.random.default_rng(seed)
