"""Central utilities for deterministic behaviour across the project."""

from __future__ import annotations

import os
import random
import zlib

import numpy as np
from pydantic import BaseModel, Field


class DeterminismConfig(BaseModel):
    """Configuration describing how deterministic execution should behave."""

    seed: int = Field(
        default=0,
        description="Fallback seed used when components do not specify one explicitly.",
    )
    numpy_seed: int | None = Field(
        default=None,
        description="Seed applied to NumPy's global random state; defaults to ``seed``.",
    )
    python_seed: int | None = Field(
        default=None,
        description="Seed applied to Python's :mod:`random`; defaults to ``seed``.",
    )
    components: dict[str, int] = Field(
        default_factory=dict,
        description="Optional component-specific seed overrides.",
    )
    enforce_hash_seed: bool = Field(
        default=True,
        description=(
            "If ``True`` ensure ``PYTHONHASHSEED`` is set so dictionary ordering remains "
            "reproducible in child processes."
        ),
    )

    @property
    def numpy_seed_value(self) -> int:
        """Return the resolved NumPy seed."""

        return self.numpy_seed if self.numpy_seed is not None else self.seed

    @property
    def python_seed_value(self) -> int:
        """Return the resolved Python ``random`` seed."""

        return self.python_seed if self.python_seed is not None else self.seed

    def component_seed(self, name: str, fallback: int | None = None) -> int:
        """Return the configured seed for ``name`` or fall back to defaults."""

        if name in self.components:
            return self.components[name]
        if fallback is not None:
            return fallback
        return self.seed


_GLOBAL_CONFIG: DeterminismConfig | None = None


def configure_determinism(
    config: DeterminismConfig | None = None,
    *,
    force: bool = False,
) -> DeterminismConfig:
    """Configure deterministic behaviour for the current Python process."""

    global _GLOBAL_CONFIG
    resolved = config or DeterminismConfig()
    if _GLOBAL_CONFIG is not None and not force and resolved == _GLOBAL_CONFIG:
        return _GLOBAL_CONFIG

    random.seed(resolved.python_seed_value)
    np.random.seed(resolved.numpy_seed_value)
    if resolved.enforce_hash_seed and "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = str(resolved.python_seed_value)

    _GLOBAL_CONFIG = resolved
    return resolved


def get_config() -> DeterminismConfig:
    """Return the active determinism configuration, initialising defaults if necessary."""

    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        configure_determinism()
    assert _GLOBAL_CONFIG is not None  # for mypy
    return _GLOBAL_CONFIG


def _stable_hash(value: str) -> int:
    """Return a platform-stable 32-bit hash for ``value``."""

    return zlib.crc32(value.encode("utf8")) & 0xFFFFFFFF


def resolve_seed(
    component: str,
    *,
    explicit: int | None = None,
    offset: int = 0,
) -> int:
    """Resolve a deterministic seed for ``component`` using the active configuration."""

    config = get_config()
    base = config.component_seed(component, explicit)
    salt = _stable_hash(f"{component}:{offset}")
    combined = (config.numpy_seed_value + base + salt) & 0xFFFFFFFF
    return int(combined)


def create_rng(
    component: str,
    *,
    explicit: int | None = None,
    offset: int = 0,
) -> np.random.Generator:
    """Return a freshly seeded NumPy generator for ``component``."""

    seed = resolve_seed(component, explicit=explicit, offset=offset)
    return np.random.default_rng(seed)


__all__ = ["DeterminismConfig", "configure_determinism", "create_rng", "get_config", "resolve_seed"]
