"""Utility helpers shared across SmartBuildSim modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from ..config import DeterminismConfig, configure_determinism

ModelT = TypeVar("ModelT", bound=BaseModel)



def ensure_directory(path: Path) -> Path:
    """Ensure that a directory exists and return the path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with path.open("r", encoding="utf8") as handle:
        content = yaml.safe_load(handle)
    if content is None:
        return {}
    if not isinstance(content, dict):  # pragma: no cover - defensive guard
        raise TypeError(f"Expected mapping in YAML file {path}")
    return content


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    """Persist a mapping to YAML format."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _parse_override(raw: str) -> Any:
    """Parse a CLI override value using YAML semantics."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:  # pragma: no cover - error path
        raise ValueError(f"Invalid override value: {raw}") from exc


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dotted-key overrides to a configuration mapping."""

    updated = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be in key=value form")
        key, raw_value = override.split("=", 1)
        target = updated
        keys = key.split(".")
        for partial in keys[:-1]:
            target = target.setdefault(partial, {})
            if not isinstance(target, dict):
                raise ValueError(f"Cannot override non-mapping key '{partial}'")
        target[keys[-1]] = _parse_override(raw_value)
    return updated


def set_random_seed(seed: int) -> np.random.Generator:
    """Seed global RNGs using :mod:`smartbuildsim.config` and return a generator."""

    configure_determinism(DeterminismConfig(seed=seed), force=True)
    return np.random.default_rng(seed)


def model_from_mapping(
    model: type[ModelT], mapping: Mapping[str, object] | None = None
) -> ModelT:
    """Instantiate ``model`` from a mapping, defaulting missing values."""

    payload: dict[str, object] = {}
    if mapping:
        payload = dict(mapping)
    return cast(ModelT, model.parse_obj(payload))

__all__ = [
    "apply_overrides",
    "dump_yaml",
    "ensure_directory",
    "model_from_mapping",
    "load_yaml",
    "set_random_seed",
]