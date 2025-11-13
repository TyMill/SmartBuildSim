from __future__ import annotations

import random

import numpy as np

from smartbuildsim.config import (
    DeterminismConfig,
    configure_determinism,
    create_rng,
    resolve_seed,
)


def test_configure_sets_python_random() -> None:
    configure_determinism(DeterminismConfig(seed=42), force=True)
    assert random.randint(0, 100) == 81
    configure_determinism(DeterminismConfig(seed=42), force=True)
    assert random.randint(0, 100) == 81


def test_create_rng_is_repeatable() -> None:
    configure_determinism(DeterminismConfig(seed=7), force=True)
    values_first = create_rng("tests.component", explicit=11).normal(size=3)
    configure_determinism(DeterminismConfig(seed=7), force=True)
    values_second = create_rng("tests.component", explicit=11).normal(size=3)
    np.testing.assert_allclose(values_first, values_second)


def test_component_override() -> None:
    config = DeterminismConfig(seed=1, components={"custom": 99})
    configure_determinism(config, force=True)
    derived = resolve_seed("custom")
    fallback = resolve_seed("other", explicit=5)
    assert derived != fallback
