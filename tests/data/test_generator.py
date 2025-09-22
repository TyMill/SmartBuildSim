"""Tests for the synthetic data generator."""

from __future__ import annotations

import pandas as pd

from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.scenarios.presets import Scenario


def test_dataset_shape(dataset: pd.DataFrame, scenario: Scenario) -> None:
    expected_sensors = sum(len(zone.sensors) for zone in scenario.building.zones)
    expected_rows = expected_sensors * len(dataset["timestamp"].unique())
    assert len(dataset) == expected_rows


def test_generation_is_deterministic(scenario: Scenario) -> None:
    config = DataGeneratorConfig(**scenario.data.dict())
    first = generate_dataset(scenario.building, config)
    second = generate_dataset(scenario.building, config)
    pd.testing.assert_frame_equal(first, second)
