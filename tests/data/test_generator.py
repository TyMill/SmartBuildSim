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
    config = DataGeneratorConfig(**scenario.data.model_dump())
    first = generate_dataset(scenario.building, config)
    second = generate_dataset(scenario.building, config)
    pd.testing.assert_frame_equal(first, second)


def test_generation_with_anomaly_labels(scenario: Scenario) -> None:
    config = DataGeneratorConfig(**scenario.data.model_dump())
    labelled = generate_dataset(
        scenario.building, config, include_anomaly_labels=True
    )
    assert "is_anomaly" in labelled.columns
    assert labelled["is_anomaly"].dtype == bool
    # Regenerate with the same seed to ensure deterministic anomaly masks
    repeat = generate_dataset(
        scenario.building, config, include_anomaly_labels=True
    )
    pd.testing.assert_series_equal(labelled["is_anomaly"], repeat["is_anomaly"], check_names=False)


def test_generation_applies_normalization(scenario: Scenario) -> None:
    config = DataGeneratorConfig(**scenario.data.model_dump())
    dataset = generate_dataset(scenario.building, config)
    assert "value_normalized" in dataset.columns
    meta = dataset.attrs.get("normalization")
    assert meta is not None
    assert meta["method"] == config.normalization
    sensor_stats = meta["stats"][scenario.forecasting.sensor]
    assert set(sensor_stats.keys()) <= {"mean", "std", "min", "max"}
