"""Clustering tests."""

from __future__ import annotations

import pandas as pd
import pytest

from smartbuildsim.data.generator import generate_dataset
from smartbuildsim.models.clustering import ClusteringConfig, cluster_zones
from smartbuildsim.scenarios.presets import Scenario


def test_cluster_assignments(dataset: pd.DataFrame, scenario: Scenario) -> None:
    config = ClusteringConfig(**scenario.clustering.dict())
    result = cluster_zones(dataset, config)
    assert not result.assignments.empty
    assert set(result.assignments.columns) == {"zone", "cluster"}
    assert 1 <= result.assignments["cluster"].nunique() <= config.n_clusters


def test_cluster_office_small_preset_dataset(scenario: Scenario) -> None:
    dataset = generate_dataset(scenario.building, scenario.data)
    result = cluster_zones(dataset, scenario.clustering)
    expected_zones = {zone.name for zone in scenario.building.zones}
    assert set(result.assignments["zone"]) == expected_zones


def test_cluster_zones_missing_sensor_data_raises() -> None:
    data = pd.DataFrame(
        [
            {"zone": "Zone A", "sensor": "temperature", "value": 20.5},
            {"zone": "Zone A", "sensor": "co2", "value": 400.0},
            {"zone": "Zone B", "sensor": "temperature", "value": 21.0},
        ]
    )
    config = ClusteringConfig(sensors=["temperature", "co2"], n_clusters=2)

    with pytest.raises(ValueError, match="Missing sensor readings for zones: Zone B"):
        cluster_zones(data, config)
