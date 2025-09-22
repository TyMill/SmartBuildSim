"""Tests for BIM schema and loader."""

from __future__ import annotations

from pathlib import Path

from smartbuildsim.bim.loader import load_building, write_default_schema
from smartbuildsim.bim.schema import Building
from smartbuildsim.scenarios.presets import Scenario


def test_building_schema_roundtrip(scenario: Scenario) -> None:
    building = scenario.building
    assert building.name == "Downtown Office"
    assert len(building.zones) == 2
    assert {sensor.name for zone in building.zones for sensor in zone.sensors}


def test_loader_writes_default(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.yaml"
    write_default_schema(schema_path)
    building = load_building(schema_path)
    assert isinstance(building, Building)
    assert building.zones
