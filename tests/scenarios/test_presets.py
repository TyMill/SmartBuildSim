"""Scenario preset tests."""

from __future__ import annotations

import pytest

from smartbuildsim.scenarios.presets import SCENARIOS, get_scenario, list_scenarios


def test_list_scenarios_matches_registry() -> None:
    names = list_scenarios()
    assert set(names) == set(SCENARIOS)


def test_get_scenario_returns_copy() -> None:
    scenario = get_scenario("office-small")
    assert scenario.name == "office-small"
    assert scenario.building.zones


def test_get_scenario_invalid() -> None:
    with pytest.raises(ValueError):
        get_scenario("missing")
