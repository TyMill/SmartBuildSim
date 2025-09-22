"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from smartbuildsim.data.generator import generate_dataset
from smartbuildsim.scenarios.presets import Scenario, get_scenario


@pytest.fixture(scope="session")
def scenario() -> Scenario:
    return get_scenario("office-small")


@pytest.fixture(scope="session")
def dataset(scenario: Scenario) -> pd.DataFrame:
    data = generate_dataset(scenario.building, scenario.data)
    return data
