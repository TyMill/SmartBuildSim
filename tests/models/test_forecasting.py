"""Forecasting model tests."""

from __future__ import annotations

import pandas as pd
import pytest

from smartbuildsim.models.forecasting import ForecastingConfig, train_forecasting_model
from smartbuildsim.scenarios.presets import Scenario


def test_forecasting_training(dataset: pd.DataFrame, scenario: Scenario) -> None:
    config = ForecastingConfig(**scenario.forecasting.dict())
    result = train_forecasting_model(dataset, config)
    assert result.predictions.size > 0
    assert result.rmse >= 0
    series = (
        dataset[dataset["sensor"] == config.sensor]
        .sort_values("timestamp")
        .set_index("timestamp")["value"]
    )
    forecast = result.model.forecast(series, steps=3)
    assert forecast.shape == (3,)


def test_forecasting_requires_minimum_samples() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="H")
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor": ["sensor-1"] * len(timestamps),
            "value": [1.0, 2.0, 3.0],
        }
    )
    config = ForecastingConfig(sensor="sensor-1", lags=[1], horizon=1, test_size=0.5)
    with pytest.raises(ValueError) as error:
        train_forecasting_model(data, config)

    message = str(error.value)
    assert "at least 2" in message
    assert "test_size=0.50" in message
