"""Visualisation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from smartbuildsim.models.anomaly import AnomalyDetectionConfig, detect_anomalies
from smartbuildsim.scenarios.presets import Scenario
from smartbuildsim.viz.plots import PlotConfig, plot_time_series


def test_plot_creation(tmp_path: Path, dataset: pd.DataFrame, scenario: Scenario) -> None:
    anomalies = detect_anomalies(dataset, AnomalyDetectionConfig(**scenario.anomaly.dict()))
    clusters = pd.DataFrame(
        {"zone": dataset["zone"].unique(), "cluster": range(len(dataset["zone"].unique()))}
    )
    config = PlotConfig(sensor=scenario.forecasting.sensor)
    output = tmp_path / "plot.png"
    plot_path = plot_time_series(
        dataset,
        config,
        output,
        anomalies=anomalies.data,
        clusters=clusters,
    )
    assert plot_path.exists()
