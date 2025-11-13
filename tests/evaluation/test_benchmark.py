"""Tests for the benchmarking utilities."""

from __future__ import annotations

import pandas as pd

from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.evaluation.benchmark import (
    SoftQLearningConfig,
    run_anomaly_benchmark,
    run_regression_benchmark,
    run_rl_benchmark,
)
from smartbuildsim.features.engineering import build_supervised_matrix
from smartbuildsim.models.rl import RLConfig
from smartbuildsim.scenarios.presets import get_scenario


def _sensor_frame() -> tuple[pd.Series, pd.DataFrame]:
    scenario = get_scenario("office-small")
    payload = scenario.data.model_dump()
    payload.update({"anomaly_chance": 0.02, "anomaly_magnitude": 3.0})
    dataset = generate_dataset(
        scenario.building,
        DataGeneratorConfig(**payload),
        include_anomaly_labels=True,
    )
    sensor_frame = dataset[dataset["sensor"] == scenario.forecasting.sensor].copy()
    sensor_frame.sort_values("timestamp", inplace=True)
    series = sensor_frame.set_index("timestamp")["value"]
    return series, sensor_frame


def _features_and_labels(lags: list[int]) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    series, sensor_frame = _sensor_frame()
    features = build_supervised_matrix(series, lags=lags, horizon=1)
    labels = sensor_frame["is_anomaly"].iloc[-len(features) :].reset_index(drop=True)
    return series, features, labels


def test_regression_benchmark_produces_statistics() -> None:
    series, _ = _sensor_frame()
    result = run_regression_benchmark(series, lags=[1, 2], horizon=1, seeds=[0, 1])
    assert "linear+standard" in result["rmse_scores"]
    metrics = result["metrics"]
    assert {"mean", "std"}.issubset(metrics.columns)
    assert not result["significance"].empty
    assert not result["scaling"].empty


def test_anomaly_benchmark_uses_labels() -> None:
    _, features, labels = _features_and_labels([1, 2])
    result = run_anomaly_benchmark(features, labels, seeds=[0, 1])
    assert "isolation_forest+standard" in result["f1_scores"]
    assert set(result.keys()) >= {"metrics", "significance", "scaling"}
    assert result["metrics"].loc["isolation_forest+standard", "mean"] <= 1.0


def test_rl_benchmark_compares_algorithms() -> None:
    scenario = get_scenario("office-small")
    base_payload = scenario.rl.model_dump()
    base_payload.update({"episodes": 10, "steps_per_episode": 12})
    base = RLConfig(**base_payload)
    soft_payload = scenario.rl.model_dump()
    soft_payload.update(
        {"episodes": 10, "steps_per_episode": 12, "temperature": 0.1}
    )
    soft = SoftQLearningConfig(**soft_payload)
    result = run_rl_benchmark(base, soft, seeds=[0, 1, 2])
    assert {"mean", "std"}.issubset(result.metrics.columns)
    assert result.metrics.index.tolist() == ["q_learning", "soft_q"]
    assert not result.significance.empty
