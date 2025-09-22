"""End-to-end usage example for SmartBuildSim."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.models.anomaly import AnomalyDetectionConfig, detect_anomalies
from smartbuildsim.models.clustering import ClusteringConfig, cluster_zones
from smartbuildsim.models.forecasting import ForecastingConfig, train_forecasting_model
from smartbuildsim.models.rl import RLConfig, train_policy
from smartbuildsim.scenarios.presets import get_scenario
from smartbuildsim.viz.plots import PlotConfig, plot_time_series


def main() -> None:
    """Run an end-to-end experiment using the ``office-small`` preset."""

    output_dir = Path("examples/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = get_scenario("office-small")
    data_config = DataGeneratorConfig(**scenario.data.dict())
    dataset = generate_dataset(scenario.building, data_config)
    dataset_path = output_dir / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    forecast_result = train_forecasting_model(
        dataset, ForecastingConfig(**scenario.forecasting.dict())
    )
    forecast_summary = {
        "rmse": forecast_result.rmse,
        "predictions": forecast_result.predictions[:5].tolist(),
    }

    anomaly_result = detect_anomalies(
        dataset, AnomalyDetectionConfig(**scenario.anomaly.dict())
    )
    cluster_result = cluster_zones(
        dataset, ClusteringConfig(**scenario.clustering.dict())
    )
    rl_result = train_policy(RLConfig(**scenario.rl.dict()))

    plot_config = PlotConfig(sensor=scenario.forecasting.sensor)
    plot_path = output_dir / "sensor_plot.png"
    plot_time_series(dataset, plot_config, plot_path, anomalies=anomaly_result.data)

    summary = pd.DataFrame(
        {
            "forecast_rmse": [forecast_result.rmse],
            "avg_rl_reward": [rl_result.average_reward()],
            "clusters": [cluster_result.assignments.to_dict(orient="records")],
        }
    )
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Dataset saved to {dataset_path}")
    print(f"Forecast summary: {forecast_summary}")
    print(f"Anomaly output rows: {len(anomaly_result.data)}")
    print(f"Cluster assignments saved with {len(cluster_result.assignments)} entries")
    print(f"RL mean reward: {rl_result.average_reward():.3f}")
    print(f"Plot saved to {plot_path}")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
