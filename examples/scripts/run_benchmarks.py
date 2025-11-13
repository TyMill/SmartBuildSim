"""Execute comparative benchmarks for forecasting, anomaly detection, and RL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smartbuildsim.config import DeterminismConfig, configure_determinism
from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.evaluation.benchmark import (
    AnomalyBenchmarkResult,
    BenchmarkResultBase,
    RegressionBenchmarkResult,
    SoftQLearningConfig,
    run_anomaly_benchmark,
    run_regression_benchmark,
    run_rl_benchmark,
)
from smartbuildsim.features.engineering import build_supervised_matrix
from smartbuildsim.models.rl import RLConfig
from smartbuildsim.scenarios.presets import get_scenario


def _serialize_benchmark(result: BenchmarkResultBase) -> dict[str, Any]:
    """Convert benchmark result frames into JSON serialisable structures."""

    return {
        "baseline": result["baseline"],
        "metrics": result["metrics"].to_dict(),
        "significance": result["significance"].to_dict(orient="records"),
        "scaling": result["scaling"].to_dict(orient="records"),
    }


def _normalization_impact(
    metric_name: str,
    baseline: str,
    raw_result: BenchmarkResultBase,
    normalized_result: BenchmarkResultBase,
) -> dict[str, float | str]:
    """Summarise the change in the baseline metric after normalization."""

    raw_mean = float(raw_result["metrics"].loc[baseline, "mean"])
    norm_mean = float(normalized_result["metrics"].loc[baseline, "mean"])
    return {
        "metric": metric_name,
        "baseline": baseline,
        "raw_mean": raw_mean,
        "normalized_mean": norm_mean,
        "delta": norm_mean - raw_mean,
    }


def main() -> None:
    output_dir = Path("examples/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_determinism(DeterminismConfig(seed=0), force=True)

    scenario = get_scenario("office-small")
    data_payload = scenario.data.model_dump()
    data_payload.update({"anomaly_chance": 0.02, "anomaly_magnitude": 3.0})
    data_config = DataGeneratorConfig(**data_payload)
    dataset = generate_dataset(
        scenario.building,
        data_config,
        include_anomaly_labels=True,
    )
    dataset.to_csv(output_dir / "benchmark_dataset.csv", index=False)

    sensor = scenario.forecasting.sensor
    sensor_frame = dataset[dataset["sensor"] == sensor].copy()
    sensor_frame.sort_values("timestamp", inplace=True)
    series = sensor_frame.set_index("timestamp")["value"]

    regression_raw: RegressionBenchmarkResult = run_regression_benchmark(
        series,
        lags=scenario.forecasting.lags,
        horizon=scenario.forecasting.horizon,
        seeds=[0, 1, 2],
    )

    regression_normalized: RegressionBenchmarkResult | None = None
    if "value_normalized" in sensor_frame.columns:
        normalized_series = sensor_frame.set_index("timestamp")["value_normalized"]
        regression_normalized = run_regression_benchmark(
            normalized_series,
            lags=scenario.forecasting.lags,
            horizon=scenario.forecasting.horizon,
            seeds=[0, 1, 2],
        )

    anomaly_features = build_supervised_matrix(
        sensor_frame.set_index("timestamp")["value"],
        lags=scenario.forecasting.lags,
        horizon=1,
    )
    labels = sensor_frame.iloc[-len(anomaly_features) :]["is_anomaly"].reset_index(drop=True)
    anomaly_raw: AnomalyBenchmarkResult = run_anomaly_benchmark(
        anomaly_features, labels, seeds=[0, 1, 2]
    )

    anomaly_normalized: AnomalyBenchmarkResult | None = None
    if "value_normalized" in sensor_frame.columns:
        normalized_features = build_supervised_matrix(
            sensor_frame.set_index("timestamp")["value_normalized"],
            lags=scenario.forecasting.lags,
            horizon=1,
        )
        anomaly_normalized = run_anomaly_benchmark(
            normalized_features, labels, seeds=[0, 1, 2]
        )

    rl_base = RLConfig(**scenario.rl.model_dump())
    soft_config = SoftQLearningConfig(**scenario.rl.model_dump(), temperature=0.08)
    rl = run_rl_benchmark(rl_base, soft_config, seeds=[7, 11, 21, 42])

    report = {
        "regression": {
            "raw": _serialize_benchmark(regression_raw),
            **(
                {
                    "normalized": _serialize_benchmark(regression_normalized),
                    "impact": _normalization_impact(
                        "rmse",
                        regression_raw["baseline"],
                        regression_raw,
                        regression_normalized,
                    ),
                }
                if regression_normalized is not None
                else {}
            ),
        },
        "anomaly": {
            "raw": _serialize_benchmark(anomaly_raw),
            **(
                {
                    "normalized": _serialize_benchmark(anomaly_normalized),
                    "impact": _normalization_impact(
                        "f1",
                        anomaly_raw["baseline"],
                        anomaly_raw,
                        anomaly_normalized,
                    ),
                }
                if anomaly_normalized is not None
                else {}
            ),
        },
        "rl": {
            "metrics": rl.metrics.to_dict(),
            "significance": rl.significance.to_dict(orient="records"),
        },
    }

    output_path = output_dir / "benchmark_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Benchmark report written to {output_path}")


if __name__ == "__main__":
    main()
