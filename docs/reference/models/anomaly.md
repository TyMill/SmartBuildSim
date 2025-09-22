# Anomaly Detection

`smartbuildsim.models.anomaly` labels anomalous sensor readings with
IsolationForest.

## Configuration

`AnomalyDetectionConfig` fields:

- `sensor`: sensor name to monitor.
- `contamination`: expected proportion of anomalies (passed to
  `IsolationForest`).
- `random_state`: ensures deterministic model output.
- `rolling_window`: forwarded to `FeatureConfig` to control rolling statistics.

Access the derived `FeatureConfig` via the `feature_config` property when calling
`smartbuildsim.features.engineering.engineer_features` directly.

## Detection workflow

`detect_anomalies(data, config)` performs the following steps:

1. Filter rows for the configured sensor.
2. Engineer rolling mean, standard deviation, and optional derivative features.
3. Train an `IsolationForest` with the engineered features.
4. Append `anomaly_score` and `is_anomaly` columns to the returned dataframe.

## CLI usage

```bash
smartbuildsim model anomalies examples/configs/default.yaml
```

The command writes `outputs/anomalies.csv`, which is then consumed by
`smartbuildsim viz plot` for annotated visualisations.

## Python example

```python
from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.models.anomaly import (
    AnomalyDetectionConfig,
    detect_anomalies,
)
from smartbuildsim.scenarios.presets import get_scenario

scenario = get_scenario("office-small")
dataset = generate_dataset(
    scenario.building, DataGeneratorConfig(**scenario.data.dict())
)
config = AnomalyDetectionConfig(**scenario.anomaly.dict())
result = detect_anomalies(dataset, config)

flagged = result.data[result.data["is_anomaly"]]
print(flagged[["timestamp", "value", "anomaly_score"]].head())
```

See `examples/scripts/run_example.py` for the full pipeline that chains anomaly
labelling with clustering and plotting.
