# Feature Engineering

`smartbuildsim.features` prepares model-ready data by computing rolling
statistics, derivatives, and supervised learning matrices.

## Configuration

`smartbuildsim.features.engineering.FeatureConfig` exposes the following
options:

- `rolling_window`: window size (in observations) used for moving averages and
  standard deviations.
- `include_derivative`: toggles first-order differences for each sensor.
- `lags`: list of lag offsets used when constructing supervised matrices.

## Core utilities

- `engineer_features(data, config)` sorts each sensor time series and computes
  the configured rolling statistics plus optional derivatives.
- `build_supervised_matrix(series, lags, horizon)` produces a regression-ready
  frame with `lag_*` features and a `lead_*` target column.

## Python example

```python
from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.features.engineering import FeatureConfig, engineer_features
from smartbuildsim.scenarios.presets import get_scenario

scenario = get_scenario("office-small")
config = DataGeneratorConfig(**scenario.data.dict())
dataset = generate_dataset(scenario.building, config)

feature_config = FeatureConfig(rolling_window=12, include_derivative=True)
features = engineer_features(dataset, feature_config)

sensor = scenario.forecasting.sensor
display = features[features["sensor"] == sensor][
    ["timestamp", "value", "rolling_mean", "rolling_std", "derivative"]
].head()
print(display)
```

The feature set aligns with the expectations of the forecasting and anomaly
models documented in the [models section](models/index.md).

## CLI tie-in

Running `smartbuildsim model forecast examples/configs/default.yaml` or
`smartbuildsim model anomalies examples/configs/default.yaml` will internally
construct `FeatureConfig` instances that mirror the configuration above, just as
in `examples/scripts/run_example.py`.
