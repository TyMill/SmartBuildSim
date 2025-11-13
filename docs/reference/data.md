# Data Generation

`smartbuildsim.data` produces deterministic, reproducible telemetry once a
`Building` description is available.

## Configuration

`smartbuildsim.data.generator.DataGeneratorConfig` controls the synthetic time
series. Important fields include:

- `start`: timezone-aware `datetime` for the first observation (defaults to
  1 January 2024 UTC).
- `days`: number of days to simulate.
- `freq_minutes`: sampling cadence in minutes.
- `seed`: seed for reproducible noise resolved through
  [`smartbuildsim.config`](determinism.md).
- `trend_per_day`, `seasonal_amplitude`, and `noise_scale`: shape the overall
  trajectory and stochastic component.
- `nonlinear_scale`: controls how strongly occupancy dynamics bend the curves
  for temperature, humidity, and energy readings.
- `shared_noise_scale` and `correlation_strength`: inject correlated latent
  drivers that keep sensors within the same zone synchronised.
- `delays_minutes`: per-sensor delays modelling transport and control lags.
- `anomaly_chance`, `anomaly_magnitude`, and `anomaly_duration_steps`: configure
  the frequency and severity of injected operational anomalies.
- `normalization`: optional unit harmonisation (`"none"`, `"standard"`,
  `"minmax"`). When enabled, the generator appends a `value_normalized` column
  alongside metadata describing the scaling statistics per sensor.

## Core functions

- `generate_dataset(building, config)` returns a tidy `pandas.DataFrame` with
  timestamp, building, zone, sensor, and value columns, including nonlinear
  behaviour, correlated noise, and injected anomalies.
- `save_dataset(data, path)` persists a CSV file.
- `generate_and_save(building, config, output)` is a convenience wrapper
  returning the dataset after writing it to disk.

## CLI usage

The `data generate` command combines BIM input, generator configuration, and
output handling. With the example configuration you can run:

```bash
smartbuildsim data generate examples/configs/default.yaml
```

This command is the second step of `examples/scripts/run_example.py` and writes
`outputs/dataset.csv` in the project root.

## Python example

```python
from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.data.validation import compare_datasets
from smartbuildsim.scenarios.presets import get_scenario
import pandas as pd

scenario = get_scenario("office-small")
config = DataGeneratorConfig(**scenario.data.model_dump())
dataset = generate_dataset(scenario.building, config)
print(dataset.attrs.get("normalization"))

reference = pd.read_csv("docs/reference/datasets/ashrae_sample.csv")
report = compare_datasets(dataset, reference, sensor_mapping={"office_energy": "meter_0_energy"})

print(report.notes)
```

The resulting dataset flows into [feature engineering](features.md) and the
[modelling pipelines](models/index.md).
