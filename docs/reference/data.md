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
- `seed`: NumPy seed for reproducible noise.
- `trend_per_day`, `seasonal_amplitude`, and `noise_scale`: shape the overall
  trajectory and stochastic component.

## Core functions

- `generate_dataset(building, config)` returns a tidy `pandas.DataFrame` with
  timestamp, building, zone, sensor, and value columns.
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
from smartbuildsim.scenarios.presets import get_scenario

scenario = get_scenario("office-small")
config = DataGeneratorConfig(**scenario.data.dict())
dataset = generate_dataset(scenario.building, config)

print(dataset.head())
print(f"Rows generated: {len(dataset)}")
```

The resulting dataset flows into [feature engineering](features.md) and the
[modelling pipelines](models/index.md).
