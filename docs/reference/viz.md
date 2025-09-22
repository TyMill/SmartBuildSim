# Visualisation

`smartbuildsim.viz.plots` turns telemetry and anomaly/clustering outputs into
annotated Matplotlib charts.

## Components

- `PlotConfig` contains figure metadata (`sensor`, `title`, `width`, `height`).
- `plot_time_series(data, config, output, anomalies=None, clusters=None)` filters
  the dataset to the configured sensor, draws the series, overlays anomalies, and
  optionally colour codes cluster assignments.

## CLI usage

```bash
smartbuildsim viz plot examples/configs/default.yaml \
  --anomalies-path outputs/anomalies.csv --clusters-path outputs/clusters.csv
```

The command relies on the outputs produced by the forecasting/anomaly/clustering
commands and saves a PNG to the configured output directory.

## Python example

```python
from pathlib import Path
from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.models.anomaly import AnomalyDetectionConfig, detect_anomalies
from smartbuildsim.scenarios.presets import get_scenario
from smartbuildsim.viz.plots import PlotConfig, plot_time_series

scenario = get_scenario("office-small")
dataset = generate_dataset(
    scenario.building, DataGeneratorConfig(**scenario.data.dict())
)
anomalies = detect_anomalies(
    dataset, AnomalyDetectionConfig(**scenario.anomaly.dict())
).data

plot_config = PlotConfig(sensor=scenario.forecasting.sensor, title="Office Energy")
output_path = Path("examples/outputs/plot.png")
plot_time_series(dataset, plot_config, output_path, anomalies=anomalies)
print(f"Plot saved to {output_path}")
```

This mirrors the final step of `examples/scripts/run_example.py`, completing the
end-to-end workflow.
