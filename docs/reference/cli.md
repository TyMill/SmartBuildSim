# Command Line Interface

`smartbuildsim.cli.app` assembles the Typer application that orchestrates each
module from configuration files. Install the package (`pip install -e .[dev]`)
to expose the `smartbuildsim` executable.

## Sub-commands

| Command | Description |
| --- | --- |
| `smartbuildsim` | Shows available scenario presets when invoked without a sub-command. |
| `smartbuildsim bim init` | Writes a BIM schema to disk, optionally pulling from a scenario preset. |
| `smartbuildsim data generate` | Generates deterministic telemetry according to the active scenario/config. |
| `smartbuildsim model forecast` | Trains a forecasting model and persists the estimator plus predictions. |
| `smartbuildsim model anomalies` | Runs IsolationForest-based anomaly detection and writes annotated CSVs. |
| `smartbuildsim cluster run` | Performs KMeans clustering over zone-level aggregates. |
| `smartbuildsim rl train` | Trains the Q-learning thermostat policy and saves the Q-table. |
| `smartbuildsim viz plot` | Produces annotated Matplotlib plots with optional anomaly/cluster overlays. |

Each command honours overrides passed via `--override key=value` which map onto
nested YAML keys through `smartbuildsim.utils.helpers.apply_overrides`.

## End-to-end invocation

The following shell snippet reproduces the workflow from
`examples/scripts/run_example.py` using the provided
`examples/configs/default.yaml` configuration:

```bash
smartbuildsim bim init outputs/schema.yaml --scenario office-small
smartbuildsim data generate examples/configs/default.yaml
smartbuildsim model forecast examples/configs/default.yaml
smartbuildsim model anomalies examples/configs/default.yaml
smartbuildsim cluster run examples/configs/default.yaml
smartbuildsim rl train examples/configs/default.yaml
smartbuildsim viz plot examples/configs/default.yaml \
  --anomalies-path outputs/anomalies.csv --clusters-path outputs/clusters.csv
```

Artifacts are written under `outputs/` (configurable via YAML) and can be
analysed further using the Python APIs documented throughout the reference
section.
