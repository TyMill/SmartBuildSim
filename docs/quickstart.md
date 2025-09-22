# Quickstart

This guide walks through the typical SmartBuildSim workflow using the bundled
`office-small` scenario.

## 1. Install

```bash
pip install -e .[dev]
```

## 2. Inspect the Scenario

List available presets and export the selected BIM schema:

```bash
smartbuildsim bim init outputs/schema.yaml --scenario office-small
```

The generated YAML includes building zones, areas, and sensor metadata.

## 3. Generate Synthetic Data

```bash
smartbuildsim data generate examples/configs/default.yaml
```

The dataset is written to `outputs/dataset.csv`. All generation steps are
seeded to ensure determinism.

## 4. Train Models

```bash
smartbuildsim model forecast examples/configs/default.yaml
smartbuildsim model anomalies examples/configs/default.yaml
smartbuildsim cluster run examples/configs/default.yaml
smartbuildsim rl train examples/configs/default.yaml
```

Artifacts are produced under `outputs/`: forecasting model (`*.joblib`),
forecast predictions (`forecast_predictions.csv`), anomaly annotations
(`anomalies.csv`), cluster assignments (`clusters.csv`), and RL Q-table
(`rl_q_table.npy`).

## 5. Plot Results

```bash
smartbuildsim viz plot examples/configs/default.yaml \
  --anomalies-path outputs/anomalies.csv --clusters-path outputs/clusters.csv
```

An annotated plot is saved as `outputs/plot_<sensor>.png`.

## 6. Extend via Overrides

Customise runs without editing YAML files:

```bash
smartbuildsim data generate examples/configs/default.yaml \
  --override data.days=5 --override data.seed=2024
```

Overrides apply to nested keys using dotted notation.

## Next steps

Consult the [reference documentation](reference/bim.md) for module-level deep
dives or open `examples/scripts/run_example.py` to explore the entire workflow
as executable Python code.
