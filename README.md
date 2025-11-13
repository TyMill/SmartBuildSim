<!-- Core status -->
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://tymill.github.io/SmartBuildSim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/smartbuildsim)](https://pypi.org/project/smartbuildsim/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17187638.svg)](https://doi.org/10.5281/zenodo.17187638)
<!-- Tech meta -->
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/TyMill/SmartBuildSim/blob/main/pyproject.toml)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-1f5081.svg)](https://mypy.readthedocs.io/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](https://docs.pytest.org/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/smartbuildsim?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/smartbuildsim)

# SmartBuildSim

SmartBuildSim is a deterministic smart-building simulation toolkit. It provides
utilities for loading BIM-style building schemas, generating synthetic sensor
data, engineering features, training forecasting/anomaly/clustering/RL models,
and producing Matplotlib visualisations. A Typer-powered CLI orchestrates the
full workflow using YAML configuration files.

## Features

* **Deterministic synthetic data** – centrally managed via
  `smartbuildsim.config` with repeatable random seeds.
* **Rich modelling suite** – forecasting (linear regression with lagged
  features), anomaly detection (IsolationForest), clustering (k-means) and
  reinforcement learning (tabular Q-learning).
* **Scenario presets** – ready-to-run building layouts with tuned defaults.
* **Matplotlib visualisations** – overlay anomalies and cluster assignments on
  sensor time series plots.
* **Typer CLI** – declarative YAML configs with dotted overrides and sensible
  output management.

## Quickstart

1. Install dependencies (Python 3.10+):

   ```bash
   pip install -e .
   ```

2. Generate a default BIM schema:

   ```bash
   smartbuildsim bim init examples/outputs/schema.yaml --scenario office-small
   ```

3. Create synthetic data, train models, detect anomalies, cluster zones, train
   RL policy, and plot results using the bundled configuration:

   ```bash
   smartbuildsim data generate examples/configs/default.yaml
   smartbuildsim model forecast examples/configs/default.yaml
   smartbuildsim model anomalies examples/configs/default.yaml
   smartbuildsim cluster run examples/configs/default.yaml
   smartbuildsim rl train examples/configs/default.yaml
   smartbuildsim viz plot examples/configs/default.yaml \
     --anomalies-path outputs/anomalies.csv --clusters-path outputs/clusters.csv
   ```

4. Run the end-to-end Python example:

   ```bash
   python examples/scripts/run_example.py
   ```

5. Explore the interactive workflow notebook located at
   [`examples/notebooks/smartbuildsim_workflow.ipynb`](examples/notebooks/smartbuildsim_workflow.ipynb)
   for a cell-by-cell walkthrough of the same pipeline. Launch it with your
   preferred Jupyter interface (e.g. `jupyter lab examples/notebooks`).

## Configuration Overview

Configuration is supplied via YAML documents. The `examples/configs/default.yaml`
file demonstrates the expected structure:

```yaml
scenario: office-small
determinism:
  seed: 123
paths:
  output_dir: ./outputs
  dataset: ./outputs/dataset.csv
data:
  days: 10
  seed: 123
models:
  forecasting:
    horizon: 2
  anomaly:
    contamination: 0.07
cluster:
  sensors:
    - cluster_energy
    - cluster_co2
viz:
  sensor: office_energy
```

Override any configuration entry directly from the CLI using dotted keys:

```bash
smartbuildsim data generate examples/configs/default.yaml \
  --override data.seed=999 --override data.days=5
```

See [Deterministic execution](docs/reference/determinism.md) for component
specific seed overrides that integrate with the CLI.

## Reproducibility

Run `./reproduce.sh` to execute the default CLI pipeline twice and assert that
the generated datasets are identical.  This lightweight check is useful both for
local development and as a CI smoke test.

## Development

Install the project in editable mode and run the quality gates:

```bash
pip install -e .[dev]
ruff check .
mypy .
pytest
```

## Post-generation Checklist

* [x] Deterministic data generation with scenario presets
* [x] Forecasting, anomaly detection, clustering, and RL modules
* [x] Matplotlib-based visualisations
* [x] Typer CLI with overrides and persisted outputs
* [x] Comprehensive tests and CI configuration

## License

This project is licensed under the MIT License.

##

With Passion: dr Tymoteusz Miller
