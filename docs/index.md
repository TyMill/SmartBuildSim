# SmartBuildSim

SmartBuildSim simulates smart-building telemetry so you can explore forecasting,
anomaly detection, clustering, and reinforcement learning workflows with a
single toolkit.

## Quick links

- [Quickstart guide](quickstart.md) — step-by-step CLI walkthrough using the
  `office-small` scenario.
- `examples/scripts/run_example.py` — Python script orchestrating the entire
  workflow programmatically.
- `examples/notebooks/smartbuildsim_workflow.ipynb` — Jupyter notebook variant
  of the scripted workflow for interactive exploration.
- `examples/configs/default.yaml` — configuration used by the CLI examples and
  reference snippets.

## Module reference

Dive into the dedicated reference pages for in-depth documentation and runnable
examples:

- [BIM schema](reference/bim.md)
- [Data generation](reference/data.md)
- [Feature engineering](reference/features.md)
- [Pipeline architektury](reference/pipeline.md)
- Models: [Overview](reference/models/index.md),
  [Forecasting](reference/models/forecasting.md),
  [Anomaly detection](reference/models/anomaly.md),
  [Clustering](reference/models/clustering.md),
  [Reinforcement learning](reference/models/rl.md)
- [Scenarios](reference/scenarios.md)
- [Command line interface](reference/cli.md)
- [Utilities](reference/utils.md)
- [Logging](reference/logging.md)
- [Visualisation](reference/viz.md)

## Typical experiment flow

1. Export or customise a BIM schema using [`smartbuildsim bim init`](reference/bim.md).
2. Generate deterministic telemetry with [`smartbuildsim data generate`](reference/data.md).
3. Engineer features and train models for forecasting, anomaly detection,
   clustering, and reinforcement learning ([models reference](reference/models/index.md)).
4. Visualise outcomes with [`smartbuildsim viz plot`](reference/viz.md) to verify
   anomalies and cluster assignments.
5. Iterate by applying configuration overrides (`--override key=value`) as shown
   in the [CLI reference](reference/cli.md).

The [quickstart](quickstart.md) expands each step and mirrors
`examples/scripts/run_example.py` to ensure the documentation remains runnable
end-to-end.

## Build and preview the documentation

Install the documentation dependencies and serve the site locally before
publishing to GitHub Pages:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Build the static site (the output can be deployed to GitHub Pages):

```bash
mkdocs build --strict
```

The MkDocs configuration (`mkdocs.yml`) targets the `docs/` directory and is
ready for GitHub Pages once you push the generated `site/` directory (or enable
Pages on the repository).
