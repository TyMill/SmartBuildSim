# Modelling Pipelines

`smartbuildsim.models` offers four focused modelling workflows built on top of
scikit-learn and NumPy utilities:

1. [Forecasting](forecasting.md) — deterministic regression for predicting
   future sensor values.
2. [Anomaly detection](anomaly.md) — IsolationForest-based labelling of
   unexpected readings.
3. [Clustering](clustering.md) — KMeans grouping of zones based on selected
   sensors.
4. [Reinforcement learning](rl.md) — tabular Q-learning for thermostat control.

Each workflow consumes the feature-engineered datasets described in
[Feature Engineering](../features.md) and many are orchestrated together in
`examples/scripts/run_example.py`.

The sections below summarise the shared workflow stages before diving into each
specialised page:

- Load deterministic data via [`smartbuildsim data generate`](../data.md) or the
  Python helpers in `smartbuildsim.data.generator`. W razie potrzeby ustaw
  parametry generatora (np. `anomaly_chance`), aby wymusić sygnał do benchmarków.
- Prepare features as needed using `FeatureConfig` from
  `smartbuildsim.features.engineering`.
- Train, evaluate, and persist artefacts using the relevant model module.
- Compare models za pomocą `smartbuildsim.evaluation.benchmark`, który
  udostępnia wielokrotne losowania, testy istotności oraz analizę wrażliwości na
  skalowanie jednostek (szczegóły w [Benchmarkach](benchmark.md)).
- Visualise outcomes with [`smartbuildsim viz plot`](../viz.md) to confirm the
  effect of the trained models.

Continue to the dedicated pages for configuration details and runnable examples.
