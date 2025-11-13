# Deterministic execution

SmartBuildSim provides deterministic behaviour across all workflows via the
:mod:`smartbuildsim.config` module.  The module exposes a
`DeterminismConfig` model together with helpers for seeding NumPy, Python's
``random`` module and component specific pseudo random number generators.

## Sources of randomness

The table below summarises where randomness occurs and how it is controlled.

| Area | Source | Seed resolution |
| ---- | ------ | ---------------- |
| Data generation | Latent feature synthesis, sensor specific noise, anomaly injection | `create_rng("data.generator.zone", explicit=config.seed, offset=zone_index)` |
| Benchmarking (regression/anomaly) | Shuffled 5-fold splits | `resolve_seed("evaluation.<task>.kfold", explicit=seed, offset=index)` |
| Reinforcement learning | Q-learning sampling, evaluation rollouts, soft Q-learning temperature exploration | `create_rng("models.rl.q_learning.train", explicit=config.seed)`, `create_rng("models.rl.q_learning.eval", explicit=config.seed, offset=1)`, `create_rng("evaluation.rl.soft.train", explicit=config.seed)` |
| Model training | IsolationForest randomness | `resolve_seed("models.anomaly.isolation_forest", explicit=config.random_state)` |

All helpers ultimately derive from a single `DeterminismConfig` instance which can be
populated from configuration files or instantiated manually.  CLI commands accept an
optional ``determinism`` section, for example:

```yaml
scenario: office-small
determinism:
  seed: 2024
  components:
    models.rl.q_learning.train: 321
```

When supplied, the CLI will call `configure_determinism` with the parsed payload
before any data is generated or models are trained.  In Python code the same effect
can be achieved programmatically:

```python
from smartbuildsim.config import DeterminismConfig, configure_determinism

configure_determinism(DeterminismConfig(seed=7))
```

Specific generators can then be requested using `create_rng` or raw integer seeds can
be obtained via `resolve_seed`.

## Limitations

* Some third-party algorithms only guarantee deterministic behaviour for a subset of
  their implementations.  SmartBuildSim selects deterministic options (e.g.
  ``IsolationForest`` with an explicit ``random_state``), but vendor updates may
  introduce non-deterministic paths.
* ``PYTHONHASHSEED`` is set when available to stabilise hash ordering, but CPython
  reads the value during interpreter start-up.  If the environment variable was
  already defined with a different value, the running interpreter will continue to
  use the pre-existing configuration.
* Determinism assumes that matrix libraries (e.g. BLAS, OpenMP) are configured for
  reproducible execution.  When using multi-threaded builds the exact floating point
  rounding may differ between platforms.

## Reproducibility shortcut

The repository ships with a ``reproduce.sh`` script.  It re-runs the default CLI data
pipeline twice and checks that both outputs match byte-for-byte.  This offers a
quick regression test for determinism in continuous integration or local
development environments.
