# Utilities

`smartbuildsim.utils.helpers` contains shared helpers for configuration
management, randomness, and filesystem preparation.

## Key functions

- `ensure_directory(path)` creates directories as needed (used throughout the
  CLI when writing outputs).
- `load_yaml(path)` and `dump_yaml(data, path)` serialise configuration files.
- `apply_overrides(config, overrides)` applies dotted `key=value` overrides from
  CLI arguments using YAML parsing semantics.
- `set_random_seed(seed)` initialises both `numpy.random.default_rng` and the
  legacy global seed for deterministic behaviour.
- `model_from_mapping(model, mapping)` instantiates a Pydantic model from a
  (possibly partial) mapping, falling back to defaults.

## Python example

```python
from pathlib import Path
from smartbuildsim.data.generator import DataGeneratorConfig
from smartbuildsim.utils.helpers import (
    apply_overrides,
    ensure_directory,
    load_yaml,
    model_from_mapping,
)

config_path = Path("examples/configs/default.yaml")
config = load_yaml(config_path)
overrides = ["data.days=5", "models.forecasting.horizon=3"]
customised = apply_overrides(config, overrides)

data_config = model_from_mapping(DataGeneratorConfig, customised.get("data"))
print(data_config)

output_dir = Path(customised.get("paths", {}).get("output_dir", "outputs"))
ensure_directory(output_dir)
print(f"Outputs will be written to {output_dir.resolve()}")
```

These helpers power the CLI flows described in [Command Line Interface](cli.md)
and the example script at `examples/scripts/run_example.py`.
