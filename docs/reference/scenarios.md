# Scenarios

`smartbuildsim.scenarios` packages buildings and tuned configurations into
reusable presets.

## Scenario model

`smartbuildsim.scenarios.presets.Scenario` is a Pydantic model containing:

- `name`: unique identifier used by the CLI (`office-small`, `campus`, ...).
- `building`: a validated BIM `Building` instance.
- `data`: `DataGeneratorConfig` for deterministic dataset creation.
- `forecasting`, `anomaly`, `clustering`, and `rl`: module-specific configuration
  objects reused across the workflow.

The module also exposes:

- `SCENARIOS`: dictionary mapping scenario names to `Scenario` objects.
- `list_scenarios()` to enumerate available presets.
- `get_scenario(name)` to fetch an individual preset.

## Python example

```python
from smartbuildsim.scenarios.presets import get_scenario, list_scenarios

print("Available presets:", list_scenarios())
scenario = get_scenario("office-small")
print(scenario.building.name)
print("Data days:", scenario.data.days)
print("Forecast sensor:", scenario.forecasting.sensor)
```

Use the returned configuration to orchestrate the workflow exactly as in
`examples/scripts/run_example.py`.

## CLI integration

Every CLI command accepts the `scenario` key in YAML to inherit the preset. The
quickstart configuration in `examples/configs/default.yaml` demonstrates how a
single scenario definition can drive data generation, model training, and
plotting by merely overriding a handful of fields.
