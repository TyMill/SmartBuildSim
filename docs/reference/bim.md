# Building Information Models

The `smartbuildsim.bim` package defines the building information model (BIM)
that powers every simulation. BIM documents describe buildings, zones, and
sensors and can be loaded from YAML or built from presets.

## Key models

- `smartbuildsim.bim.schema.SensorType` enumerates supported sensor categories
  such as temperature, energy, humidity, and CO₂.
- `smartbuildsim.bim.schema.Sensor` captures metadata (name, type, unit, and an
  optional baseline) for a single device.
- `smartbuildsim.bim.schema.Zone` groups one or more sensors together with the
  zone area and validation that at least one sensor exists.
- `smartbuildsim.bim.schema.Building` is the root object that aggregates zones
  and enforces that a building has at least one zone.

## Loading and writing schemas

`smartbuildsim.bim.loader` provides helpers for turning YAML into strongly
validated BIM objects:

- `load_building(path)` parses a YAML document into a `Building` instance.
- `build_default_schema()` returns an in-memory dictionary you can persist or
  tweak before serialising.
- `write_default_schema(path)` writes the default schema to disk and returns the
  path for convenience.

### CLI usage

The Typer CLI exposes these helpers via the `bim init` sub-command. Using the
provided `examples/configs/default.yaml` configuration, you can export either a
preset or the default schema:

```bash
smartbuildsim bim init outputs/schema.yaml --scenario office-small
```

This mirrors the first step in `examples/scripts/run_example.py` where the
`office-small` scenario is loaded and reused across modules.

### Python example

```python
from pathlib import Path
from smartbuildsim.bim.loader import load_building, write_default_schema

schema_path = Path("examples/outputs/schema.yaml")
schema_path.parent.mkdir(parents=True, exist_ok=True)
write_default_schema(schema_path)
building = load_building(schema_path)

print(building.name)
for zone in building.zones:
    sensor_names = [sensor.name for sensor in zone.sensors]
    print(f"{zone.name}: {sensor_names}")
```

The resulting `Building` instance can be passed directly to the
[data generation pipeline](data.md) or to a preset scenario described in
[scenarios](scenarios.md).
