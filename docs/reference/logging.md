# Logging

`smartbuildsim.logging.config` defines the reusable logging configuration for
both CLI runs and Python scripts.

## Components

- `DEFAULT_LOGGING_CONFIG` is a `dict` compatible with `logging.config.dictConfig`
  that sets up a console handler with timestamped log messages.
- `setup_logging(level="INFO", config=None)` merges custom settings with the
  defaults and applies the configuration.
- `get_logger(name)` returns a logger configured via `setup_logging`.

## Python example

```python
from smartbuildsim.logging.config import DEFAULT_LOGGING_CONFIG, get_logger, setup_logging

custom = DEFAULT_LOGGING_CONFIG.copy()
custom["handlers"]["console"]["level"] = "DEBUG"
setup_logging(level="DEBUG", config=custom)
logger = get_logger("smartbuildsim.example")
logger.info("Information message")
logger.debug("Verbose diagnostics enabled")
```

The CLI (`smartbuildsim.cli.app`) calls `setup_logging` before generating outputs
so that commands such as `smartbuildsim data generate examples/configs/default.yaml`
provide concise progress updates.
