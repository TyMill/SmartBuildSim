# Forecasting

`smartbuildsim.models.forecasting` trains deterministic regression models for a
single sensor using scikit-learn pipelines.

## Configuration

`ForecastingConfig` requires:

- `sensor`: the sensor name to forecast.
- `horizon`: number of steps ahead to predict (`lead_horizon` target column).
- `lags`: list of lag offsets that become features.
- `test_size`: fraction of rows reserved for evaluation.

Access `FeatureConfig` via the `feature_config` property to align lag selection
with [`smartbuildsim.features.engineering.build_supervised_matrix`](../features.md).

## Training workflow

- `_prepare_training_frame(data, config)` filters the dataset to the selected
  sensor and constructs the supervised matrix.
- `train_forecasting_model(data, config)` fits a `Pipeline` with `StandardScaler`
  and `LinearRegression`, returning a `ForecastingResult` with predictions and
  RMSE.
- `ForecastingModel.forecast(history, steps)` generates iterative forecasts using
  the trained pipeline.
- `persist_model(model, path)` and `load_model(path)` save and restore trained
  models via `joblib`.

## CLI usage

```bash
smartbuildsim model forecast examples/configs/default.yaml
```

The command reads `outputs/dataset.csv`, trains using configuration overrides
from the YAML (or the scenario defaults), and writes `forecast_<sensor>.joblib`
and `forecast_predictions.csv`.

## Python example

```python
from pathlib import Path
from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.models.forecasting import (
    ForecastingConfig,
    load_model,
    persist_model,
    train_forecasting_model,
)
from smartbuildsim.scenarios.presets import get_scenario

scenario = get_scenario("office-small")
dataset = generate_dataset(
    scenario.building, DataGeneratorConfig(**scenario.data.dict())
)
config = ForecastingConfig(**scenario.forecasting.dict())
result = train_forecasting_model(dataset, config)

print(f"Validation RMSE: {result.rmse:.3f}")
series = dataset[dataset["sensor"] == config.sensor].set_index("timestamp")["value"]
model_path = Path("examples/outputs/forecast.joblib")
model_path.parent.mkdir(parents=True, exist_ok=True)
persist_model(result.model, model_path)
restored = load_model(model_path)
print(restored.forecast(series, steps=3))
```

This mirrors the forecasting stage in `examples/scripts/run_example.py` and sets
up downstream anomaly, clustering, and RL analyses.
