"""Deterministic forecasting pipeline built on scikit-learn."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.engineering import FeatureConfig, build_supervised_matrix


class ForecastingConfig(BaseModel):
    """Configuration for the forecasting workflow."""

    sensor: str
    horizon: int = Field(default=1, gt=0)
    lags: list[int] = Field(default_factory=lambda: [1, 2, 24])
    test_size: float = Field(default=0.2, ge=0.1, le=0.9)

    @property
    def feature_config(self) -> FeatureConfig:
        return FeatureConfig(lags=self.lags)


@dataclass
class ForecastingModel:
    """Trained forecasting model with metadata."""

    pipeline: Pipeline
    sensor: str
    horizon: int
    lags: list[int]

    def forecast(self, history: pd.Series, steps: int) -> np.ndarray:
        """Iteratively forecast ``steps`` values using the trained pipeline."""

        history = history.sort_index()
        ordered_lags = sorted(self.lags)
        if len(history) < max(ordered_lags):
            raise ValueError("Not enough history to compute requested lags")
        values = list(history.iloc[-max(ordered_lags) :])
        forecasts: list[float] = []
        for _ in range(steps):
            features = [values[-lag] for lag in ordered_lags]
            prediction = float(self.pipeline.predict([features])[0])
            forecasts.append(prediction)
            values.append(prediction)
        return np.asarray(forecasts)


@dataclass
class ForecastingResult:
    """Container bundling the trained model and evaluation metrics."""

    model: ForecastingModel
    rmse: float
    predictions: np.ndarray
    targets: np.ndarray


def _prepare_training_frame(data: pd.DataFrame, config: ForecastingConfig) -> pd.DataFrame:
    """Filter the dataset to a single sensor and build the supervised matrix."""
    subset = data[data["sensor"] == config.sensor]
    if subset.empty:
        raise ValueError(f"No observations available for sensor '{config.sensor}'")
    series = subset.sort_values("timestamp").set_index("timestamp")["value"]
    return build_supervised_matrix(series, config.lags, config.horizon)


def train_forecasting_model(
    data: pd.DataFrame, config: ForecastingConfig
) -> ForecastingResult:
    """Train and evaluate a deterministic forecasting model."""

    frame = _prepare_training_frame(data, config)
    features = [f"lag_{lag}" for lag in sorted(config.lags)]
    target_column = f"lead_{config.horizon}"
    split_index = int(len(frame) * (1.0 - config.test_size))
    if split_index <= 0 or split_index >= len(frame):
        minimum_rows = max(2, math.ceil(1.0 / (1.0 - config.test_size)))
        raise ValueError(
            "Supervised training frame is too short to create a train/test split: "
            f"need at least {minimum_rows} rows for test_size={config.test_size:.2f} "
            f"but only {len(frame)} rows are available."
        )
    train = frame.iloc[:split_index]
    test = frame.iloc[split_index:]
    pipeline = Pipeline(
        steps=[("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )
    pipeline.fit(train[features], train[target_column])
    predictions = pipeline.predict(test[features])
    rmse = float(np.sqrt(mean_squared_error(test[target_column], predictions)))
    model = ForecastingModel(
        pipeline=pipeline, sensor=config.sensor, horizon=config.horizon, lags=config.lags
    )
    return ForecastingResult(
        model=model,
        rmse=rmse,
        predictions=np.asarray(predictions),
        targets=test[target_column].to_numpy(),
    )


def persist_model(model: ForecastingModel, path: Path) -> Path:
    """Persist a trained forecasting model to disk."""

    joblib.dump(model, path)
    return path


def load_model(path: Path) -> ForecastingModel:
    """Load a previously persisted forecasting model."""

    loaded = joblib.load(path)
    if not isinstance(loaded, ForecastingModel):  # pragma: no cover - defensive
        raise TypeError("Unexpected model type in persisted artifact")
    return loaded


__all__ = [
    "ForecastingConfig",
    "ForecastingModel",
    "ForecastingResult",
    "load_model",
    "persist_model",
    "train_forecasting_model",
]
