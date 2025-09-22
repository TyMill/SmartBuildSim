"""Synthetic data generation routines."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from ..bim.schema import Building, Sensor, SensorType
from ..utils.helpers import ensure_directory, set_random_seed


class DataGeneratorConfig(BaseModel):
    """Configuration driving the synthetic time-series generator."""

    start: datetime = Field(
        default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc)
    )
    days: int = Field(default=7, gt=0)
    freq_minutes: int = Field(default=60, gt=0)
    seed: int = 7
    trend_per_day: float = 0.2
    seasonal_amplitude: float = 0.35
    noise_scale: float = Field(default=0.05, ge=0.0)

    @validator("start")
    def _ensure_timezone(cls, value: datetime) -> datetime:  # noqa: N805
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


def _base_level(sensor: Sensor, zone_area: float) -> float:
    """Return an appropriate baseline magnitude for a sensor."""
    base = sensor.baseline
    if base:
        return base
    if sensor.sensor_type is SensorType.TEMPERATURE:
        return 20.0
    if sensor.sensor_type is SensorType.HUMIDITY:
        return 45.0
    if sensor.sensor_type is SensorType.CO2:
        return 450.0
    return zone_area * 0.03


def _generate_series(
    sensor: Sensor,
    zone_area: float,
    timestamps: pd.DatetimeIndex,
    config: DataGeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a deterministic but noisy series for a sensor."""
    steps = np.arange(len(timestamps))
    minutes = steps * config.freq_minutes
    days_elapsed = minutes / (24 * 60)
    base = _base_level(sensor, zone_area)
    trend = config.trend_per_day * days_elapsed
    seasonal = config.seasonal_amplitude * np.sin(2 * np.pi * minutes / (24 * 60))
    noise = rng.normal(scale=config.noise_scale * max(base, 1.0), size=len(timestamps))
    scaling = 1.0
    if sensor.sensor_type is SensorType.ENERGY:
        scaling = 1.0 + zone_area / 1000.0
    if sensor.sensor_type is SensorType.CO2:
        scaling = 1.0 + zone_area / 2000.0
    return (base + trend + seasonal + noise) * scaling


def generate_dataset(building: Building, config: DataGeneratorConfig) -> pd.DataFrame:
    """Generate a deterministic dataset for the provided building."""

    periods = config.days * (24 * 60 // config.freq_minutes)
    index = pd.date_range(
        config.start,
        periods=periods,
        freq=f"{config.freq_minutes}min",
        tz=config.start.tzinfo,
    )
    rng = set_random_seed(config.seed)
    records: list[dict[str, object]] = []
    for zone in building.zones:
        zone_rng = np.random.default_rng(rng.integers(0, 1_000_000_000))
        for sensor in zone.sensors:
            series = _generate_series(sensor, zone.area_sq_m, index, config, zone_rng)
            for timestamp, value in zip(index, series, strict=True):
                records.append(
                    {
                        "timestamp": timestamp,
                        "building": building.name,
                        "zone": zone.name,
                        "sensor": sensor.name,
                        "type": sensor.sensor_type.value,
                        "value": float(value),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def save_dataset(data: pd.DataFrame, path: Path) -> Path:
    """Persist a generated dataset to disk as CSV."""

    ensure_directory(path.parent)
    data.to_csv(path, index=False)
    return path


def generate_and_save(
    building: Building, config: DataGeneratorConfig, output: Path
) -> pd.DataFrame:
    """Generate data then persist it to ``output``."""

    dataset = generate_dataset(building, config)
    save_dataset(dataset, output)
    return dataset


__all__ = [
    "DataGeneratorConfig",
    "generate_and_save",
    "generate_dataset",
    "save_dataset",
]