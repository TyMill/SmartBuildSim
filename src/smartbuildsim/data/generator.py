"""Synthetic data generation routines."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, Field, validator

from ..bim.schema import Building, Sensor, SensorType
from ..config import create_rng
from ..utils.helpers import ensure_directory

DEFAULT_DELAYS_MINUTES: dict[SensorType, int] = {
    SensorType.TEMPERATURE: 0,
    SensorType.HUMIDITY: 45,
    SensorType.CO2: 15,
    SensorType.ENERGY: 30,
}


NormalizationStrategy = Literal["none", "standard", "minmax"]


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


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
    nonlinear_scale: float = Field(default=0.15, ge=0.0)
    shared_noise_scale: float = Field(default=0.08, ge=0.0)
    correlation_strength: float = Field(default=0.6, ge=0.0, le=1.0)
    delays_minutes: dict[SensorType, int] = Field(
        default_factory=lambda: dict(DEFAULT_DELAYS_MINUTES)
    )
    anomaly_chance: float = Field(default=0.0025, ge=0.0, le=1.0)
    anomaly_magnitude: float = Field(default=2.5, ge=0.0)
    anomaly_duration_steps: tuple[int, int] = Field(default=(1, 6))
    normalization: NormalizationStrategy = Field(default="none")

    @validator("start")
    def _ensure_timezone(cls, value: datetime) -> datetime:  # noqa: N805
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @validator("anomaly_duration_steps")
    def _check_anomaly_steps(
        cls, value: tuple[int, int]
    ) -> tuple[int, int]:  # noqa: N805
        minimum, maximum = value
        if minimum <= 0:
            raise ValueError("Minimum anomaly duration must be positive")
        if minimum > maximum:
            raise ValueError("Minimum anomaly duration cannot exceed maximum")
        return value

    @validator("delays_minutes", pre=True)
    def _coerce_delay_keys(
        cls, value: Mapping[str, Any] | None
    ) -> dict[SensorType, int]:  # noqa: N805
        if value is None:
            return dict(DEFAULT_DELAYS_MINUTES)
        converted: dict[SensorType, int] = {}
        for key, delay in value.items():
            enum_key = key
            if not isinstance(enum_key, SensorType):
                enum_key = SensorType(str(key))
            converted[enum_key] = int(delay)
        return converted


def _delay_series(series: FloatArray, steps: int) -> FloatArray:
    """Return ``series`` delayed by ``steps`` samples without wrap-around."""

    if steps <= 0:
        return series
    delayed = np.empty_like(series)
    delayed[:steps] = series[0]
    delayed[steps:] = series[:-steps]
    return delayed


def _build_zone_features(
    timestamps: pd.DatetimeIndex,
    config: DataGeneratorConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Construct latent zone-wide drivers used to correlate sensor behaviour."""

    steps = np.arange(len(timestamps))
    minutes = steps * config.freq_minutes
    minutes_per_day = 24 * 60
    day_fraction = (minutes % minutes_per_day) / minutes_per_day
    work_cycle = np.sin(np.pi * day_fraction)
    work_cycle = np.clip(work_cycle, 0.0, None)
    weekend_mask = timestamps.dayofweek >= 5
    occupancy = work_cycle**2
    occupancy[weekend_mask] *= 0.35
    occupancy = 1.0 / (1.0 + np.exp(-8.0 * (occupancy - 0.45)))
    occupancy = np.clip(occupancy, 0.0, 1.0)

    hvac_delay = max(
        0,
        int(
            round(
                config.delays_minutes.get(SensorType.ENERGY, 0) / config.freq_minutes
            )
        ),
    )
    hvac_recovery = np.convolve(occupancy, np.ones(6) / 6.0, mode="same")
    hvac_recovery = _delay_series(hvac_recovery, hvac_delay)

    thermal_mass = np.convolve(occupancy, np.ones(12) / 12.0, mode="same")
    humidity_memory = np.convolve(occupancy, np.ones(18) / 18.0, mode="same")

    co2_delay = max(
        0,
        int(
            round(config.delays_minutes.get(SensorType.CO2, 0) / config.freq_minutes)
        ),
    )
    occupancy_delayed = _delay_series(occupancy, co2_delay)

    shared_noise = rng.normal(scale=config.shared_noise_scale, size=len(timestamps))

    return {
        "occupancy": occupancy,
        "occupancy_delayed": occupancy_delayed,
        "thermal_mass": thermal_mass,
        "hvac_recovery": hvac_recovery,
        "humidity_memory": humidity_memory,
        "shared_noise": shared_noise,
    }


def _apply_anomalies(
    series: FloatArray,
    rng: np.random.Generator,
    config: DataGeneratorConfig,
    mask: BoolArray | None = None,
) -> FloatArray:
    """Inject operational anomalies such as spikes, drops and drifts."""

    if config.anomaly_chance <= 0:
        return series

    result = cast(FloatArray, series.copy())
    std = float(np.std(series)) or 1.0
    min_steps, max_steps = config.anomaly_duration_steps
    upper = max(min_steps, max_steps)
    idx = 0
    while idx < len(result):
        if rng.random() < config.anomaly_chance:
            duration = int(rng.integers(min_steps, upper + 1))
            end = min(len(result), idx + duration)
            anomaly_type = rng.choice(["spike", "drop", "drift"])
            magnitude = config.anomaly_magnitude * std
            if mask is not None:
                mask[idx:end] = True
            if anomaly_type == "spike":
                result[idx:end] += magnitude
            elif anomaly_type == "drop":
                result[idx:end] -= magnitude
            else:  # drift
                direction = rng.choice([-1.0, 1.0])
                drift: FloatArray = np.linspace(
                    0.0, direction * magnitude, end - idx, dtype=float
                )
                result[idx:end] += drift
            idx = end
        else:
            idx += 1
    return result


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
    zone_features: Mapping[str, np.ndarray],
    record_anomalies: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Generate a deterministic but noisy series for a sensor."""

    steps = np.arange(len(timestamps))
    minutes = steps * config.freq_minutes
    days_elapsed = minutes / (24 * 60)
    base = _base_level(sensor, zone_area)
    trend = config.trend_per_day * days_elapsed
    seasonal = config.seasonal_amplitude * np.sin(2 * np.pi * minutes / (24 * 60))
    noise = rng.normal(scale=config.noise_scale * max(base, 1.0), size=len(timestamps))

    occupancy = zone_features["occupancy"]
    occupancy_delayed = zone_features["occupancy_delayed"]
    thermal_mass = zone_features["thermal_mass"]
    hvac_recovery = zone_features["hvac_recovery"]
    humidity_memory = zone_features["humidity_memory"]
    shared_noise = zone_features["shared_noise"]

    correlated_noise = (
        config.correlation_strength * shared_noise * max(base * 0.05, 1.0)
    )

    nonlinear_component = np.zeros_like(noise)
    if sensor.sensor_type is SensorType.TEMPERATURE:
        dynamic = 2.0 * thermal_mass - 1.5 * hvac_recovery + 0.4 * occupancy
        nonlinear_component = config.nonlinear_scale * dynamic
    elif sensor.sensor_type is SensorType.HUMIDITY:
        dynamic = 5.0 * humidity_memory + 0.5 * thermal_mass
        nonlinear_component = config.nonlinear_scale * dynamic
    elif sensor.sensor_type is SensorType.CO2:
        dynamic = 1300.0 * occupancy_delayed + 50.0 * shared_noise
        nonlinear_component = config.nonlinear_scale * dynamic
    elif sensor.sensor_type is SensorType.ENERGY:
        nonlinear_load = occupancy**1.5 + 0.3 * hvac_recovery
        dynamic = base * nonlinear_load
        nonlinear_component = config.nonlinear_scale * dynamic

    scaling = 1.0
    if sensor.sensor_type is SensorType.ENERGY:
        scaling = 1.0 + zone_area / 950.0
    if sensor.sensor_type is SensorType.CO2:
        scaling = 1.0 + zone_area / 2000.0

    series = (
        base
        + trend
        + seasonal
        + noise
        + correlated_noise
        + nonlinear_component
    ) * scaling

    delay_minutes = config.delays_minutes.get(sensor.sensor_type, 0)
    delay_steps = max(0, int(round(delay_minutes / config.freq_minutes)))
    if delay_steps:
        series = _delay_series(series, delay_steps)

    anomaly_mask = np.zeros(len(series), dtype=bool) if record_anomalies else None
    series = _apply_anomalies(series, rng, config, anomaly_mask)

    if sensor.sensor_type is SensorType.HUMIDITY:
        series = np.clip(series, 0.0, 100.0)
    elif sensor.sensor_type is SensorType.CO2:
        series = np.clip(series, 250.0, None)
    elif sensor.sensor_type is SensorType.ENERGY:
        series = np.clip(series, 0.0, None)

    return series, anomaly_mask


def _normalize_dataset(
    dataset: pd.DataFrame, method: NormalizationStrategy
) -> tuple[pd.Series, dict[str, dict[str, float]]]:
    """Return normalized values and statistics grouped by sensor."""

    normalized = np.zeros(len(dataset), dtype=float)
    metadata: dict[str, dict[str, float]] = {}
    for sensor, indices in dataset.groupby("sensor").groups.items():
        values = dataset.loc[indices, "value"].to_numpy(dtype=float)
        if method == "standard":
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if std == 0.0:
                scaled = np.zeros_like(values)
            else:
                scaled = (values - mean) / std
            metadata[str(sensor)] = {"mean": mean, "std": std}
        else:  # "minmax"
            minimum = float(values.min())
            maximum = float(values.max())
            denom = maximum - minimum
            if denom == 0.0:
                scaled = np.zeros_like(values)
            else:
                scaled = (values - minimum) / denom
            metadata[str(sensor)] = {"min": minimum, "max": maximum}
        normalized[indices] = scaled
    return pd.Series(normalized, index=dataset.index, name="value_normalized"), metadata


def generate_dataset(
    building: Building,
    config: DataGeneratorConfig,
    *,
    include_anomaly_labels: bool = False,
) -> pd.DataFrame:
    """Generate a deterministic dataset for the provided building."""

    periods = config.days * (24 * 60 // config.freq_minutes)
    index = pd.date_range(
        config.start,
        periods=periods,
        freq=f"{config.freq_minutes}min",
        tz=config.start.tzinfo,
    )
    records: list[dict[str, Any]] = []
    for zone_index, zone in enumerate(building.zones):
        zone_rng = create_rng(
            "data.generator.zone",
            explicit=config.seed,
            offset=zone_index,
        )
        zone_features = _build_zone_features(index, config, zone_rng)
        for sensor in zone.sensors:
            series, anomaly_mask = _generate_series(
                sensor,
                zone.area_sq_m,
                index,
                config,
                zone_rng,
                zone_features,
                record_anomalies=include_anomaly_labels,
            )
            mask_iter = anomaly_mask if anomaly_mask is not None else np.zeros_like(series, dtype=bool)
            for timestamp, value, is_anomaly in zip(index, series, mask_iter, strict=True):
                records.append(
                    {
                        "timestamp": timestamp,
                        "building": building.name,
                        "zone": zone.name,
                        "sensor": sensor.name,
                        "type": sensor.sensor_type.value,
                        "value": float(value),
                        **({"is_anomaly": bool(is_anomaly)} if include_anomaly_labels else {}),
                    }
                )
    frame = pd.DataFrame.from_records(records)
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if config.normalization != "none":
        normalized, stats = _normalize_dataset(frame, config.normalization)
        frame[normalized.name] = normalized
        frame.attrs["normalization"] = {"method": config.normalization, "stats": stats}
    return frame


def save_dataset(data: pd.DataFrame, path: Path) -> Path:
    """Persist a generated dataset to disk as CSV."""

    ensure_directory(path.parent)
    data.to_csv(path, index=False)
    return path


def generate_and_save(
    building: Building,
    config: DataGeneratorConfig,
    output: Path,
    *,
    include_anomaly_labels: bool = False,
) -> pd.DataFrame:
    """Generate data then persist it to ``output``."""

    dataset = generate_dataset(
        building, config, include_anomaly_labels=include_anomaly_labels
    )
    save_dataset(dataset, output)
    return dataset


__all__ = [
    "DataGeneratorConfig",
    "generate_and_save",
    "generate_dataset",
    "save_dataset",
]
