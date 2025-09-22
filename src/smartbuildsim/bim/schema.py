"""Pydantic models describing the BIM schema for SmartBuildSim."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, validator


class SensorType(str, Enum):
    """Supported sensor categories."""

    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    ENERGY = "energy"
    CO2 = "co2"


class Sensor(BaseModel):
    """A physical sensor installed within a zone."""

    name: str
    sensor_type: SensorType = Field(alias="type")
    unit: str
    baseline: float = Field(
        0.0, description="Baseline magnitude used for synthetic data generation"
    )


class Zone(BaseModel):
    """Building zone grouping one or more sensors."""

    name: str
    area_sq_m: float = Field(..., gt=0)
    sensors: list[Sensor]

    @validator("sensors")
    def _check_sensors(cls, value: list[Sensor]) -> list[Sensor]:  # noqa: N805
        if not value:
            raise ValueError("Zones must define at least one sensor")
        return value


class Building(BaseModel):
    """Root BIM object describing the simulated building."""

    name: str
    timezone: str = "UTC"
    zones: list[Zone]

    @validator("zones")
    def _check_zones(cls, value: list[Zone]) -> list[Zone]:  # noqa: N805
        if not value:
            raise ValueError("Building must declare at least one zone")
        return value


__all__ = ["SensorType", "Sensor", "Zone", "Building"]