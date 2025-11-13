"""Pre-defined scenarios for quick experimentation."""

from __future__ import annotations

from pydantic import BaseModel

from ..bim.schema import Building
from ..data.generator import DataGeneratorConfig
from ..models.anomaly import AnomalyDetectionConfig
from ..models.clustering import ClusteringConfig
from ..models.forecasting import ForecastingConfig
from ..models.rl import RLConfig


class Scenario(BaseModel):
    """Aggregated configuration for a scenario."""

    name: str
    building: Building
    data: DataGeneratorConfig
    forecasting: ForecastingConfig
    anomaly: AnomalyDetectionConfig
    clustering: ClusteringConfig
    rl: RLConfig


def _build_office_scenario() -> Scenario:
    """Construct a compact office scenario preset."""
    cluster_sensors = [
        {
            "name": "cluster_energy",
            "type": "energy",
            "unit": "kWh",
            "baseline": 150.0,
        },
        {
            "name": "cluster_co2",
            "type": "co2",
            "unit": "ppm",
            "baseline": 500.0,
        },
    ]
    building = Building.parse_obj(
        {
            "name": "Downtown Office",
            "timezone": "UTC",
            "zones": [
                {
                    "name": "Open Office",
                    "area_sq_m": 550,
                    "sensors": [
                        {"name": "office_temp", "type": "temperature", "unit": "C"},
                        {
                            "name": "office_energy",
                            "type": "energy",
                            "unit": "kWh",
                            "baseline": 170.0,
                        },
                    ]
                    + [dict(sensor) for sensor in cluster_sensors],
                },
                {
                    "name": "Conference",
                    "area_sq_m": 200,
                    "sensors": [
                        {
                            "name": "conference_temp",
                            "type": "temperature",
                            "unit": "C",
                        },
                        {
                            "name": "conference_co2",
                            "type": "co2",
                            "unit": "ppm",
                            "baseline": 520.0,
                        },
                    ]
                    + [dict(sensor) for sensor in cluster_sensors],
                },
            ],
        }
    )
    return Scenario(
        name="office-small",
        building=building,
        data=DataGeneratorConfig(days=14, freq_minutes=60, seed=42, normalization="standard"),
        forecasting=ForecastingConfig(sensor="office_energy", horizon=1, lags=[1, 2, 24]),
        anomaly=AnomalyDetectionConfig(sensor="office_energy", contamination=0.05),
        clustering=ClusteringConfig(sensors=["cluster_energy", "cluster_co2"], n_clusters=2),
        rl=RLConfig(episodes=150, steps_per_episode=48, epsilon=0.15),
    )


def _build_campus_scenario() -> Scenario:
    """Construct a multi-building campus scenario preset."""
    building = Building.parse_obj(
        {
            "name": "University Campus",
            "timezone": "UTC",
            "zones": [
                {
                    "name": "Library",
                    "area_sq_m": 800,
                    "sensors": [
                        {"name": "library_temp", "type": "temperature", "unit": "C"},
                        {
                            "name": "library_energy",
                            "type": "energy",
                            "unit": "kWh",
                            "baseline": 160.0,
                        },
                    ],
                },
                {
                    "name": "Lab",
                    "area_sq_m": 400,
                    "sensors": [
                        {"name": "lab_temp", "type": "temperature", "unit": "C"},
                        {
                            "name": "lab_co2",
                            "type": "co2",
                            "unit": "ppm",
                            "baseline": 480.0,
                        },
                    ],
                },
                {
                    "name": "Dormitory",
                    "area_sq_m": 1200,
                    "sensors": [
                        {
                            "name": "dorm_energy",
                            "type": "energy",
                            "unit": "kWh",
                            "baseline": 240.0,
                        },
                        {"name": "dorm_temp", "type": "temperature", "unit": "C"},
                    ],
                },
            ],
        }
    )
    return Scenario(
        name="campus",
        building=building,
        data=DataGeneratorConfig(days=21, freq_minutes=30, seed=99, normalization="standard"),
        forecasting=ForecastingConfig(sensor="dorm_energy", horizon=2, lags=[1, 2, 3, 24]),
        anomaly=AnomalyDetectionConfig(sensor="lab_co2", contamination=0.04),
        clustering=ClusteringConfig(
            sensors=["library_energy", "dorm_energy", "lab_co2"], n_clusters=3
        ),
        rl=RLConfig(episodes=250, steps_per_episode=96, epsilon=0.1),
    )


SCENARIOS: dict[str, Scenario] = {
    "office-small": _build_office_scenario(),
    "campus": _build_campus_scenario(),
}


def list_scenarios() -> list[str]:
    """Return the available scenario identifiers."""

    return sorted(SCENARIOS)


def get_scenario(name: str) -> Scenario:
    """Return a scenario by name."""

    try:
        return SCENARIOS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        available = ", ".join(list_scenarios())
        raise ValueError(f"Unknown scenario '{name}'. Available: {available}") from exc


__all__ = ["Scenario", "SCENARIOS", "get_scenario", "list_scenarios"]
