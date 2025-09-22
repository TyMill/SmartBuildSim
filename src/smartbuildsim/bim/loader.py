"""Utilities for loading BIM schemas from YAML sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ..utils.helpers import dump_yaml, load_yaml
from .schema import Building


def load_building(path: Path) -> Building:
    """Load a building definition from a YAML document."""

    raw = load_yaml(path)
    return cast(Building, Building.parse_obj(raw))


def build_default_schema() -> dict[str, Any]:
    """Return a default building schema mapping."""

    return {
        "name": "Demo Campus",
        "timezone": "UTC",
        "zones": [
            {
                "name": "Lobby",
                "area_sq_m": 250.0,
                "sensors": [
                    {"name": "lobby_temp", "type": "temperature", "unit": "C"},
                    {"name": "lobby_energy", "type": "energy", "unit": "kWh"},
                ],
            },
            {
                "name": "Office",
                "area_sq_m": 600.0,
                "sensors": [
                    {"name": "office_temp", "type": "temperature", "unit": "C"},
                    {"name": "office_co2", "type": "co2", "unit": "ppm"},
                    {"name": "office_energy", "type": "energy", "unit": "kWh"},
                ],
            },
        ],
    }


def write_default_schema(path: Path) -> Path:
    """Write the default schema to disk and return the path."""

    schema = build_default_schema()
    dump_yaml(schema, path)
    return path


__all__ = ["load_building", "build_default_schema", "write_default_schema"]