"""Core package metadata for SmartBuildSim."""

from importlib import metadata


def get_version() -> str:
    """Return the installed package version."""

    try:
        return metadata.version("smartbuildsim")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback path
        return "0.0.0"


__all__ = ["get_version"]
