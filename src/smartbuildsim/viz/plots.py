"""Matplotlib based visualisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, Field

from ..utils.helpers import ensure_directory


class PlotConfig(BaseModel):
    """Configuration for plotting time series data."""

    sensor: str
    title: str | None = None
    width: float = Field(default=10.0, gt=0)
    height: float = Field(default=5.0, gt=0)



def plot_time_series(
    data: pd.DataFrame,
    config: PlotConfig,
    output: Path,
    anomalies: pd.DataFrame | None = None,
    clusters: pd.DataFrame | None = None,
) -> Path:
    """Plot a time-series with optional anomalies and cluster colour coding."""

    subset = data[data["sensor"] == config.sensor].copy()
    if subset.empty:
        raise ValueError(f"No data for sensor '{config.sensor}'")
    subset.sort_values("timestamp", inplace=True)
    fig, ax = plt.subplots(figsize=(config.width, config.height))
    ax.plot(subset["timestamp"], subset["value"], label=config.sensor, color="C0")
    if anomalies is not None and not anomalies.empty:
        flagged = anomalies[(anomalies["sensor"] == config.sensor) & anomalies["is_anomaly"]]
        if not flagged.empty:
            ax.scatter(
                flagged["timestamp"],
                flagged["value"],
                color="red",
                label="Anomaly",
                zorder=5,
            )
    if clusters is not None and not clusters.empty:
        merged = subset.merge(clusters, on="zone", how="left")
        for cluster_id, group in merged.groupby("cluster"):
            ax.scatter(
                group["timestamp"],
                group["value"],
                label=f"Cluster {cluster_id}",
                alpha=0.4,
            )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")
    ax.set_title(config.title or f"Sensor {config.sensor}")
    ax.legend()
    fig.autofmt_xdate()
    ensure_directory(output.parent)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


__all__ = ["PlotConfig", "plot_time_series"]