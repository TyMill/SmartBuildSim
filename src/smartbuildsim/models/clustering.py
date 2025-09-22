"""Clustering utilities for grouping building zones."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusteringConfig(BaseModel):
    """Configuration for clustering."""

    sensors: list[str]
    n_clusters: int = Field(default=3, gt=1)
    random_state: int = 11


@dataclass
class ClusteringResult:
    """Result of a clustering run."""

    assignments: pd.DataFrame
    model: KMeans
    features: pd.DataFrame

    def describe_clusters(self) -> pd.DataFrame:
        """Return cluster centroids as a tidy dataframe."""

        centers = pd.DataFrame(self.model.cluster_centers_, columns=self.features.columns)
        centers.insert(0, "cluster", range(len(centers)))
        return centers


def _aggregate_features(data: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Pivot the dataset to produce per-zone averages for selected sensors."""
    subset = data[data["sensor"].isin(sensors)]
    if subset.empty:
        raise ValueError("No data available for requested sensors")
    pivot = subset.pivot_table(
        index="zone",
        columns="sensor",
        values="value",
        aggfunc="mean",
    )
    missing_sensors = [sensor for sensor in sensors if sensor not in pivot.columns]
    if missing_sensors:
        formatted = ", ".join(sorted(missing_sensors))
        raise ValueError(f"Missing sensors in data: {formatted}")

    features = pivot[sensors]
    missing_zones = features.index[features.isna().any(axis=1)].tolist()
    if missing_zones:
        formatted = ", ".join(sorted(missing_zones))
        raise ValueError(f"Missing sensor readings for zones: {formatted}")

    return features


def cluster_zones(data: pd.DataFrame, config: ClusteringConfig) -> ClusteringResult:
    """Cluster building zones based on selected sensors."""

    features = _aggregate_features(data, config.sensors)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    model = KMeans(n_clusters=config.n_clusters, random_state=config.random_state, n_init=10)
    labels = model.fit_predict(transformed)
    assignments = pd.DataFrame(
        {"zone": features.index.tolist(), "cluster": labels}
    ).sort_values("cluster")
    return ClusteringResult(assignments=assignments, model=model, features=features)


__all__ = ["ClusteringConfig", "ClusteringResult", "cluster_zones"]