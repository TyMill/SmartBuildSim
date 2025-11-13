"""Anomaly detection utilities built on IsolationForest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest

from ..config import resolve_seed
from ..features.engineering import FeatureConfig, engineer_features


class AnomalyDetectionConfig(BaseModel):
    """Configuration for anomaly detection."""

    sensor: str
    contamination: float = Field(default=0.03, gt=0.0, le=0.5)
    random_state: int = 99
    rolling_window: int = Field(default=24, gt=0)

    @property
    def feature_config(self) -> FeatureConfig:
        return FeatureConfig(rolling_window=self.rolling_window)


@dataclass
class AnomalyDetectionResult:
    """Result of an anomaly detection run."""

    data: pd.DataFrame
    scores: np.ndarray
    is_anomaly: np.ndarray


def detect_anomalies(data: pd.DataFrame, config: AnomalyDetectionConfig) -> AnomalyDetectionResult:
    """Detect anomalies using IsolationForest on engineered features."""

    subset = data[data["sensor"] == config.sensor].copy()
    if subset.empty:
        raise ValueError(f"No observations for sensor '{config.sensor}'")
    features = engineer_features(subset, config.feature_config)
    feature_columns = ["value", "rolling_mean", "rolling_std"]
    if config.feature_config.include_derivative and "derivative" in features:
        feature_columns.append("derivative")
    model = IsolationForest(
        contamination=config.contamination,
        random_state=resolve_seed(
            "models.anomaly.isolation_forest",
            explicit=config.random_state,
        ),
    )
    feature_matrix = features[feature_columns]
    model.fit(feature_matrix)
    scores = model.decision_function(feature_matrix)
    is_anomaly = model.predict(feature_matrix) == -1
    features["anomaly_score"] = scores
    features["is_anomaly"] = is_anomaly
    return AnomalyDetectionResult(data=features, scores=scores, is_anomaly=is_anomaly)


__all__ = ["AnomalyDetectionConfig", "AnomalyDetectionResult", "detect_anomalies"]