"""Anomaly detection tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

import smartbuildsim.models.anomaly as anomaly
from smartbuildsim.models.anomaly import AnomalyDetectionConfig, detect_anomalies
from smartbuildsim.scenarios.presets import Scenario


def test_anomaly_detection(dataset: pd.DataFrame, scenario: Scenario) -> None:
    config = AnomalyDetectionConfig(**scenario.anomaly.dict())
    result = detect_anomalies(dataset, config)
    assert "anomaly_score" in result.data
    assert "is_anomaly" in result.data
    assert result.is_anomaly.dtype == np.bool_


def test_detect_anomalies_trains_model_before_scoring(
    dataset: pd.DataFrame, scenario: Scenario, monkeypatch
) -> None:
    config = AnomalyDetectionConfig(**scenario.anomaly.dict())

    class DummyIsolationForest:
        def __init__(self, *args, **kwargs):
            self._fitted_features: np.ndarray | None = None

        def fit(self, X, y=None):  # type: ignore[override]
            self._fitted_features = np.asarray(X)
            return self

        def _assert_fitted_with_same_features(self, X) -> None:
            if self._fitted_features is None:
                raise AssertionError("Model used before fit was called.")
            np.testing.assert_allclose(np.asarray(X), self._fitted_features)

        def decision_function(self, X):  # type: ignore[override]
            self._assert_fitted_with_same_features(X)
            return np.zeros(len(X), dtype=float)

        def predict(self, X):  # type: ignore[override]
            self._assert_fitted_with_same_features(X)
            return np.ones(len(X), dtype=int)

    monkeypatch.setattr(anomaly, "IsolationForest", DummyIsolationForest)

    result = anomaly.detect_anomalies(dataset, config)
    assert set(["value", "rolling_mean", "rolling_std"]).issubset(result.data.columns)
    assert "anomaly_score" in result.data
    assert "is_anomaly" in result.data
    np.testing.assert_allclose(result.scores, 0.0)
    assert not result.is_anomaly.any()
