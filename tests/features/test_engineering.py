"""Feature engineering tests."""

from __future__ import annotations

import pandas as pd

from smartbuildsim.features.engineering import (
    FeatureConfig,
    build_supervised_matrix,
    engineer_features,
)


def test_engineered_columns_present(dataset: pd.DataFrame) -> None:
    config = FeatureConfig(rolling_window=3)
    engineered = engineer_features(dataset, config)
    assert {"rolling_mean", "rolling_std", "derivative"}.issubset(engineered.columns)


def test_build_supervised_matrix(dataset: pd.DataFrame) -> None:
    sensor_series = (
        dataset[dataset["sensor"] == dataset["sensor"].iloc[0]]
        .sort_values("timestamp")
        .set_index("timestamp")["value"]
    )
    frame = build_supervised_matrix(sensor_series, [1, 2], 1)
    assert not frame.empty
    assert frame.columns.tolist() == ["target", "lag_1", "lag_2", "lead_1"]
