"""Feature engineering utilities."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    """Configuration for feature engineering steps."""

    rolling_window: int = Field(default=24, gt=0)
    include_derivative: bool = True
    lags: list[int] = Field(default_factory=lambda: [1, 2, 24])


def engineer_features(data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Compute rolling and derivative features for a time-series dataset."""

    engineered = data.copy()
    engineered.sort_values("timestamp", inplace=True)
    window = config.rolling_window
    engineered["rolling_mean"] = (
        engineered.groupby("sensor")["value"].transform(
            lambda column: column.rolling(window, min_periods=1).mean()
        )
    )
    engineered["rolling_std"] = (
        engineered.groupby("sensor")["value"].transform(
            lambda column: column.rolling(window, min_periods=1).std().fillna(0.0)
        )
    )
    if config.include_derivative:
        engineered["derivative"] = (
            engineered.groupby("sensor")["value"].transform(
                lambda column: column.diff().fillna(0.0)
            )
        )
    return engineered


def build_supervised_matrix(
    series: pd.Series, lags: Iterable[int], horizon: int
) -> pd.DataFrame:
    """Construct a supervised learning matrix using lagged values."""

    frame = pd.DataFrame({"target": series})
    for lag in sorted(set(lags)):
        frame[f"lag_{lag}"] = series.shift(lag)
    frame[f"lead_{horizon}"] = series.shift(-horizon)
    return frame.dropna()


__all__ = ["FeatureConfig", "engineer_features", "build_supervised_matrix"]