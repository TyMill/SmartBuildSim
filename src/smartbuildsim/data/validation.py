"""Validation utilities comparing synthetic datasets with real observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pydantic import BaseModel, Field


def _ensure_datetime(data: pd.Series) -> pd.Series:
    if not is_datetime64_any_dtype(data):
        return pd.to_datetime(data, utc=True)
    if getattr(data.dtype, "tz", None) is None:
        return data.dt.tz_localize("UTC")
    return data.dt.tz_convert("UTC")


def _prepare_series(frame: pd.DataFrame, sensor: str) -> pd.Series:
    subset = frame.loc[frame["sensor"] == sensor, ["timestamp", "value"]].copy()
    if subset.empty:
        return pd.Series(dtype=float)
    subset["timestamp"] = _ensure_datetime(subset["timestamp"])
    subset.sort_values("timestamp", inplace=True)
    subset.set_index("timestamp", inplace=True)
    return subset["value"].astype(float)


def _kolmogorov_smirnov(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if len(sample_a) == 0 or len(sample_b) == 0:
        return 0.0
    sample_a = np.sort(sample_a)
    sample_b = np.sort(sample_b)
    all_values = np.concatenate([sample_a, sample_b])
    cdf_a = np.searchsorted(sample_a, all_values, side="right") / len(sample_a)
    cdf_b = np.searchsorted(sample_b, all_values, side="right") / len(sample_b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _autocorrelation(series: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(series) <= lag:
        return float("nan")
    series = series - np.mean(series)
    numerator = np.dot(series[:-lag], series[lag:])
    denominator = np.dot(series, series)
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _dtw_distance(series_a: np.ndarray, series_b: np.ndarray) -> float:
    if len(series_a) == 0 or len(series_b) == 0:
        return float("nan")
    n, m = len(series_a), len(series_b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = abs(series_a[i - 1] - series_b[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m] / (n + m))


class DistributionComparison(BaseModel):
    """Summary statistics comparing synthetic and reference distributions."""

    mean_generated: float
    mean_reference: float
    std_generated: float
    std_reference: float
    ks_statistic: float


class TemporalMetrics(BaseModel):
    """Temporal similarity metrics."""

    autocorrelation_generated: dict[int, float] = Field(default_factory=dict)
    autocorrelation_reference: dict[int, float] = Field(default_factory=dict)
    dtw_distance: float


class SensorComparison(BaseModel):
    """Comparison payload for a single sensor."""

    sensor_generated: str
    sensor_reference: str
    distribution: DistributionComparison
    temporal: TemporalMetrics
    qualitative: str


class CorrelationComparison(BaseModel):
    """Correlation matrices and their delta."""

    generated: dict[str, dict[str, float]]
    reference: dict[str, dict[str, float]]
    delta: dict[str, dict[str, float]]
    frobenius_delta: float


class ValidationReport(BaseModel):
    """Aggregated validation report."""

    sensors: list[SensorComparison]
    correlations: CorrelationComparison
    notes: list[str]


def _distribution_metrics(series_a: pd.Series, series_b: pd.Series) -> DistributionComparison:
    return DistributionComparison(
        mean_generated=float(series_a.mean()),
        mean_reference=float(series_b.mean()),
        std_generated=float(series_a.std(ddof=0)),
        std_reference=float(series_b.std(ddof=0)),
        ks_statistic=_kolmogorov_smirnov(series_a.to_numpy(), series_b.to_numpy()),
    )


def _temporal_metrics(
    series_a: pd.Series, series_b: pd.Series, lags: Sequence[int]
) -> TemporalMetrics:
    series_a_values = series_a.to_numpy()
    series_b_values = series_b.to_numpy()
    autocorr_a = {lag: _autocorrelation(series_a_values, lag) for lag in lags}
    autocorr_b = {lag: _autocorrelation(series_b_values, lag) for lag in lags}
    distance = _dtw_distance(series_a_values, series_b_values)
    return TemporalMetrics(
        autocorrelation_generated=autocorr_a,
        autocorrelation_reference=autocorr_b,
        dtw_distance=distance,
    )


def _qualitative_summary(
    distribution: DistributionComparison, temporal: TemporalMetrics
) -> str:
    notes: list[str] = []
    ref_mean = abs(distribution.mean_reference) or 1.0
    mean_gap = abs(distribution.mean_generated - distribution.mean_reference) / ref_mean
    if mean_gap > 0.1:
        notes.append("różnica średniej powyżej 10%")
    ref_std = distribution.std_reference or 1.0
    std_gap = abs(distribution.std_generated - distribution.std_reference) / ref_std
    if std_gap > 0.15:
        notes.append("odchylenie standardowe odbiega o ponad 15%")
    if distribution.ks_statistic > 0.25:
        notes.append("istotna różnica rozkładów (KS > 0.25)")
    if temporal.dtw_distance and temporal.dtw_distance > ref_std * 3:
        notes.append("wysoki dystans DTW wskazuje na inny przebieg czasowy")
    if not notes:
        return "profil zgodny w granicach tolerancji"
    return "; ".join(notes)


def _correlation_matrix(frame: pd.DataFrame, sensors: Sequence[str]) -> pd.DataFrame:
    subset = frame.loc[frame["sensor"].isin(sensors), ["timestamp", "sensor", "value"]].copy()
    if subset.empty:
        return pd.DataFrame(dtype=float)
    subset["timestamp"] = _ensure_datetime(subset["timestamp"])
    pivot = subset.pivot_table(index="timestamp", columns="sensor", values="value")
    return pivot.corr()

def _correlation_comparison(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    sensor_mapping: Mapping[str, str],
) -> CorrelationComparison:
    gen_sensors = list(sensor_mapping.keys())
    ref_sensors = [sensor_mapping[name] for name in gen_sensors]
    corr_gen = _correlation_matrix(generated, gen_sensors)
    corr_ref = _correlation_matrix(reference, ref_sensors)
    corr_gen = corr_gen.reindex(index=gen_sensors, columns=gen_sensors)
    corr_ref = corr_ref.reindex(index=ref_sensors, columns=ref_sensors)

    generated_dict: dict[str, dict[str, float]] = {}
    reference_dict: dict[str, dict[str, float]] = {}
    delta_values: dict[str, dict[str, float]] = {}
    squared: list[float] = []

    for idx, gen_sensor in enumerate(gen_sensors):
        ref_sensor = ref_sensors[idx]
        gen_row: dict[str, float] = {}
        ref_row: dict[str, float] = {}
        delta_row: dict[str, float] = {}
        for jdx, gen_sensor_col in enumerate(gen_sensors):
            ref_sensor_col = ref_sensors[jdx]
            gen_val = (
                float(corr_gen.loc[gen_sensor, gen_sensor_col])
                if gen_sensor in corr_gen.index and gen_sensor_col in corr_gen.columns
                else float("nan")
            )
            ref_val = (
                float(corr_ref.loc[ref_sensor, ref_sensor_col])
                if ref_sensor in corr_ref.index and ref_sensor_col in corr_ref.columns
                else float("nan")
            )
            gen_row[gen_sensor_col] = gen_val
            ref_row[gen_sensor_col] = ref_val
            diff = gen_val - ref_val
            delta_row[gen_sensor_col] = diff
            if not np.isnan(diff):
                squared.append(diff**2)
        generated_dict[gen_sensor] = gen_row
        reference_dict[gen_sensor] = ref_row
        delta_values[gen_sensor] = delta_row

    frob = float(np.sqrt(np.mean(squared))) if squared else float("nan")
    return CorrelationComparison(
        generated=generated_dict,
        reference=reference_dict,
        delta=delta_values,
        frobenius_delta=frob,
    )


def compare_datasets(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    sensor_mapping: Mapping[str, str] | None = None,
    lags: Sequence[int] = (1, 24),
) -> ValidationReport:
    """Compare synthetic ``generated`` data against ``reference`` observations."""

    generated = generated.copy()
    reference = reference.copy()
    generated["timestamp"] = _ensure_datetime(generated["timestamp"])
    reference["timestamp"] = _ensure_datetime(reference["timestamp"])

    if sensor_mapping is None:
        common = sorted(set(generated["sensor"]).intersection(reference["sensor"]))
        sensor_mapping = {sensor: sensor for sensor in common}

    sensor_reports: list[SensorComparison] = []
    notes: list[str] = []
    for generated_sensor, reference_sensor in sensor_mapping.items():
        series_generated = _prepare_series(generated, generated_sensor)
        series_reference = _prepare_series(reference, reference_sensor)
        if series_generated.empty or series_reference.empty:
            continue
        distribution = _distribution_metrics(series_generated, series_reference)
        temporal = _temporal_metrics(series_generated, series_reference, lags)
        qualitative = _qualitative_summary(distribution, temporal)
        sensor_reports.append(
            SensorComparison(
                sensor_generated=generated_sensor,
                sensor_reference=reference_sensor,
                distribution=distribution,
                temporal=temporal,
                qualitative=qualitative,
            )
        )
        notes.append(f"{generated_sensor}: {qualitative}")

    correlation_report = _correlation_comparison(generated, reference, sensor_mapping)

    return ValidationReport(sensors=sensor_reports, correlations=correlation_report, notes=notes)


__all__ = [
    "CorrelationComparison",
    "DistributionComparison",
    "SensorComparison",
    "TemporalMetrics",
    "ValidationReport",
    "compare_datasets",
]
