"""Tests for the dataset validation utilities."""

from __future__ import annotations

import pandas as pd

from smartbuildsim.data.validation import compare_datasets


def test_compare_datasets_produces_report(dataset: pd.DataFrame) -> None:
    generated = dataset.loc[dataset["sensor"] == "office_energy"].copy()
    reference = generated.copy()
    reference["value"] = reference["value"] * 1.05

    report = compare_datasets(generated, reference, lags=(1,))

    assert report.sensors, "Expected at least one sensor report"
    sensor = report.sensors[0]
    assert sensor.sensor_generated == "office_energy"
    assert "office_energy" in report.correlations.generated
    assert report.notes
