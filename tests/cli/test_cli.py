"""CLI integration tests."""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]
from typer.testing import CliRunner

from smartbuildsim.cli.app import app


def _write_config(path: Path) -> Path:
    config = {
        "scenario": "office-small",
        "paths": {
            "output_dir": str(path.parent / "outputs"),
            "dataset": str(path.parent / "outputs" / "dataset.csv"),
        },
        "data": {"days": 3, "seed": 101},
        "models": {"forecasting": {"horizon": 1}},
        "cluster": {"sensors": ["cluster_energy", "cluster_co2"]},
        "viz": {"sensor": "office_energy"},
        "rl": {"episodes": 40, "steps_per_episode": 16},
    }
    with path.open("w", encoding="utf8") as handle:
        yaml.safe_dump(config, handle)
    return path


def test_cli_end_to_end(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    runner = CliRunner()

    result = runner.invoke(app, ["data", "generate", str(config_path)])
    assert result.exit_code == 0, result.stdout

    dataset_path = tmp_path / "outputs" / "dataset.csv"
    assert dataset_path.exists()

    result = runner.invoke(app, ["model", "forecast", str(config_path)])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "outputs" / "forecast_predictions.csv").exists()

    result = runner.invoke(app, ["model", "anomalies", str(config_path)])
    assert result.exit_code == 0, result.stdout
    anomalies_path = tmp_path / "outputs" / "anomalies.csv"
    assert anomalies_path.exists()

    result = runner.invoke(app, ["cluster", "run", str(config_path)])
    assert result.exit_code == 0, result.stdout
    clusters_path = tmp_path / "outputs" / "clusters.csv"
    assert clusters_path.exists()

    result = runner.invoke(
        app,
        [
            "rl",
            "train",
            str(config_path),
            "--override",
            "rl.episodes=20",
            "--override",
            "rl.steps_per_episode=12",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "outputs" / "rl_q_table.npy").exists()

    result = runner.invoke(
        app,
        [
            "viz",
            "plot",
            str(config_path),
            "--anomalies-path",
            str(anomalies_path),
            "--clusters-path",
            str(clusters_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "outputs" / "plot_office_energy.png").exists()
