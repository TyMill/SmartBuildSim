"""Typer-based CLI entry point for SmartBuildSim."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import typer

from ..bim.loader import load_building, write_default_schema
from ..bim.schema import Building
from ..config import DeterminismConfig, configure_determinism
from ..data.generator import DataGeneratorConfig, generate_and_save
from ..models.anomaly import AnomalyDetectionConfig, detect_anomalies
from ..models.clustering import ClusteringConfig, cluster_zones
from ..models.forecasting import (
    ForecastingConfig,
    persist_model,
    train_forecasting_model,
)
from ..models.rl import RLConfig, evaluate_policy, train_policy
from ..scenarios.presets import Scenario, get_scenario, list_scenarios
from ..utils.helpers import (
    apply_overrides,
    dump_yaml,
    ensure_directory,
    load_yaml,
    model_from_mapping,
)
from ..viz.plots import PlotConfig, plot_time_series

app = typer.Typer(help="Synthetic smart building experimentation toolkit")
bim_app = typer.Typer()
data_app = typer.Typer()
model_app = typer.Typer()
cluster_app = typer.Typer()
rl_app = typer.Typer()
viz_app = typer.Typer()

app.add_typer(bim_app, name="bim")
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(cluster_app, name="cluster")
app.add_typer(rl_app, name="rl")
app.add_typer(viz_app, name="viz")


def _load_config(path: Path, overrides: list[str]) -> dict[str, object]:
    """Load YAML configuration and apply CLI overrides."""
    config = load_yaml(path)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def _resolve_scenario(config: dict[str, object]) -> Scenario | None:
    """Return a scenario preset referenced by the configuration, if any."""
    scenario_name = config.get("scenario")
    if isinstance(scenario_name, str):
        return get_scenario(scenario_name)
    return None


def _resolve_building(
    config: dict[str, object], scenario: Scenario | None, base_path: Path
) -> Building:

    """Resolve a building definition from scenario or inline configuration."""
    if scenario is not None:
        return scenario.building
    if "building_path" in config:
        return load_building((base_path / str(config["building_path"])).resolve())
    raw_building = config.get("building")
    if isinstance(raw_building, dict):
        return cast(Building, Building.parse_obj(raw_building))
    raise ValueError("Configuration must provide a building definition or scenario")


def _merge_config(
    defaults: dict[str, object], overrides: dict[str, object] | None
) -> dict[str, object]:

    """Merge override mappings on top of default dictionaries."""
    merged = defaults.copy()
    if overrides:
        merged.update(overrides)
    return merged


def _maybe_mapping(value: object) -> dict[str, object] | None:
    """Return the value if it is a mapping, otherwise ``None``."""
    if isinstance(value, dict):
        return value
    return None


def _configure_determinism_from_config(config: dict[str, object]) -> None:
    """Initialise deterministic behaviour from the configuration mapping."""

    section = _maybe_mapping(config.get("determinism"))
    if section is not None:
        determinism = model_from_mapping(DeterminismConfig, section)
        configure_determinism(determinism, force=True)
    else:
        configure_determinism()


def _load_dataset(path: Path) -> pd.DataFrame:
    """Read a dataset CSV with timestamp parsing."""
    data = pd.read_csv(path, parse_dates=["timestamp"])
    return data


def _default_output_dir(config: dict[str, object], base_path: Path) -> Path:
    """Determine the default output directory from config or fall back to ``outputs``."""
    paths = config.get("paths")
    if isinstance(paths, dict) and "output_dir" in paths:
        return (base_path / str(paths["output_dir"])).resolve()
    return (base_path / "outputs").resolve()


@bim_app.command("init")
def bim_init(
    output: Path = typer.Argument(..., help="Where to write the BIM schema"),
    scenario: str | None = typer.Option(None, help="Scenario preset to export"),
) -> None:
    """Write a BIM schema to disk."""

    if scenario:
        preset = get_scenario(scenario)
        dump_yaml(preset.building.dict(by_alias=True), output)
        typer.echo(f"Wrote scenario '{scenario}' schema to {output}")
    else:
        write_default_schema(output)
        typer.echo(f"Wrote default schema to {output}")


@data_app.command("generate")
def data_generate(
    config_path: Path = typer.Argument(..., help="YAML configuration"),
    overrides: list[str] = typer.Option(
        [], "--override", "-O", help="Override configuration values (key=value)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output dataset path (defaults from config)"
    ),
) -> None:
    """Generate deterministic synthetic data."""

    config = _load_config(config_path, overrides)
    _configure_determinism_from_config(config)
    base_path = config_path.parent
    scenario = _resolve_scenario(config)
    building = _resolve_building(config, scenario, base_path)
    generator_dict: dict[str, object] = scenario.data.dict() if scenario else {}
    generator_dict = _merge_config(generator_dict, _maybe_mapping(config.get("data")))
    generator = model_from_mapping(DataGeneratorConfig, generator_dict)
    output_dir = _default_output_dir(config, base_path)
    ensure_directory(output_dir)
    dataset_path = output or output_dir / "dataset.csv"
    dataset = generate_and_save(building, generator, dataset_path)
    typer.echo(f"Generated dataset with {len(dataset)} rows at {dataset_path}")


def _resolve_dataset_path(
    config: dict[str, object], base_path: Path, explicit: Path | None
) -> Path:
    """Resolve the dataset path from CLI args or configuration."""
    if explicit is not None:
        return explicit
    paths = config.get("paths")
    if isinstance(paths, dict) and "dataset" in paths:
        return (base_path / str(paths["dataset"])).resolve()
    return (base_path / "outputs" / "dataset.csv").resolve()


def _resolve_output(config: dict[str, object], base_path: Path, default_name: str) -> Path:
    """Construct an output path inside the configured output directory."""
    output_dir = _default_output_dir(config, base_path)
    ensure_directory(output_dir)
    return output_dir / default_name


@model_app.command("forecast")
def model_forecast(
    config_path: Path = typer.Argument(..., help="YAML configuration"),
    overrides: list[str] = typer.Option(
        [], "--override", "-O", help="Override configuration values (key=value)"
    ),
    data_path: Path | None = typer.Option(None, help="Dataset CSV location"),
) -> None:
    """Train and persist a forecasting model."""

    config = _load_config(config_path, overrides)
    _configure_determinism_from_config(config)
    base_path = config_path.parent
    scenario = _resolve_scenario(config)
    dataset_path = _resolve_dataset_path(config, base_path, data_path)
    data = _load_dataset(dataset_path)
    forecast_dict: dict[str, object] = (
        scenario.forecasting.dict() if scenario else {}
    )
    models_config = _maybe_mapping(config.get("models"))
    forecast_overrides = (
        _maybe_mapping(models_config.get("forecasting")) if models_config else None
    )
    forecast_dict = _merge_config(forecast_dict, forecast_overrides)
    forecast_config = model_from_mapping(ForecastingConfig, forecast_dict)
    result = train_forecasting_model(data, forecast_config)
    model_path = _resolve_output(config, base_path, f"forecast_{forecast_config.sensor}.joblib")
    persist_model(result.model, model_path)
    predictions_path = _resolve_output(config, base_path, "forecast_predictions.csv")
    pd.DataFrame(
        {"prediction": result.predictions, "target": result.targets}
    ).to_csv(predictions_path, index=False)
    typer.echo(
        f"Forecast model persisted to {model_path} (RMSE={result.rmse:.3f}); predictions saved to {predictions_path}"
    )


@model_app.command("anomalies")
def model_anomalies(
    config_path: Path = typer.Argument(..., help="YAML configuration"),
    overrides: list[str] = typer.Option(
        [], "--override", "-O", help="Override configuration values (key=value)"
    ),
    data_path: Path | None = typer.Option(None, help="Dataset CSV location"),
) -> None:
    """Run anomaly detection and persist the annotated dataset."""

    config = _load_config(config_path, overrides)
    _configure_determinism_from_config(config)
    base_path = config_path.parent
    scenario = _resolve_scenario(config)
    dataset_path = _resolve_dataset_path(config, base_path, data_path)
    data = _load_dataset(dataset_path)
    anomaly_dict: dict[str, object] = scenario.anomaly.dict() if scenario else {}
    models_config = _maybe_mapping(config.get("models"))
    anomaly_overrides = (
        _maybe_mapping(models_config.get("anomaly")) if models_config else None
    )
    anomaly_dict = _merge_config(anomaly_dict, anomaly_overrides)
    anomaly_config = model_from_mapping(AnomalyDetectionConfig, anomaly_dict)
    result = detect_anomalies(data, anomaly_config)
    anomalies_path = _resolve_output(config, base_path, "anomalies.csv")
    result.data.to_csv(anomalies_path, index=False)
    typer.echo(
        f"Anomaly results for sensor '{anomaly_config.sensor}' written to {anomalies_path}"
    )


@cluster_app.command("run")
def cluster_run(
    config_path: Path = typer.Argument(..., help="YAML configuration"),
    overrides: list[str] = typer.Option(
        [], "--override", "-O", help="Override configuration values (key=value)"
    ),
    data_path: Path | None = typer.Option(None, help="Dataset CSV location"),
) -> None:
    """Cluster building zones and write assignments to disk."""

    config = _load_config(config_path, overrides)
    _configure_determinism_from_config(config)
    base_path = config_path.parent
    scenario = _resolve_scenario(config)
    dataset_path = _resolve_dataset_path(config, base_path, data_path)
    data = _load_dataset(dataset_path)
    cluster_dict: dict[str, object] = scenario.clustering.dict() if scenario else {}
    cluster_dict = _merge_config(cluster_dict, _maybe_mapping(config.get("cluster")))
    cluster_config = model_from_mapping(ClusteringConfig, cluster_dict)
    result = cluster_zones(data, cluster_config)
    clusters_path = _resolve_output(config, base_path, "clusters.csv")
    result.assignments.to_csv(clusters_path, index=False)
    typer.echo(
        f"Cluster assignments for sensors {cluster_config.sensors} saved to {clusters_path}"
    )


@rl_app.command("train")
def rl_train(
    config_path: Path = typer.Argument(..., help="YAML configuration"),
    overrides: list[str] = typer.Option(
        [], "--override", "-O", help="Override configuration values (key=value)"
    ),
) -> None:
    """Train the reinforcement learning policy and persist the Q-table."""

    config = _load_config(config_path, overrides)
    _configure_determinism_from_config(config)
    base_path = config_path.parent
    scenario = _resolve_scenario(config)
    rl_dict: dict[str, object] = scenario.rl.dict() if scenario else {}
    rl_dict = _merge_config(rl_dict, _maybe_mapping(config.get("rl")))
    rl_config = model_from_mapping(RLConfig, rl_dict)
    result = train_policy(rl_config)
    q_path = _resolve_output(config, base_path, "rl_q_table.npy")
    np.save(q_path, result.q_table)
    avg_reward = result.average_reward()
    eval_reward = evaluate_policy(result)
    typer.echo(
        f"RL training complete. Q-table saved to {q_path}. Average reward {avg_reward:.3f}, evaluation reward {eval_reward:.3f}"
    )


@viz_app.command("plot")
def viz_plot(
    config_path: Path = typer.Argument(..., help="YAML configuration"),
    overrides: list[str] = typer.Option(
        [], "--override", "-O", help="Override configuration values (key=value)"
    ),
    data_path: Path | None = typer.Option(None, help="Dataset CSV location"),
    anomalies_path: Path | None = typer.Option(None, help="Optional anomaly CSV"),
    clusters_path: Path | None = typer.Option(None, help="Optional cluster CSV"),
) -> None:
    """Generate a Matplotlib plot for a configured sensor."""

    config = _load_config(config_path, overrides)
    _configure_determinism_from_config(config)
    base_path = config_path.parent
    dataset_path = _resolve_dataset_path(config, base_path, data_path)
    data = _load_dataset(dataset_path)
    scenario = _resolve_scenario(config)
    plot_defaults: dict[str, object] = {}
    if scenario:
        plot_defaults = {"sensor": scenario.forecasting.sensor}
    plot_dict = _merge_config(plot_defaults, _maybe_mapping(config.get("viz")))
    plot_config = model_from_mapping(PlotConfig, plot_dict)
    anomalies = None
    if anomalies_path:
        anomalies = pd.read_csv(anomalies_path, parse_dates=["timestamp"])
    clusters = None
    if clusters_path:
        clusters = pd.read_csv(clusters_path)
    plot_path = _resolve_output(config, base_path, f"plot_{plot_config.sensor}.png")
    plot_time_series(data, plot_config, plot_path, anomalies=anomalies, clusters=clusters)
    typer.echo(f"Plot saved to {plot_path}")


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Show available scenario presets when no command is provided."""

    if ctx.invoked_subcommand is None:
        presets = ", ".join(list_scenarios())
        typer.echo(f"Available scenarios: {presets}")


def main() -> None:
    """CLI entry point used by ``pyproject.toml`` scripts."""

    app()


__all__ = ["app", "main"]