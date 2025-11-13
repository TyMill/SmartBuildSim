"""SmartBuildSim Google Colab showcase notebook.

Copy the entire contents of this file into a single Google Colab notebook
cell to reproduce all tables and figures used in Sections 5.2–5.3 of the
SmartBuildSim paper-style documentation. The workflow installs the library,
constructs the office-small scenario, benchmarks forecasting/anomaly/RL
models, and generates publication-ready visualisations.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure local repository is importable when running from a clone (e.g. CI)
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if REPO_ROOT.exists():
        sys.path.insert(0, str(REPO_ROOT))
        src_dir = REPO_ROOT / "src"
        if src_dir.exists():
            sys.path.insert(0, str(src_dir))
except NameError:  # pragma: no cover - __file__ is undefined inside notebooks
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # rich display in notebooks
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - fallback for plain Python
    def display(obj: object) -> None:  # type: ignore[override]
        print(obj)

    def Markdown(text: str) -> str:  # type: ignore[override]
        return text


@dataclass
class RewardHistory:
    """Container storing reward history for plotting."""

    algorithm: str
    seed: int
    rewards: list[float]


# ---------------------------------------------------------------------------
# 0. Environment preparation
# ---------------------------------------------------------------------------

def _ensure_package(name: str, spec: str | None = None) -> None:
    """Import ``name`` and install ``spec`` via pip if necessary."""

    try:
        importlib.import_module(name)
    except ImportError:  # pragma: no cover - executed in Colab
        subprocess.run([sys.executable, "-m", "pip", "install", spec or name], check=True)
try:
    from smartbuildsim.data.generator import generate_dataset
    from smartbuildsim.data.validation import compare_datasets
    from smartbuildsim.evaluation.benchmark import (
        SoftQLearningConfig,
        run_anomaly_benchmark,
        run_regression_benchmark,
        run_rl_benchmark,
        train_soft_q_policy,
    )
    from smartbuildsim.features.engineering import build_supervised_matrix
    from smartbuildsim.models.rl import RLConfig, train_policy
    from smartbuildsim.scenarios.presets import get_scenario
except ImportError:  # pragma: no cover - executed in Colab
    _ensure_package("smartbuildsim", "git+https://github.com/TyMill/SmartBuildSim.git")
    from smartbuildsim.data.generator import generate_dataset
    from smartbuildsim.data.validation import compare_datasets
    from smartbuildsim.evaluation.benchmark import (
        SoftQLearningConfig,
        run_anomaly_benchmark,
        run_regression_benchmark,
        run_rl_benchmark,
        train_soft_q_policy,
    )
    from smartbuildsim.features.engineering import build_supervised_matrix
    from smartbuildsim.models.rl import RLConfig, train_policy
    from smartbuildsim.scenarios.presets import get_scenario


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="invalid value encountered", category=RuntimeWarning
)

OUTPUT_ROOT = Path("smartbuildsim_colab_outputs")
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data generation and validation (Table 5 & Figure 5 series)
# ---------------------------------------------------------------------------

display(Markdown("## 1. Synthetic data generation and validation"))

scenario = get_scenario("office-small")
data_config = scenario.data.model_copy(update={"anomaly_chance": 0.02, "anomaly_magnitude": 3.0})
dataset = generate_dataset(
    scenario.building,
    data_config,
    include_anomaly_labels=True,
)
dataset_path = OUTPUT_ROOT / "office_small_dataset.csv"
dataset.to_csv(dataset_path, index=False)

display(Markdown(f"Dataset saved to `{dataset_path}` with {len(dataset):,} rows."))

reference_url = (
    "https://raw.githubusercontent.com/TyMill/SmartBuildSim/main/"
    "docs/reference/datasets/ashrae_sample.csv"
)
local_reference = None
if "REPO_ROOT" in globals():
    candidate = REPO_ROOT / "docs" / "reference" / "datasets" / "ashrae_sample.csv"
    if candidate.exists():
        local_reference = candidate
reference = pd.read_csv(local_reference or reference_url, parse_dates=["timestamp"])

validation = compare_datasets(
    dataset,
    reference,
    sensor_mapping={"office_energy": "meter_0_energy"},
    lags=(1, 24),
)

validation_rows: list[dict[str, float | str]] = []
for sensor_report in validation.sensors:
    row: dict[str, float | str] = {
        "synthetic_sensor": sensor_report.sensor_generated,
        "reference_sensor": sensor_report.sensor_reference,
        "mean_synth": sensor_report.distribution.mean_generated,
        "mean_ref": sensor_report.distribution.mean_reference,
        "std_synth": sensor_report.distribution.std_generated,
        "std_ref": sensor_report.distribution.std_reference,
        "ks_stat": sensor_report.distribution.ks_statistic,
        "dtw_distance": sensor_report.temporal.dtw_distance,
        "qualitative": sensor_report.qualitative,
    }
    for lag in sorted(sensor_report.temporal.autocorrelation_generated):
        gen_val = sensor_report.temporal.autocorrelation_generated[lag]
        ref_val = sensor_report.temporal.autocorrelation_reference.get(lag, float("nan"))
        row[f"autocorr_{lag}h_gen"] = gen_val
        row[f"autocorr_{lag}h_ref"] = ref_val
        row[f"autocorr_{lag}h_delta"] = gen_val - ref_val
    validation_rows.append(row)

table5 = pd.DataFrame(validation_rows)
correlation_delta = validation.correlations.frobenius_delta
table5_path = TABLE_DIR / "table_5_validation_summary.csv"
table5.to_csv(table5_path, index=False)

summary_notes = "\n".join(f"* {note}" for note in validation.notes)

display(Markdown("### Table 5. Validation summary (synthetic vs. ASHRAE)"))
display(table5.round(4))
display(Markdown(f"Correlation Frobenius Δ: **{correlation_delta:.4f}**"))
display(Markdown("Validation notes:"))
display(Markdown(summary_notes))

display(Markdown("#### Figure 5A — Time-series comparison"))
reference_series = (
    reference[reference["sensor"] == "meter_0_energy"].sort_values("timestamp")["value"].reset_index(drop=True)
)
office_series = (
    dataset[dataset["sensor"] == "office_energy"].sort_values("timestamp")["value"].reset_index(drop=True)
)
length = min(len(reference_series), len(office_series))
comparison_frame = pd.DataFrame(
    {
        "hour": np.arange(length),
        "Synthetic (office_energy)": office_series.iloc[:length],
        "ASHRAE meter_0": reference_series.iloc[:length],
    }
)
fig_5a, ax = plt.subplots(figsize=(10, 4))
ax.plot(comparison_frame["hour"], comparison_frame["Synthetic (office_energy)"], label="Synthetic")
ax.plot(comparison_frame["hour"], comparison_frame["ASHRAE meter_0"], label="ASHRAE", alpha=0.8)
ax.set_xlabel("Elapsed hour")
ax.set_ylabel("Energy (kWh)")
ax.set_title("Figure 5A — Time-series comparison")
ax.legend()
fig_5a.tight_layout()
fig_5a_path = FIG_DIR / "figure_5A_time_series.png"
fig_5a.savefig(fig_5a_path, dpi=180)
plt.show()


def _autocorrelation(series: pd.Series, max_lag: int) -> pd.Series:
    return pd.Series({lag: series.autocorr(lag=lag) for lag in range(1, max_lag + 1)})


display(Markdown("#### Figure 5B — Autocorrelation comparison"))
max_lag = 24
acf_synth = _autocorrelation(office_series, max_lag)
acf_ref = _autocorrelation(reference_series, max_lag)
fig_5b, ax = plt.subplots(figsize=(10, 4))
ax.plot(acf_synth.index, acf_synth.values, label="Synthetic", marker="o")
ax.plot(acf_ref.index, acf_ref.values, label="ASHRAE", marker="o")
ax.set_xlabel("Lag (hours)")
ax.set_ylabel("Autocorrelation")
ax.set_title("Figure 5B — Autocorrelation comparison")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
fig_5b.tight_layout()
fig_5b_path = FIG_DIR / "figure_5B_autocorrelation.png"
fig_5b.savefig(fig_5b_path, dpi=180)
plt.show()


display(Markdown("#### Figure 5C — Distribution and CDF with KS divergence"))
fig_5c, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(12, 4))
bins = max(15, int(np.sqrt(length)))
ax_hist.hist(
    office_series,
    bins=bins,
    density=True,
    alpha=0.5,
    label="Synthetic",
)
ax_hist.hist(
    reference_series,
    bins=bins,
    density=True,
    alpha=0.5,
    label="ASHRAE",
)
ax_hist.set_title("Probability density")
ax_hist.set_xlabel("Energy (kWh)")
ax_hist.set_ylabel("Density")
ax_hist.legend()

for values, label in [
    (np.sort(office_series), "Synthetic"),
    (np.sort(reference_series), "ASHRAE"),
]:
    cdf = np.linspace(0, 1, len(values), endpoint=False)
    ax_cdf.plot(values, cdf, label=label)
ax_cdf.set_title(f"CDF comparison (KS={table5.loc[0, 'ks_stat']:.3f})")
ax_cdf.set_xlabel("Energy (kWh)")
ax_cdf.set_ylabel("Cumulative probability")
ax_cdf.legend()
fig_5c.tight_layout()
fig_5c_path = FIG_DIR / "figure_5C_distribution_cdf.png"
fig_5c.savefig(fig_5c_path, dpi=180)
plt.show()


# ---------------------------------------------------------------------------
# 2. Benchmark results (Tables 6–8 & Figures 6A–6C)
# ---------------------------------------------------------------------------

display(Markdown("## 2. Benchmark results"))

sensor_name = scenario.forecasting.sensor
sensor_frame = dataset[dataset["sensor"] == sensor_name].copy()
sensor_frame.sort_values("timestamp", inplace=True)
sensor_series = sensor_frame.set_index("timestamp")["value"].astype(float)

regression_result = run_regression_benchmark(
    sensor_series,
    lags=scenario.forecasting.lags,
    horizon=scenario.forecasting.horizon,
    seeds=[0, 1, 2],
)

baseline_reg = regression_result["baseline"]
metrics_reg = regression_result["metrics"].copy().reset_index().rename(columns={"index": "model"})
metrics_reg["delta_vs_baseline"] = (
    metrics_reg["mean"] - metrics_reg.loc[metrics_reg["model"] == baseline_reg, "mean"].item()
)
significance_reg = regression_result["significance"].set_index("model")
table6 = metrics_reg.join(significance_reg, on="model")
table6_path = TABLE_DIR / "table_6_regression_benchmark.csv"
table6.to_csv(table6_path, index=False)

display(Markdown("### Table 6. Regression benchmark results (RMSE)"))
display(table6.round(4))

anomaly_features = build_supervised_matrix(
    sensor_series,
    lags=scenario.forecasting.lags,
    horizon=1,
)
labels = sensor_frame.iloc[-len(anomaly_features) :]["is_anomaly"].reset_index(drop=True)

anomaly_result = run_anomaly_benchmark(
    anomaly_features,
    labels,
    seeds=[0, 1, 2],
)

baseline_anom = anomaly_result["baseline"]
metrics_anom = anomaly_result["metrics"].copy().reset_index().rename(columns={"index": "model"})
metrics_anom["delta_vs_baseline"] = (
    metrics_anom["mean"] - metrics_anom.loc[metrics_anom["model"] == baseline_anom, "mean"].item()
)
significance_anom = anomaly_result["significance"].set_index("model")
table7 = metrics_anom.join(significance_anom, on="model")
table7_path = TABLE_DIR / "table_7_anomaly_benchmark.csv"
table7.to_csv(table7_path, index=False)

display(Markdown("### Table 7. Anomaly detection benchmark results (F1-score)"))
display(table7.round(4))

rl_seeds = [7, 11, 21, 42]
rl_base_config = RLConfig(**scenario.rl.model_dump())
soft_config = SoftQLearningConfig(**scenario.rl.model_dump(), temperature=0.08, policy_smoothing=0.1)

rl_results: list[RewardHistory] = []
soft_results: list[RewardHistory] = []
for seed in rl_seeds:
    base_conf = rl_base_config.model_copy(update={"seed": seed})
    base_result = train_policy(base_conf)
    rl_results.append(RewardHistory("Q-learning", seed, base_result.reward_history))

    soft_conf = soft_config.model_copy(update={"seed": seed})
    soft_result = train_soft_q_policy(soft_conf)
    soft_results.append(RewardHistory("Soft Q", seed, soft_result.reward_history))

rl_benchmark = run_rl_benchmark(rl_base_config, soft_config, seeds=rl_seeds)
metrics_rl = rl_benchmark.metrics.reset_index().rename(columns={"index": "algorithm"})
metrics_rl.rename(columns={"mean": "mean_reward", "std": "std_reward"}, inplace=True)
metrics_rl["delta_vs_q_learning"] = metrics_rl["mean_reward"] - metrics_rl.loc[
    metrics_rl["algorithm"] == "q_learning", "mean_reward"
].item()
rl_significance = rl_benchmark.significance
if not rl_significance.empty:
    rl_significance = rl_significance.rename(columns={"model": "algorithm"})
    metrics_rl = metrics_rl.merge(rl_significance, on="algorithm", how="left")

table8_path = TABLE_DIR / "table_8_rl_benchmark.csv"
metrics_rl.to_csv(table8_path, index=False)

display(Markdown("### Table 8. Reinforcement learning benchmark results"))
display(metrics_rl.round(4))


# Figures 6A–6C
display(Markdown("#### Figure 6A — RMSE distributions"))
rmse_long = pd.DataFrame(
    (
        {"model": model, "rmse": score}
        for model, scores in regression_result["rmse_scores"].items()
        for score in scores
    )
)
fig_6a, ax = plt.subplots(figsize=(8, 4))
rmse_models = list(rmse_long["model"].unique())
rmse_data = [rmse_long.loc[rmse_long["model"] == model, "rmse"].to_numpy() for model in rmse_models]
positions = np.arange(1, len(rmse_models) + 1)
violin: dict[str, Any] = ax.violinplot(
    rmse_data, positions=positions, showmeans=True, widths=0.8
)
for body in violin.get("bodies", []):
    body.set_alpha(0.6)
ax.axhline(
    metrics_reg.loc[metrics_reg["model"] == baseline_reg, "mean"].item(),
    color="red",
    linestyle="--",
    label="Baseline mean",
)
ax.set_title("Figure 6A — RMSE distributions")
ax.set_xlabel("Model + scaler")
ax.set_ylabel("RMSE")
ax.set_xticks(positions, rmse_models, rotation=35, ha="right")
ax.legend(loc="upper right")
fig_6a.tight_layout()
fig_6a_path = FIG_DIR / "figure_6A_rmse_distributions.png"
fig_6a.savefig(fig_6a_path, dpi=180)
plt.show()

f1_long = pd.DataFrame(
    (
        {"model": model, "f1": score}
        for model, scores in anomaly_result["f1_scores"].items()
        for score in scores
    )
)
display(Markdown("#### Figure 6B — F1-score distributions"))
fig_6b, ax = plt.subplots(figsize=(8, 4))
f1_models = list(f1_long["model"].unique())
f1_data = [f1_long.loc[f1_long["model"] == model, "f1"].to_numpy() for model in f1_models]
positions = np.arange(1, len(f1_models) + 1)
box = ax.boxplot(f1_data, positions=positions, patch_artist=True, widths=0.7)
for patch in box["boxes"]:
    patch.set_alpha(0.6)
ax.axhline(
    metrics_anom.loc[metrics_anom["model"] == baseline_anom, "mean"].item(),
    color="red",
    linestyle="--",
    label="Baseline mean",
)
ax.set_title("Figure 6B — F1-score distributions")
ax.set_xlabel("Detector + scaler")
ax.set_ylabel("F1-score")
ax.set_xticks(positions, f1_models, rotation=35, ha="right")
ax.legend(loc="lower right")
fig_6b.tight_layout()
fig_6b_path = FIG_DIR / "figure_6B_f1_distributions.png"
fig_6b.savefig(fig_6b_path, dpi=180)
plt.show()

reward_frames: list[pd.DataFrame] = []
for container in rl_results + soft_results:
    reward_frames.append(
        pd.DataFrame(
            {
                "episode": np.arange(1, len(container.rewards) + 1),
                "reward": container.rewards,
                "algorithm": container.algorithm,
                "seed": container.seed,
            }
        )
    )
rewards_long = pd.concat(reward_frames, ignore_index=True)
mean_rewards = rewards_long.groupby(["algorithm", "episode"], as_index=False)["reward"].mean()
display(Markdown("#### Figure 6C — Reward trajectories"))
fig_6c, ax = plt.subplots(figsize=(8, 4))
for algorithm, group in mean_rewards.groupby("algorithm"):
    ax.plot(group["episode"], group["reward"], label=algorithm)
ax.set_title("Figure 6C — Reward trajectories (mean over seeds)")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.legend()
fig_6c.tight_layout()
fig_6c_path = FIG_DIR / "figure_6C_reward_trajectories.png"
fig_6c.savefig(fig_6c_path, dpi=180)
plt.show()


# ---------------------------------------------------------------------------
# 3. Scenario-specific analysis (Figures 7A–7C)
# ---------------------------------------------------------------------------

display(Markdown("## 3. Scenario-specific analysis"))

# Figure 7A — Daily energy cycles
display(Markdown("#### Figure 7A — Daily energy cycles"))
office = sensor_frame.copy()
office["day"] = office["timestamp"].dt.tz_convert("UTC").dt.date
office["hour"] = office["timestamp"].dt.tz_convert("UTC").dt.hour
daily_profile = office.pivot_table(index="hour", columns="day", values="value", aggfunc="mean")
fig_7a, ax = plt.subplots(figsize=(9, 4))
ax.plot(daily_profile.index, daily_profile, alpha=0.8)
ax.set_title("Figure 7A — Daily office energy cycles")
ax.set_xlabel("Hour of day (UTC)")
ax.set_ylabel("Energy (kWh)")
ax.legend([str(col) for col in daily_profile.columns], title="Day", bbox_to_anchor=(1.02, 1), loc="upper left")
fig_7a.tight_layout()
fig_7a_path = FIG_DIR / "figure_7A_daily_cycles.png"
fig_7a.savefig(fig_7a_path, dpi=180)
plt.show()

# Figure 7B — Sensor heatmap / multi-zone map
display(Markdown("#### Figure 7B — Sensor heatmap"))
heatmap_frame = (
    dataset.groupby(["zone", "sensor"], as_index=False)["value"].mean().pivot(index="zone", columns="sensor", values="value")
)
fig_7b, ax = plt.subplots(figsize=(8, 4))
heatmap_values = heatmap_frame.to_numpy(dtype=float)
masked = np.ma.masked_invalid(heatmap_values)
im = ax.imshow(masked, aspect="auto", cmap="viridis")
ax.set_xticks(np.arange(len(heatmap_frame.columns)))
ax.set_xticklabels(heatmap_frame.columns, rotation=45, ha="right")
ax.set_yticks(np.arange(len(heatmap_frame.index)))
ax.set_yticklabels(heatmap_frame.index)
avg_val = np.nanmean(heatmap_values)
for (i, j), value in np.ndenumerate(heatmap_values):
    if np.isnan(value):
        continue
    text_color = "white" if value > avg_val else "black"
    ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color)
ax.set_title("Figure 7B — Average sensor magnitude by zone")
fig_7b.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean value")
fig_7b.tight_layout()
fig_7b_path = FIG_DIR / "figure_7B_sensor_heatmap.png"
fig_7b.savefig(fig_7b_path, dpi=180)
plt.show()

# Figure 7C — RL policy convergence curves
display(Markdown("#### Figure 7C — RL policy convergence"))
rolling_window = 20
sorted_rewards = rewards_long.sort_values(["algorithm", "seed", "episode"])
smoothed_groups: list[pd.DataFrame] = []
for (_, _), group in sorted_rewards.groupby(["algorithm", "seed"]):
    smoothed = group.copy()
    smoothed["rolling_reward"] = smoothed["reward"].rolling(rolling_window, min_periods=1).mean()
    smoothed_groups.append(smoothed)
rewards_smoothed = pd.concat(smoothed_groups, ignore_index=True)
fig_7c, ax = plt.subplots(figsize=(8, 4))
for algorithm, group in rewards_smoothed.groupby("algorithm"):
    averaged = group.groupby("episode", as_index=False)["rolling_reward"].mean()
    ax.plot(averaged["episode"], averaged["rolling_reward"], label=f"{algorithm} (rolling {rolling_window})")
ax.set_title("Figure 7C — RL policy convergence curves")
ax.set_xlabel("Episode")
ax.set_ylabel("Rolling mean reward")
ax.legend()
fig_7c.tight_layout()
fig_7c_path = FIG_DIR / "figure_7C_rl_convergence.png"
fig_7c.savefig(fig_7c_path, dpi=180)
plt.show()


# ---------------------------------------------------------------------------
# 4. Summary of generated assets
# ---------------------------------------------------------------------------

display(Markdown("### Generated assets"))
display(Markdown("Tables saved under `smartbuildsim_colab_outputs/tables`"))
display(pd.DataFrame(sorted(TABLE_DIR.glob("*.csv")), columns=["table_path"]))
display(Markdown("Figures saved under `smartbuildsim_colab_outputs/figures`"))
display(pd.DataFrame(sorted(FIG_DIR.glob("*.png")), columns=["figure_path"]))

print("Demo complete. Figures and tables are stored locally for download from Colab.")
