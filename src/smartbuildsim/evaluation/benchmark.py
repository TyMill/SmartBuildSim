"""Benchmark utilities for forecasting, anomaly detection, and RL."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..config import create_rng, resolve_seed
from ..features.engineering import build_supervised_matrix
from ..models.rl import RLConfig, RLTrainingResult, evaluate_policy, train_policy


class BenchmarkResultBase(TypedDict):
    """Common fields present in benchmark results."""

    metrics: pd.DataFrame
    significance: pd.DataFrame
    scaling: pd.DataFrame
    baseline: str


class RegressionBenchmarkResult(BenchmarkResultBase):
    """Return type for :func:`run_regression_benchmark`."""

    rmse_scores: dict[str, list[float]]


class AnomalyBenchmarkResult(BenchmarkResultBase):
    """Return type for :func:`run_anomaly_benchmark`."""

    f1_scores: dict[str, list[float]]
    precision_scores: dict[str, list[float]]
    recall_scores: dict[str, list[float]]


@dataclass
class BenchmarkSummary:
    """Summary of benchmark scores with statistics and significance tests."""

    metrics: pd.DataFrame
    significance: pd.DataFrame
    scaling_sensitivity: pd.DataFrame


def _metric_summary(
    scores: Mapping[str, list[float]],
    metric_name: str,
    baseline: str,
) -> BenchmarkSummary:
    """Return aggregated metrics, pairwise tests, and scaling sensitivity."""

    metrics = pd.DataFrame(
        {model: {"mean": np.mean(vals), "std": np.std(vals, ddof=1)} for model, vals in scores.items()}
    ).T
    comparisons = []
    base_scores = np.asarray(scores[baseline])
    for model, values in scores.items():
        if model == baseline:
            continue
        values_arr = np.asarray(values)
        t_stat, t_p = stats.ttest_rel(base_scores, values_arr, alternative="two-sided")
        try:
            w_stat, w_p = stats.wilcoxon(base_scores, values_arr, alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        comparisons.append(
            {
                "model": model,
                "t_stat": float(t_stat),
                "t_pvalue": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_pvalue": float(w_p),
            }
        )
    significance = pd.DataFrame(comparisons)
    scaling = pd.DataFrame()
    if metric_name in {"rmse", "f1"}:  # populated by higher-level helpers
        scaling = pd.DataFrame()
    return BenchmarkSummary(metrics=metrics, significance=significance, scaling_sensitivity=scaling)


def run_regression_benchmark(
    series: pd.Series,
    lags: Sequence[int],
    horizon: int,
    seeds: Sequence[int],
    scalers: Mapping[str, Callable[[], object]] | None = None,
) -> RegressionBenchmarkResult:
    """Evaluate forecasting models with repeated cross-validation."""

    frame = build_supervised_matrix(series, lags, horizon)
    features = [col for col in frame.columns if col.startswith("lag_")]
    target = f"lead_{horizon}"
    scalers = scalers or {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "none": lambda: None,
    }

    estimators: Mapping[str, Callable[[], object]] = {
        "linear": LinearRegression,
        "hist_gbr": HistGradientBoostingRegressor,
    }

    rmse_scores: dict[str, list[float]] = {f"{name}+{scaler_name}": [] for name in estimators for scaler_name in scalers}

    for index, seed in enumerate(seeds):
        splitter = KFold(
            n_splits=5,
            shuffle=True,
            random_state=resolve_seed(
                "evaluation.regression.kfold",
                explicit=seed,
                offset=index,
            ),
        )
        for train_idx, test_idx in splitter.split(frame):
            train = frame.iloc[train_idx]
            test = frame.iloc[test_idx]
            for estimator_name, estimator_factory in estimators.items():
                for scaler_name, scaler_factory in scalers.items():
                    steps = []
                    scaler = scaler_factory()
                    if scaler is not None:
                        steps.append(("scaler", scaler))
                    steps.append(("estimator", estimator_factory()))
                    pipeline = Pipeline(steps=steps)
                    pipeline.fit(train[features], train[target])
                    preds = pipeline.predict(test[features])
                    rmse = float(np.sqrt(mean_squared_error(test[target], preds)))
                    rmse_scores[f"{estimator_name}+{scaler_name}"].append(rmse)

    baseline_key = "linear+standard"
    summary = _metric_summary(rmse_scores, "rmse", baseline_key)
    scaling_rows = []
    for scaler_name in scalers:
        matching = [key for key in rmse_scores if key.endswith(f"+{scaler_name}")]
        values = [np.mean(rmse_scores[key]) for key in matching]
        scaling_rows.append(
            {
                "scaler": scaler_name,
                "mean_rmse": float(np.mean(values)),
                "std_rmse": float(np.std(values, ddof=1)),
            }
        )
    summary.scaling_sensitivity = pd.DataFrame(scaling_rows)
    return {
        "rmse_scores": rmse_scores,
        "metrics": summary.metrics,
        "significance": summary.significance,
        "scaling": summary.scaling_sensitivity,
        "baseline": baseline_key,
    }


def run_anomaly_benchmark(
    features: pd.DataFrame,
    labels: pd.Series,
    seeds: Sequence[int],
    scalers: Mapping[str, Callable[[], object]] | None = None,
) -> AnomalyBenchmarkResult:
    """Evaluate anomaly detectors using labelled synthetic anomalies."""

    feature_columns = [col for col in features.columns if col.startswith("lag_")]
    scalers = scalers or {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "none": lambda: None,
    }
    detectors: Mapping[str, Callable[[], object]] = {
        "isolation_forest": lambda: IsolationForest(random_state=0, contamination=0.05),
        "lof": lambda: LocalOutlierFactor(novelty=True, contamination=0.05),
    }
    f1_scores: dict[str, list[float]] = {f"{name}+{scaler_name}": [] for name in detectors for scaler_name in scalers}
    precision_scores: dict[str, list[float]] = {
        f"{name}+{scaler_name}": [] for name in detectors for scaler_name in scalers
    }
    recall_scores: dict[str, list[float]] = {
        f"{name}+{scaler_name}": [] for name in detectors for scaler_name in scalers
    }

    for index, seed in enumerate(seeds):
        splitter = KFold(
            n_splits=5,
            shuffle=True,
            random_state=resolve_seed(
                "evaluation.anomaly.kfold",
                explicit=seed,
                offset=index,
            ),
        )
        for train_idx, test_idx in splitter.split(features):
            train = features.iloc[train_idx]
            test = features.iloc[test_idx]
            y_test = labels.iloc[test_idx]
            for detector_name, detector_factory in detectors.items():
                for scaler_name, scaler_factory in scalers.items():
                    steps = []
                    scaler = scaler_factory()
                    if scaler is not None:
                        steps.append(("scaler", scaler))
                    steps.append(("detector", detector_factory()))
                    pipeline = Pipeline(steps=steps)
                    pipeline.fit(train[feature_columns])
                    predictions = pipeline.predict(test[feature_columns])
                    is_anomaly = predictions == -1
                    f1_scores[f"{detector_name}+{scaler_name}"].append(
                        f1_score(y_test, is_anomaly, zero_division=0)
                    )
                    precision_scores[f"{detector_name}+{scaler_name}"].append(
                        precision_score(y_test, is_anomaly, zero_division=0)
                    )
                    recall_scores[f"{detector_name}+{scaler_name}"].append(
                        recall_score(y_test, is_anomaly, zero_division=0)
                    )

    baseline_key = "isolation_forest+standard"
    summary = _metric_summary(f1_scores, "f1", baseline_key)
    scaling_rows = []
    for scaler_name in scalers:
        matching = [key for key in f1_scores if key.endswith(f"+{scaler_name}")]
        values = [np.mean(f1_scores[key]) for key in matching]
        scaling_rows.append(
            {
                "scaler": scaler_name,
                "mean_f1": float(np.mean(values)),
                "std_f1": float(np.std(values, ddof=1)),
            }
        )
    summary.scaling_sensitivity = pd.DataFrame(scaling_rows)
    return {
        "f1_scores": f1_scores,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "metrics": summary.metrics,
        "significance": summary.significance,
        "scaling": summary.scaling_sensitivity,
        "baseline": baseline_key,
    }


class SoftQLearningConfig(RLConfig):
    """Configuration for entropy-regularised Q-learning (SAC-inspired)."""

    temperature: float = Field(default=0.05, gt=0.0)
    policy_smoothing: float = Field(default=0.05, ge=0.0, le=1.0)


@dataclass
class RLBenchmarkResult:
    """Container storing RL benchmark statistics."""

    metrics: pd.DataFrame
    significance: pd.DataFrame


def train_soft_q_policy(config: SoftQLearningConfig) -> RLTrainingResult:
    """Train a discrete entropy-regularised policy inspired by SAC."""

    rng = create_rng("evaluation.rl.soft.train", explicit=config.seed)
    n_states = 11
    n_actions = 3
    q_table = np.zeros((n_states, n_actions))
    log_policy = np.full((n_states, n_actions), np.log(1.0 / n_actions))
    rewards: list[float] = []
    for _ in range(config.episodes):
        temperature = float(rng.normal(22.0, 1.5))
        state = int(np.clip(round(temperature - config.target_temperature), -5, 5) + 5)
        episode_reward = 0.0
        for _ in range(config.steps_per_episode):
            probs = np.exp(log_policy[state])
            probs /= probs.sum()
            action = int(rng.choice(n_actions, p=probs))
            if action == 0:
                temperature += float(rng.normal(0.0, 0.5))
            elif action == 1:
                temperature -= 0.8 + float(rng.normal(0.0, 0.3))
            else:
                temperature += 0.8 + float(rng.normal(0.0, 0.3))
            next_state = int(np.clip(round(temperature - config.target_temperature), -5, 5) + 5)
            deviation = abs(temperature - config.target_temperature)
            reward = (1.0 - deviation) - (0.15 if action != 0 else 0.0)
            episode_reward += reward
            logsumexp = np.log(np.sum(np.exp(log_policy[next_state])))
            target = reward + config.discount * (logsumexp * config.temperature)
            q_table[state, action] = (1 - config.learning_rate) * q_table[state, action] + config.learning_rate * target
            logits = q_table[state] / max(config.temperature, 1e-6)
            logits = (1 - config.policy_smoothing) * log_policy[state] + config.policy_smoothing * logits
            log_policy[state] = logits - np.log(np.sum(np.exp(logits)))
            state = next_state
        rewards.append(episode_reward)
    return RLTrainingResult(q_table=q_table, reward_history=rewards, config=config)


def run_rl_benchmark(
    base_config: RLConfig,
    soft_config: SoftQLearningConfig,
    seeds: Iterable[int],
) -> RLBenchmarkResult:
    """Compare Q-learning with entropy-regularised soft Q-learning across seeds."""

    baseline_scores: list[float] = []
    soft_scores: list[float] = []
    for index, seed in enumerate(seeds):
        base_conf = base_config.model_copy(
            update={
                "seed": resolve_seed(
                    "evaluation.rl.q_learning",
                    explicit=seed,
                    offset=index,
                )
            }
        )
        result = train_policy(base_conf)
        baseline_scores.append(evaluate_policy(result))

        soft_conf = soft_config.model_copy(
            update={
                "seed": resolve_seed(
                    "evaluation.rl.soft",
                    explicit=seed,
                    offset=index,
                )
            }
        )
        soft_result = train_soft_q_policy(soft_conf)
        soft_scores.append(evaluate_policy(soft_result))

    metrics = pd.DataFrame(
        {
            "q_learning": {"mean": np.mean(baseline_scores), "std": np.std(baseline_scores, ddof=1)},
            "soft_q": {"mean": np.mean(soft_scores), "std": np.std(soft_scores, ddof=1)},
        }
    ).T
    t_stat, t_p = stats.ttest_rel(baseline_scores, soft_scores, alternative="two-sided")
    w_stat, w_p = stats.wilcoxon(baseline_scores, soft_scores, alternative="two-sided")
    significance = pd.DataFrame(
        [
            {
                "model": "soft_q",
                "t_stat": float(t_stat),
                "t_pvalue": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_pvalue": float(w_p),
            }
        ]
    )
    return RLBenchmarkResult(metrics=metrics, significance=significance)


__all__ = [
    "BenchmarkSummary",
    "RLBenchmarkResult",
    "SoftQLearningConfig",
    "run_anomaly_benchmark",
    "run_regression_benchmark",
    "run_rl_benchmark",
    "train_soft_q_policy",
]
