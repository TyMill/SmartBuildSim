"""Simple reinforcement learning utilities for HVAC control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from ..config import create_rng


class RLConfig(BaseModel):
    """Configuration for Q-learning training."""

    episodes: int = Field(default=200, gt=0)
    steps_per_episode: int = Field(default=48, gt=0)
    learning_rate: float = Field(default=0.1, gt=0.0)
    discount: float = Field(default=0.95, gt=0.0, le=1.0)
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = 21
    target_temperature: float = Field(default=22.0)


@dataclass
class RLTrainingResult:
    """Result of RL training."""

    q_table: np.ndarray
    reward_history: list[float]
    config: RLConfig

    def average_reward(self, last_n: int = 50) -> float:
        """Return the rolling average reward over the final ``last_n`` episodes."""

        window = self.reward_history[-last_n:]
        return float(np.mean(window)) if window else 0.0


def _initial_state(rng: np.random.Generator) -> float:
    """Sample an initial indoor temperature."""
    return float(rng.normal(22.0, 1.5))


def _state_index(temperature: float, target: float) -> int:
    """Discretise the temperature difference around the target set point."""
    diff = int(round(temperature - target))
    diff = max(-5, min(5, diff))
    return diff + 5


def _transition(temperature: float, action: int, rng: np.random.Generator) -> float:
    """Apply the HVAC action dynamics to transition to a new state."""
    if action == 0:  # hold
        temperature += float(rng.normal(0.0, 0.5))
    elif action == 1:  # cool
        temperature -= 0.8 + float(rng.normal(0.0, 0.3))
    else:  # heat
        temperature += 0.8 + float(rng.normal(0.0, 0.3))
    return float(temperature)


def _reward(temperature: float, target: float, action: int) -> float:
    """Return the immediate reward given the current state and action."""
    deviation = abs(temperature - target)
    comfort_reward = 1.0 - deviation
    energy_penalty = 0.15 if action != 0 else 0.0
    return comfort_reward - energy_penalty


def train_policy(config: RLConfig) -> RLTrainingResult:
    """Train a Q-learning policy for simple thermostat control."""

    rng = create_rng("models.rl.q_learning.train", explicit=config.seed)
    n_states = 11  # discretised temperature difference from -5..5
    n_actions = 3  # hold, cool, heat
    q_table = np.zeros((n_states, n_actions))
    rewards: list[float] = []
    for _ in range(config.episodes):
        temperature = _initial_state(rng)
        state = _state_index(temperature, config.target_temperature)
        episode_reward = 0.0
        for _ in range(config.steps_per_episode):
            if rng.random() < config.epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(q_table[state]))
            next_temp = _transition(temperature, action, rng)
            next_state = _state_index(next_temp, config.target_temperature)
            reward = _reward(next_temp, config.target_temperature, action)
            episode_reward += reward
            best_next = float(np.max(q_table[next_state]))
            q_table[state, action] = (1 - config.learning_rate) * q_table[state, action] + config.learning_rate * (
                reward + config.discount * best_next
            )
            temperature = next_temp
            state = next_state
        rewards.append(episode_reward)
    return RLTrainingResult(q_table=q_table, reward_history=rewards, config=config)


def evaluate_policy(result: RLTrainingResult, episodes: int = 50) -> float:
    """Evaluate the greedy policy derived from the Q-table."""

    rng = create_rng(
        "models.rl.q_learning.eval",
        explicit=result.config.seed,
        offset=1,
    )
    total_reward = 0.0
    for _ in range(episodes):
        temperature = _initial_state(rng)
        for _ in range(result.config.steps_per_episode):
            state = _state_index(temperature, result.config.target_temperature)
            action = int(np.argmax(result.q_table[state]))
            temperature = _transition(temperature, action, rng)
            total_reward += _reward(
                temperature, result.config.target_temperature, action
            )
    return total_reward / (episodes * result.config.steps_per_episode)


__all__ = ["RLConfig", "RLTrainingResult", "evaluate_policy", "train_policy"]