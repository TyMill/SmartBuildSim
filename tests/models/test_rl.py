"""Reinforcement learning tests."""

from __future__ import annotations

import numpy as np

from smartbuildsim.models.rl import RLConfig, evaluate_policy, train_policy


def test_rl_training_converges() -> None:
    config = RLConfig(episodes=120, steps_per_episode=32, seed=5)
    result = train_policy(config)
    assert result.q_table.shape == (11, 3)
    avg_reward = result.average_reward()
    assert np.isfinite(avg_reward)
    eval_reward = evaluate_policy(result, episodes=20)
    assert eval_reward > -1.5
