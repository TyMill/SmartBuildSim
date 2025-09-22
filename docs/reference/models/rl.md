# Reinforcement Learning

`smartbuildsim.models.rl` implements a compact tabular Q-learning agent for
thermostat control experiments.

## Configuration

`RLConfig` options:

- `episodes` and `steps_per_episode` control training duration.
- `learning_rate`, `discount`, and `epsilon` tune the learning dynamics.
- `target_temperature` defines the comfort set point around which the discrete
  state space is centred.
- `seed` ensures deterministic training runs.

## Training and evaluation

- `train_policy(config)` initialises a Q-table for 11 temperature-derived states
  and three actions (hold, cool, heat), then iterates for the configured number
  of episodes, returning an `RLTrainingResult`.
- `RLTrainingResult.average_reward(last_n=50)` summarises performance.
- `evaluate_policy(result, episodes=50)` rolls out the greedy policy to estimate
  long-term reward.

## CLI usage

```bash
smartbuildsim rl train examples/configs/default.yaml
```

The command writes the learned Q-table to `outputs/rl_q_table.npy` and prints
both training and evaluation rewards, mirroring the reinforcement learning stage
in `examples/scripts/run_example.py`.

## Python example

```python
from smartbuildsim.models.rl import RLConfig, evaluate_policy, train_policy
from smartbuildsim.scenarios.presets import get_scenario

scenario = get_scenario("office-small")
config = RLConfig(**scenario.rl.dict())
result = train_policy(config)

print(f"Average reward (last 50): {result.average_reward():.3f}")
print(f"Evaluation reward: {evaluate_policy(result):.3f}")
```

Combine the resulting metrics with the forecasting and anomaly outputs for a
holistic evaluation of an experiment.
