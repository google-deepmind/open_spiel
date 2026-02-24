# Parameterized Social Dilemma

## Overview

The Parameterized Social Dilemma is a flexible N-player simultaneous-move game designed for multi-agent reinforcement learning (MARL) research. It generalizes traditional 2-player matrix games to support variable numbers of agents, dynamic payoff structures, and stochastic rewards.

## Features

### 1. Variable Number of Agents (N-Player)
- Support for N ≥ 2 agents
- Default: 3 players
- Compatible with OpenSpiel's simultaneous-move game API

### 2. Dynamic Payoff Matrices
- Payoff matrices can change across timesteps
- Generated from parameterized functions
- Enables experiments on non-stationary environments

### 3. Stochastic Rewards
- Optional reward noise with configurable standard deviation
- Useful for robustness and exploration studies

## Usage

### Basic Example

```python
import pyspiel

game = pyspiel.load_game("python_param_social_dilemma", {
    "players": 3,
    "max_game_length": 10
})

state = game.new_initial_state()
state.apply_actions([0, 1, 0])
rewards = state.rewards()
```

### Configuration Parameters

- `num_players` (int): Number of agents (default: 3)
- `num_actions` (int): Number of actions per agent (default: 2)
- `max_game_length` (int): Maximum timesteps (default: 10)
- `payoff_matrix` (array): Custom payoff structure (default: auto-generated)
- `reward_noise_std` (float): Standard deviation of reward noise (default: 0.0)
- `dynamic_payoffs` (bool): Enable dynamic payoff changes (default: False)
- `payoff_change_prob` (float): Probability of payoff change per timestep (default: 0.0)

### Custom Payoff Matrix Example

```python
import numpy as np

custom_payoff = np.zeros((2, 2, 2))
custom_payoff[0, 0] = [3, 3]
custom_payoff[0, 1] = [0, 5]
custom_payoff[1, 0] = [5, 0]
custom_payoff[1, 1] = [1, 1]

game = pyspiel.load_game("python_param_social_dilemma", {
    "players": 2,
    "payoff_matrix": custom_payoff.tolist()
})
```

### Stochastic Rewards

```python
game = pyspiel.load_game("python_param_social_dilemma", {
    "players": 2,
    "reward_noise_std": 0.5
})
```

### Dynamic Payoffs

```python
game = pyspiel.load_game("python_param_social_dilemma", {
    "players": 4,
    "dynamic_payoffs": True,
    "payoff_change_prob": 0.1
})
```

## Axelrod-Style Bots

The implementation includes well-known Axelrod tournament strategies:

- **AlwaysCooperateBot**: Always cooperates
- **AlwaysDefectBot**: Always defects
- **TitForTatBot**: Mimics opponent's last action
- **GrimTriggerBot**: Cooperates until opponent defects, then defects forever
- **PavlovBot**: Win-stay, lose-shift strategy
- **TitForTwoTatsBot**: Defects only after two consecutive defections
- **GradualBot**: Punishes defections with increasing severity

### Bot Usage

```python
from open_spiel.python.games import param_social_dilemma_bots

bot = param_social_dilemma_bots.TitForTatBot(player_id=0, num_players=2)
action = bot.step(state)
```

## Default Payoff Structure

The default payoff matrix implements a public goods game:
- Cooperating players receive: 3.0 × (cooperators / total_players)
- Defecting players receive: 5.0 × (cooperators / total_players)

This creates a social dilemma where individual incentive (defecting) conflicts with collective benefit (cooperating).

## Testing

Run the test suite:

```bash
python -m pytest open_spiel/python/games/param_social_dilemma_test.py
```

## Examples

See `open_spiel/python/examples/param_social_dilemma_example.py` for comprehensive usage examples including:
- Basic multi-player games
- Stochastic reward scenarios
- Dynamic payoff experiments
- Custom payoff matrices
- Scalability tests

## Implementation Details

- **Base Class**: `pyspiel.Game` and `pyspiel.State`
- **Game Type**: Simultaneous-move
- **Dynamics**: SIMULTANEOUS
- **Chance Mode**: DETERMINISTIC (or EXPLICIT_STOCHASTIC with noise)
- **Information**: PERFECT_INFORMATION
- **Utility**: GENERAL_SUM

## References

This implementation is inspired by:
- Axelrod's Iterated Prisoner's Dilemma tournaments
- Public goods games from behavioral economics
- Multi-agent reinforcement learning benchmarks
