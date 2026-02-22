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
    "num_players": 3,
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
    "num_players": 2,
    "payoff_matrix": custom_payoff.tolist()
})
```

### Stochastic Rewards

```python
game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 2,
    "reward_noise_std": 0.5
})
```

### Dynamic Payoffs

```python
game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 4,
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

---

## Implementation Summary

### Overview

This implementation addresses GitHub issue for adding a parameterized social dilemma game to OpenSpiel. The implementation provides a flexible N-player simultaneous-move game that supports variable agent counts, dynamic payoff matrices, and stochastic rewards.

### Files Created/Modified

#### Core Implementation

1. **`open_spiel/python/games/param_social_dilemma.py`** (Already exists)
   - Main game implementation
   - Inherits from `pyspiel.Game` and `pyspiel.State`
   - Supports N-player games (N ≥ 2)
   - Dynamic payoff matrices with optional noise
   - Stochastic rewards via configurable noise parameter

2. **`open_spiel/python/games/param_social_dilemma_test.py`** (Already exists)
   - Comprehensive unit tests
   - Tests default parameters, custom payoff matrices
   - Tests stochastic rewards and dynamic payoffs
   - Tests game progression and returns accumulation

#### New Files Created

3. **`open_spiel/python/games/param_social_dilemma_bots.py`** (NEW)
   - Axelrod-style bots implementation
   - Bots included:
     - `AlwaysCooperateBot`
     - `AlwaysDefectBot`
     - `TitForTatBot`
     - `GrimTriggerBot`
     - `PavlovBot` (win-stay, lose-shift)
     - `TitForTwoTatsBot`
     - `GradualBot`

4. **`open_spiel/python/games/param_social_dilemma_bots_test.py`** (NEW)
   - Unit tests for all bot strategies
   - Tests bot behavior in various scenarios
   - Tests bot interactions in games

5. **`open_spiel/python/games/param_social_dilemma_README.md`** (NEW)
   - Comprehensive documentation
   - Usage examples
   - Configuration parameter reference
   - Bot descriptions

#### Examples

6. **`open_spiel/python/examples/param_social_dilemma_example.py`** (Already exists)
   - Demonstrates basic game usage
   - Shows stochastic rewards
   - Shows dynamic payoffs
   - Custom payoff matrix examples
   - Scalability tests with varying player counts

7. **`open_spiel/python/examples/param_social_dilemma_bots_example.py`** (NEW)
   - Axelrod-style tournament implementation
   - Bot-vs-bot competition with scoring matrix
   - N-player tournament scenarios
   - Demonstrates bot strategies in practice

#### Registration

8. **`open_spiel/python/games/__init__.py`** (Already updated)
   - Game is already registered in the imports

### Features Implemented

#### ✅ Variable Number of Agents (N-Player)
- Configurable agent count (N ≥ 2)
- Default: 3 players
- Tested with 2, 3, 5, and 8 players
- Compatible with `SimultaneousMoveGame` API

#### ✅ Dynamic Payoff Matrices
- Payoffs can change across timesteps
- Controlled by `dynamic_payoffs` and `payoff_change_prob` parameters
- Enables non-stationary environment experiments

#### ✅ Stochastic Rewards
- Optional Gaussian noise with configurable standard deviation
- Controlled by `reward_noise_std` parameter
- Useful for robustness studies

#### ✅ Python API Exposure
All parameters exposed via Python interface:
- `num_players`: Number of agents
- `num_actions`: Actions per agent
- `max_game_length`: Maximum timesteps
- `payoff_matrix`: Custom payoff structure
- `reward_noise_std`: Reward noise level
- `dynamic_payoffs`: Enable dynamic payoffs
- `payoff_change_prob`: Probability of payoff changes

#### ✅ Axelrod-Style Bots
Seven classic strategies implemented:
1. Always Cooperate
2. Always Defect
3. Tit-for-Tat
4. Grim Trigger
5. Pavlov (Win-Stay, Lose-Shift)
6. Tit-for-Two-Tats
7. Gradual

### Game Type Details
- **Dynamics**: SIMULTANEOUS
- **Chance Mode**: DETERMINISTIC (or EXPLICIT_STOCHASTIC with noise)
- **Information**: PERFECT_INFORMATION
- **Utility**: GENERAL_SUM
- **Reward Model**: REWARDS

### Default Payoff Structure Details
Public goods game formulation:
- Cooperators receive: 3.0 × (cooperators / total_players)
- Defectors receive: 5.0 × (cooperators / total_players)

This creates a social dilemma where individual incentive conflicts with collective benefit.

### Testing Coverage
- Core game tests: 13 test cases
- Bot tests: 8 test cases
- All tests use `absltest` framework
- Tests cover edge cases and various configurations

### Usage Examples

#### Basic Game
```python
import pyspiel

game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 3,
    "max_game_length": 10
})

state = game.new_initial_state()
state.apply_actions([0, 1, 0])
rewards = state.rewards()
```

#### With Bots
```python
from open_spiel.python.games import param_social_dilemma_bots

bot = param_social_dilemma_bots.TitForTatBot(player_id=0, num_players=2)
action = bot.step(state)
```

#### Tournament
```python
python3 open_spiel/python/examples/param_social_dilemma_bots_example.py
```

### Compliance with OpenSpiel Standards

✅ Follows OpenSpiel game structure
✅ Uses `pyspiel.Game` and `pyspiel.State` base classes
✅ Implements required methods: `current_player()`, `_legal_actions()`, `_apply_actions()`, etc.
✅ Proper game registration with `pyspiel.register_game()`
✅ Observer implementation for game state observation
✅ Comprehensive test coverage
✅ Example scripts for demonstration
✅ Documentation provided

### Differences from Original Plan

The original issue suggested placing implementation in `open_spiel/games/param_social_dilemma` (C++), but following the maintainer's guidance, this was implemented entirely in Python under `open_spiel/python/games/` for:
- Faster iteration
- Easier parameter configuration
- Better integration with Python-based MARL experiments
- Simpler maintenance

### Future Extensions

Possible enhancements:
1. Add more sophisticated bot strategies
2. Implement tournament ranking systems
3. Add visualization tools for game dynamics
4. Support for asymmetric payoff matrices
5. Integration with learning algorithms
6. Additional social dilemma variants (e.g., public goods with punishment)

### Additional References

- OpenSpiel Developer Guide
- Axelrod's Evolution of Cooperation
- Iterated Prisoner's Dilemma implementation in OpenSpiel
- Multi-agent reinforcement learning benchmarks
