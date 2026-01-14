# Parameterized Social Dilemma Game

A flexible framework for N-player social dilemma games in OpenSpiel with support for dynamic payoffs, stochastic rewards, and Axelrod-style bots.

## Features

- **Variable Number of Agents**: Support for N ≥ 2 players
- **Dynamic Payoff Matrices**: Payoffs can change over time or be generated from parameterized functions
- **Stochastic Rewards**: Optional reward noise (Gaussian, uniform, or discrete)
- **Axelrod-Style Bots**: Implementation of classic strategies (Tit-for-Tat, Grim Trigger, etc.)
- **Python API**: Full integration with OpenSpiel's Python interface

## Installation

The game is implemented in pure Python and automatically registered with OpenSpiel when imported.

## Basic Usage

### Creating a Game

```python
import pyspiel

# Basic 2-player prisoner's dilemma
game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 2,
    "num_actions": 2,
    "termination_probability": 0.125,
    "max_game_length": 9999
})

# 4-player game with custom settings
game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 4,
    "num_actions": 3,
    "termination_probability": 0.1,
    "max_game_length": 50
})
```

### Playing a Game

```python
state = game.new_initial_state()

while not state.is_terminal():
    # Get actions for each player
    actions = [player_policy(state) for player in range(game.num_players())]
    
    # Apply actions
    state.apply_actions(actions)
    
    # Handle chance node (game termination)
    if state.current_player() == pyspiel.PlayerId.CHANCE:
        # Apply chance outcome (0 = continue, 1 = stop)
        state.apply_action(0)  # Continue for this example
    
    # Check rewards
    rewards = state.rewards()
    returns = state.returns()
```

## Advanced Features

### Dynamic Payoffs

```python
def time_varying_payoffs(base_matrix, timestep):
    """Payoffs that change over time."""
    multiplier = 1 + 0.1 * timestep
    return [[cell * multiplier for cell in row] for row in base_matrix]

game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 2,
    "payoff_function": time_varying_payoffs
})
```

### Stochastic Rewards

```python
# Gaussian noise
game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 2,
    "reward_noise": {"type": "gaussian", "std": 0.1}
})

# Uniform noise
game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 2,
    "reward_noise": {"type": "uniform", "range": 0.2}
})

# Discrete noise
game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 2,
    "reward_noise": {"type": "discrete", "values": [-0.1, 0, 0.1]}
})
```

### Custom Payoff Matrices

```python
# Custom prisoner's dilemma
custom_payoff = [
    [[3, 0], [5, 1]],  # Player 0 payoffs
    [[3, 5], [0, 1]]   # Player 1 payoffs
]

game = pyspiel.load_game("python_param_social_dilemma", {
    "num_players": 2,
    "payoff_matrix": custom_payoff
})
```

## Axelrod-Style Bots

### Available Bot Types

- `always_cooperate`: Always chooses action 0 (cooperate)
- `always_defect`: Always chooses the last action (defect)
- `tit_for_tat`: Cooperates first, then copies opponent's last move
- `grim_trigger`: Cooperates until opponent defects, then defects forever
- `generous_tit_for_tat`: Like tit-for-tat but occasionally forgives defection
- `suspicious_tit_for_tat`: Like tit-for-tat but defects first
- `random`: Chooses actions uniformly at random
- `pavlov`: Win-stay, lose-shift strategy
- `adaptive`: Adapts based on opponent cooperation rates

### Using Bots

```python
from open_spiel.python.games.param_social_dilemma_bots import create_bot

# Create bots
bot1 = create_bot("tit_for_tat", 0, game)
bot2 = create_bot("always_defect", 1, game)

# Play with bots
state = game.new_initial_state()
while not state.is_terminal():
    actions = [bot1.step(state), bot2.step(state)]
    state.apply_actions(actions)
    
    if state.current_player() == pyspiel.PlayerId.CHANCE:
        state.apply_action(0)  # Continue
```

### Bot Tournament

```python
from open_spiel.python.games.param_social_dilemma_bots import get_available_bot_types

# Get all available bot types
bot_types = get_available_bot_types()

# Run a tournament
scores = {}
for bot1_type in bot_types:
    for bot2_type in bot_types:
        # Play multiple games and compare performance
        scores[(bot1_type, bot2_type)] = play_tournament_game(bot1_type, bot2_type)
```

## Game Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_players` | int | 2 | Number of players (≥ 2) |
| `num_actions` | int | 2 | Number of actions per player (≥ 2) |
| `payoff_matrix` | list | None | Custom payoff matrix (auto-generated if None) |
| `termination_probability` | float | 0.125 | Probability of game ending each round |
| `max_game_length` | int | 9999 | Maximum number of rounds |
| `payoff_function` | callable | None | Function for dynamic payoffs |
| `reward_noise` | dict | None | Noise configuration for rewards |
| `seed` | int | None | Random seed for reproducibility |

## Default Payoff Structure

For 2-player, 2-action games, the default payoff matrix follows the classic prisoner's dilemma:

```
          Player 1
          C   D
Player 0 C [3,3] [0,5]
         D [5,0] [1,1]
```

For N-player games, payoffs are structured to create a social dilemma where:
- Cooperation (action 0) provides moderate benefits to all
- Defection (action 1) provides higher personal payoff but reduces group benefits

## Observation Space

The game provides observations including:
- Action history for all players
- Current timestep (normalized)
- Full game state for perfect information settings

## Integration with RL Algorithms

The game integrates seamlessly with OpenSpiel's reinforcement learning algorithms:

```python
from open_spiel.python.algorithms import random_agent
from open_spiel.python.rl_environment import RLEnvironment

# Create RL environment
env_config = {"players": pyspiel.PlayerId.SIMULTANEOUS}
env = RLEnvironment(game, env_config)

# Use with RL algorithms
agent = random_agent.RandomAgent(player_id=0, num_actions=game.num_distinct_actions())
# ... train your agent
```

## Examples

See `open_spiel/python/examples/param_social_dilemma_example.py` for comprehensive examples including:
- Basic 2-player games
- N-player scenarios
- Dynamic payoffs
- Stochastic rewards
- Bot tournaments
- Custom payoff matrices

## Testing

Run the test suite:

```bash
python -m pytest open_spiel/python/games/param_social_dilemma_test.py
```

## Research Applications

This framework is designed to support modern MARL research on:

- **Multi-agent cooperation**: Study how agents learn to cooperate in groups
- **Social dilemmas**: Explore the tragedy of the commons and collective action problems
- **Dynamic environments**: Test robustness to non-stationary payoffs
- **Stochastic settings**: Evaluate performance under uncertainty
- **Strategy evolution**: Analyze emergence of cooperation vs competition

## Extending the Framework

### Custom Bot Strategies

```python
from open_spiel.python.games.param_social_dilemma_bots import SocialDilemmaBot

class MyCustomBot(SocialDilemmaBot):
    def step(self, state):
        # Implement your strategy here
        return action
```

### Custom Payoff Functions

```python
def my_payoff_function(base_matrix, timestep):
    # Implement your dynamic payoff logic
    return modified_matrix
```

## Citation

If you use this framework in your research, please cite:

```
@software{openspiel_param_social_dilemma,
  title={Parameterized Social Dilemma Games for OpenSpiel},
  author={OpenSpiel Contributors},
  year={2024},
  url={https://github.com/deepmind/open_spiel}
}
```
