# Parameterized Social Dilemma Game (C++ Implementation)

A high-performance C++ implementation of parameterized social dilemma games for OpenSpiel, supporting N-player scenarios with dynamic payoffs and stochastic rewards.

## Overview

This implementation provides a flexible framework for social dilemma games that addresses the limitations of fixed 2-player matrix games by supporting:

- **Variable Number of Agents**: N ≥ 2 players (configurable up to 10)
- **Dynamic Payoff Matrices**: Time-varying payoff structures
- **Stochastic Rewards**: Multiple noise types for robustness testing
- **High Performance**: Optimized C++ implementation for large-scale simulations

## Game Registration

The C++ implementation is registered as `"param_social_dilemma"` in OpenSpiel.

```cpp
#include "open_spiel/games/param_social_dilemma.h"

// Game is automatically registered when compiled
```

## Usage

### Basic Game Creation

```cpp
// Create a 2-player game
GameParameters params;
params.Set("num_players", 2);
params.Set("num_actions", 2);
params.Set("termination_probability", 0.125);
params.Set("max_game_length", 9999);

std::shared_ptr<const Game> game = LoadGame("param_social_dilemma", params);
```

### Python Interface

```python
import pyspiel

# Basic 2-player game
game = pyspiel.load_game("param_social_dilemma", {
    "num_players": 2,
    "num_actions": 2,
    "termination_probability": 0.125,
    "max_game_length": 9999
})
```

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|--------|-------------|
| `num_players` | int | 2 | [2, 10] | Number of players |
| `num_actions` | int | 2 | [2, ∞) | Actions per player |
| `payoff_matrix` | double[] | auto-generated | - | Custom payoff matrix |
| `termination_probability` | double | 0.125 | [0, 1] | Game ending chance per round |
| `max_game_length` | int | 9999 | [1, ∞) | Maximum number of rounds |
| `reward_noise_std` | double | 0.0 | [0, ∞) | Standard deviation for reward noise |
| `reward_noise_type` | string | "none" | - | Type: "gaussian", "uniform", "discrete", "none" |
| `seed` | int | -1 | - | Random seed (-1 = random) |

## Default Payoff Structure

### 2-Player, 2-Action (Prisoner's Dilemma)

```
          Player 1
          C   D
Player 0 C [3,3] [0,5]
         D [5,0] [1,1]
```

### N-Player Games

For N > 2 or action count > 2, payoffs are generated using a social dilemma formula:
- **Cooperation** (action 0): `2.0 + num_cooperators * 0.5`
- **Defection** (action 1): `4.0 + num_cooperators * 0.2`

This creates a tension between individual and collective rationality.

## Advanced Features

### Stochastic Rewards

```cpp
// Gaussian noise
params.Set("reward_noise_std", 0.1);
params.Set("reward_noise_type", "gaussian");

// Uniform noise
params.Set("reward_noise_std", 0.1);
params.Set("reward_noise_type", "uniform");

// Discrete noise
params.Set("reward_noise_std", 0.1);
params.Set("reward_noise_type", "discrete");
```

### Custom Payoff Matrices

```cpp
// Custom 2x2 payoff matrix
std::vector<double> custom_payoff = {
    10, 0,  // Player 0: C-C, C-D
    20, 5,  // Player 0: D-C, D-D
    10, 20, // Player 1: C-C, D-C
    0, 5    // Player 1: C-D, D-D
};

params.Set("payoff_matrix", custom_payoff);
```

## Game Mechanics

### State Progression

1. **Initial State**: Simultaneous move node
2. **Action Application**: Players choose actions simultaneously
3. **Reward Calculation**: Payoffs determined by payoff matrix + noise
4. **Chance Node**: Game termination decision
5. **Termination**: Game ends or continues to next round

### Information Structure

- **Perfect Information**: Full action history available
- **Observation Tensor**: Encodes action history + current iteration
- **Information State**: String representation of action history

## Performance Characteristics

The C++ implementation is optimized for:

- **Memory Efficiency**: Compact state representation
- **Speed**: Fast action application and state cloning
- **Scalability**: Handles up to 10 players efficiently
- **Thread Safety**: Safe for concurrent simulations

## Integration Examples

### C++ Bot Integration

```cpp
class MyBot : public Bot {
 public:
  Action Step(const State& state) override {
    const auto* psd_state = down_cast<const ParamSocialDilemmaState*>(&state);
    // Implement strategy using psd_state->action_history()
    return action;
  }
};
```

### Python Bot Integration

```python
from open_spiel.python.games.param_social_dilemma_bots import create_bot

# Create and use bots
bot = create_bot("tit_for_tat", 0, game)
action = bot.step(state)
```

## Testing

### Unit Tests

```bash
# Run C++ tests
./open_spiel/games/param_social_dilemma/param_social_dilemma_test

# Run Python comparison tests
python -m pytest open_spiel/python/games/param_social_dilemma_test.py
```

### Performance Benchmarks

```python
# See performance demo
python open_spiel/python/examples/cpp_param_social_dilemma_demo.py
```

## File Structure

```
open_spiel/games/param_social_dilemma/
├── param_social_dilemma.h              # Header file
├── param_social_dilemma.cc             # Implementation
└── param_social_dilemma_test.cc        # Unit tests

open_spiel/python/games/
├── param_social_dilemma.py              # Python implementation
├── param_social_dilemma_bots.py          # Axelrod-style bots
├── param_social_dilemma_test.py         # Python tests
└── param_social_dilemma_README.md       # Python docs

open_spiel/python/examples/
├── cpp_param_social_dilemma_demo.py   # C++ demo
└── param_social_dilemma_example.py       # General examples
```

## Research Applications

### Multi-Agent Reinforcement Learning

```cpp
// Create environment for MARL
auto game = LoadGame("param_social_dilemma", params);
auto env = MakeRLEnvironment(game);

// Train multiple agents simultaneously
std::vector<std::unique_ptr<Agent>> agents;
for (int i = 0; i < num_players; ++i) {
  agents.push_back(CreateAgent(i));
}

// Training loop
for (int episode = 0; episode < num_episodes; ++episode) {
  auto state = game->NewInitialState();
  while (!state->IsTerminal()) {
    std::vector<Action> actions;
    for (int i = 0; i < num_players; ++i) {
      actions.push_back(agents[i]->Step(*state));
    }
    state->ApplyActions(actions);
    
    // Update agents with rewards
    auto rewards = state->Rewards();
    for (int i = 0; i < num_players; ++i) {
      agents[i]->Learn(rewards[i]);
    }
  }
}
```

### Game Theory Experiments

```cpp
// Test equilibrium concepts
auto game = LoadGame("param_social_dilemma", params);
auto solver = CreateCFRSolver(game);

// Compute and analyze equilibria
auto equilibrium = solver->Solve();
AnalyzeEquilibrium(equilibrium);
```

## Comparison with Python Implementation

| Feature | C++ Version | Python Version |
|---------|-------------|---------------|
| Performance | High (compiled) | Moderate (interpreted) |
| Flexibility | Good | Excellent (dynamic features) |
| Ease of Modification | Requires recompilation | Runtime changes |
| Memory Usage | Low | Higher |
| Integration | Native C++ OpenSpiel | Python OpenSpiel |
| Debugging | C++ tools | Python tools |

## Compilation

The game is automatically included when building OpenSpiel with CMake:

```bash
cd open_spiel
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Extension Points

### Custom Payoff Functions

For dynamic payoffs, extend the `CreateDefaultPayoffMatrix` method or use the Python version.

### Additional Noise Types

Add new noise types in the `AddNoise` method:

```cpp
double ParamSocialDilemmaState::AddNoise(double base_reward) {
  if (reward_noise_type_ == "my_custom_noise") {
    return base_reward + MyCustomNoise(rng_);
  }
  // ... existing implementations
}
```

### Custom Observations

Override observation methods for specialized information:

```cpp
void ParamSocialDilemmaState::ObservationTensor(
    Player player, absl::Span<float> values) const {
  // Custom observation encoding
  EncodeCustomObservation(player, values);
}
```

## Citation

If you use this implementation in research, please cite:

```bibtex
@software{openspiel_param_social_dilemma_cpp,
  title={Parameterized Social Dilemma Games for OpenSpiel (C++ Implementation)},
  author={OpenSpiel Contributors},
  year={2024},
  url={https://github.com/deepmind/open_spiel}
}
```

## Contributing

When contributing to the C++ implementation:

1. **Follow OpenSpiel conventions** for code style and structure
2. **Add comprehensive tests** for new features
3. **Update documentation** for parameter changes
4. **Ensure performance** is maintained
5. **Test both implementations** for compatibility

The C++ implementation provides the foundation for high-performance MARL research while maintaining full compatibility with OpenSpiel's ecosystem.
