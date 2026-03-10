# OpenSpiel state methods: player_reward

[Back to Core API reference](../api_reference.md) \
<br>

`player_reward(player: int)`

Returns the reward for the specified player from the most recent state
transition. This is useful for RL-style games with intermediate rewards.
For games that only have terminal rewards, this returns 0 for non-terminal
states. This is a single-player convenience method; if you need rewards for all
players, use `rewards()` instead as it is more efficient.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)
print(state.player_reward(0))    # Output: 0.0

# Play to terminal
state.apply_action(1)
state.apply_action(2)
state.apply_action(5)
state.apply_action(6)
print(state.player_reward(0))    # Output: 1.0
print(state.player_reward(1))    # Output: -1.0
```
