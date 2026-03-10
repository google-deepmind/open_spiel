# OpenSpiel state methods: is_player_node

[Back to Core API reference](../api_reference.md) \
<br>

`is_player_node()`

Returns `True` if the current state is a decision node where a single player
acts (i.e. `current_player() >= 0`). Returns `False` for chance nodes,
simultaneous nodes, mean field nodes, and terminal states.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.is_player_node())      # Output: True
print(state.is_chance_node())      # Output: False

game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()
print(state.is_player_node())      # Output: False (initial state is a chance node)
print(state.is_chance_node())      # Output: True
```
