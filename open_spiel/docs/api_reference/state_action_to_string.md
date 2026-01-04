# OpenSpiel state methods: action_to_string

[Back to Core API reference](../api_reference.md) \
<br>

`action_to_string(player: int, action: int)`

Returns a string representation of the specified player's action.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("breakthrough")
state = game.new_initial_state()
player = state.current_player()
for action in state.legal_actions():
  print(state.action_to_string(player, action))
```
