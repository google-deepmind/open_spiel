# OpenSpiel game methods: action_to_string

[Back to Core API reference](../api_reference.md) \
<br>

`action_to_string(player: int, action: int)`

Returns a string representation of the specified player's action, independent of
state.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("matrix_pd")
print(game.action_to_string(0, 0))
# Output: Cooperate

# Print first player's second action (1).
game = pyspiel.load_game("tic_tac_toe")
print(game.action_to_string(0, 1))
# Output: x(0, 1)
```
