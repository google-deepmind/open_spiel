# OpenSpiel state methods: is_terminal

[Back to Core API reference](../api_reference.md) \
<br>

`is_terminal()`

Returns True if the state is terminal (the game has ended), False otherwise.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.is_terminal())    # Output: False

game = pyspiel.load_game("matrix_rps")
state = game.new_initial_state()
print(state.is_terminal())    # Output: False
state.apply_actions([1, 1])
print(state.is_terminal())    # Output: True
```
