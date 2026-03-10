# OpenSpiel state methods: is_initial_state

[Back to Core API reference](../api_reference.md) \
<br>

`is_initial_state()`

Returns `True` if this is the initial state of the game (the root node of the
game tree), i.e. no actions have been taken yet. Returns `False` otherwise.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.is_initial_state())    # Output: True

state.apply_action(4)
print(state.is_initial_state())    # Output: False

game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()
print(state.is_initial_state())    # Output: True (even though it's a chance node)
```
