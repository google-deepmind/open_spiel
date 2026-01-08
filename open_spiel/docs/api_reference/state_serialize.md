# OpenSpiel state methods: serialize

[Back to Core API reference](../api_reference.md) \
<br>

`serialize()`

Returns a string representation of the state be used to reconstruct the state.
By default, it is a string list of each action taken in the history.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)
state.apply_action(2)
state.apply_action(1)
state.apply_action(5)

state_copy = game.deserialize_state(state.serialize())
print(state_copy)

# Output:
# .xo
# .xo
# ...
```
