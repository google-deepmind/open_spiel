# OpenSpiel game methods: deserialize_state

[Back to Core API reference](../api_reference.md) \
<br>

`deserialize_state(serialized_data: str)`

Reconstruct a state object from the state's serialized data (from
`state.serialize()`). The game used to reconstruct must be the same as the game
that created the original state.

To serialize a state along with the game, use `pyspiel.serialize_game_and_state`
instead.

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
