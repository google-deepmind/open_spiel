# OpenSpiel state methods: history_str

[Back to Core API reference](../api_reference.md) \
<br>

`history_str()`

Returns a string representation of the action history, with actions separated by
commas. This is a convenience method equivalent to joining the elements of
`history()` with `", "`.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)
state.apply_action(0)
state.apply_action(8)

print(state.history_str())    # Output: 4, 0, 8
print(state.history())        # Output: [4, 0, 8]
```
