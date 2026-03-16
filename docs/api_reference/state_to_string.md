# OpenSpiel state methods: to_string

[Back to Core API reference](../api_reference.md) \
<br>

`to_string()`

Returns a human-readable string representation of the state. This is the same as
calling `str(state)` in Python. Two states are considered equal if their
`to_string()` outputs match (this is the default equality comparison).

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)
state.apply_action(0)

print(state.to_string())
# Output:
# o..
# .x.
# ...

# Equivalent to str(state)
print(str(state) == state.to_string())    # Output: True
```
