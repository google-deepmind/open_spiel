# OpenSpiel state methods: child

[Back to Core API reference](../api_reference.md) \
<br>

`child(action: int)`

Returns a new state that results from applying the specified action to a clone
of the current state. The original state is not modified. This is equivalent to
cloning the state and then calling `apply_action()` on the clone.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

# Create a child state by taking the center square
child_state = state.child(4)
print(child_state)
# The original state is unchanged
print(state.history())       # Output: []
print(child_state.history()) # Output: [4]
```
