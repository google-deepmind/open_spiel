# OpenSpiel state methods: clone

[Back to Core API reference](../api_reference.md) \
<br>

`clone()`

Returns a deep copy of the current state. The cloned state is independent of the
original: modifying one does not affect the other.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)   # Take center

clone = state.clone()
clone.apply_action(0)   # Take top-left in clone only

print(len(state.history()))    # Output: 1
print(len(clone.history()))    # Output: 2
```
