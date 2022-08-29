# OpenSpiel state methods: rewards

[Back to Core API reference](../api_reference.md) \
<br>

`rewards()`

Returns the list of intermediate rewards (rewards obtained since the last time
the player acted): one value per player. Note that for many games in OpenSpiel,
this function will return zeroes unless the state is terminal.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("matrix_pd")
state = game.new_initial_state()

# Defect, Defect
state.apply_actions([1, 1])

# Rewards and returns equal in this case
print(state.rewards())
print(state.returns())

# Output:
# [1.0, 1.0]
# [1.0, 1.0]
```
