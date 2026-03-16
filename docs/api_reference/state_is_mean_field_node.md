# OpenSpiel state methods: is_mean_field_node

[Back to Core API reference](../api_reference.md) \
<br>

`is_mean_field_node()`

Returns `True` if the current state is a mean field node, `False` otherwise. At
a mean field node, no action should be applied. Instead, the state distribution
should be updated via `update_distribution()`. This method is relevant for mean
field games only.

## Examples:

```python
import pyspiel

# Standard game: never a mean field node
game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.is_mean_field_node())    # Output: False

# Mean field game: may have mean field nodes
game = pyspiel.load_game("mfg_crowd_modelling_2d")
state = game.new_initial_state()
print(state.is_mean_field_node())    # Depends on game dynamics
```
