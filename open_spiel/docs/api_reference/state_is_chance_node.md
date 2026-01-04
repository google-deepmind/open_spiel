# OpenSpiel state methods: is_chance_node

[Back to Core API reference](../api_reference.md) \
<br>

`is_chance_node()`

Returns True if the state represents a chance node, False otherwise.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.is_chance_node())    # Output: False

game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()
print(state.is_chance_node())    # Output: True

game = pyspiel.load_game("matrix_sh")
state = game.new_initial_state()
print(state.is_chance_node())    # Output: False
```
