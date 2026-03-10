# OpenSpiel state methods: get_game

[Back to Core API reference](../api_reference.md) \
<br>

`get_game()`

Returns the `Game` object that created this state. This is useful for accessing
game-level properties (such as `num_distinct_actions()` or `get_type()`) from a
state without needing to keep a separate reference to the game.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

retrieved_game = state.get_game()
print(retrieved_game.get_type().short_name)    # Output: tic_tac_toe
print(retrieved_game.num_distinct_actions())   # Output: 9
```
