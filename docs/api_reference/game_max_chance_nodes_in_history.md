# OpenSpiel game methods: max_chance_nodes_in_history

[Back to Core API reference](../api_reference.md) \
<br>

`max_chance_nodes_in_history()`

Returns the maximum number of chance nodes that can appear in any history
(sequence of states) of the game. For deterministic games this is 0. For
stochastic games it defaults to `max_game_length()` as a loose upper bound,
though individual games may override this with a tighter value.

## Examples:

```python
import pyspiel

# Deterministic game: no chance nodes
game = pyspiel.load_game("tic_tac_toe")
print(game.max_chance_nodes_in_history())   # Output: 0

# Stochastic game: chance nodes from card dealing
game = pyspiel.load_game("kuhn_poker")
print(game.max_chance_nodes_in_history())   # Output: 12
```
