# OpenSpiel game methods: max_move_number

[Back to Core API reference](../api_reference.md) \
<br>

`max_move_number()`

Returns the maximum value that `state.move_number()` can reach in the game. By
default this equals `max_game_length() + max_chance_nodes_in_history()`, since
move number counts both player decisions and chance transitions. Simultaneous
moves count as a single move.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
print(game.max_move_number())    # Output: 9
print(game.max_game_length())    # Output: 9

game = pyspiel.load_game("kuhn_poker")
print(game.max_move_number())    # Output: 15
```
