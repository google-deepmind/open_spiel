# OpenSpiel game methods: get_parameters

[Back to Core API reference](../api_reference.md) \
<br>

`get_parameters()`

Returns a dictionary of the game's parameter values, including any parameters
that were set to their default values. This is useful for inspecting how a game
was configured, or for reconstructing a game with the same settings.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
print(game.get_parameters())
# Output: {}

game = pyspiel.load_game("leduc_poker")
print(game.get_parameters())
# Output: {'players': 2}

game = pyspiel.load_game("go(board_size=9,komi=5.5)")
print(game.get_parameters())
# Output: {'board_size': 9, 'handicap': 0, 'komi': 5.5, 'max_game_length': 162}
```
