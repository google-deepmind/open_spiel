# OpenSpiel game methods: num_players

[Back to Core API reference](../api_reference.md) \
<br>

`num_players()`

Returns the number of players in this instantiation of the game. Does not
include the chance player.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
print(game.num_players())    # Output: 2

game = pyspiel.load_game("leduc_poker(players=3)")
print(game.num_players())    # Output: 3

game = pyspiel.load_game("catch")
print(game.num_players())    # Output: 1
```
