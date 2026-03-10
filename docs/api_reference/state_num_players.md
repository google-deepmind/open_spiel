# OpenSpiel state methods: num_players

[Back to Core API reference](../api_reference.md) \
<br>

`num_players()`

Returns the number of players in the game. Does not include the chance player.
This is a state-independent value, equivalent to calling `game.num_players()`.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.num_players())    # Output: 2

game = pyspiel.load_game("catch")
state = game.new_initial_state()
print(state.num_players())    # Output: 1
```
