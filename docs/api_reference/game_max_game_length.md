# OpenSpiel game methods: max_game_length

[Back to Core API reference](../api_reference.md) \
<br>

`max_game_length()`

The maximum length of any one game (in terms of number of decision nodes 
visited in the game tree).

For a simultaneous action game, this is the maximum number of joint decisions.
In a turn-based game, this is the maximum number of individual decisions summed
over all players. Outcomes of chance nodes are not included in this length.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
print(game.max_game_length())    # Output: 9

# Normal-form games always have one
game = pyspiel.load_game("blotto")
print(game.max_game_length())    # Output: 1

# The maximum is arbitrarily defined (and/or customizable) is some games.
game = pyspiel.load_game("coop_box_pushing")
print(game.max_game_length())    # Output: 100
game = pyspiel.load_game("coop_box_pushing(horizon=250)")
print(game.max_game_length())    # Output: 250
```
