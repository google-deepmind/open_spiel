# OpenSpiel functions: load_game

[Back to Core API reference](../api_reference.md) \
<br>

1.  `load_game(game_string: str)`
2.  `load_game(game_string: str, parameters: Dict[str, Any])`

Returns a newly-loaded game. The game string can be the short name of any game
on its own, or the short name followed by a comma-separated list of `key=value`
pairs within parentheses.

## Examples:

```python
import pyspiel

# Loads the game with no/default parameters.
game1 = pyspiel.load_game("tic_tac_toe")

# Loads the game with no/default parameters (8x8 Breakthrough)
game2 = pyspiel.load_game("breakthrough")

# Load a three-player Kuhn poker game.
game3 = pyspiel.load_game("kuhn_poker(players=3)")

# Load the imperfect information variant of Goofspiel with five cards, and the
# unspecified parameters get their default values (two different ways):
game4 = pyspiel.load_game("goofspiel(imp_info=True,num_cards=5,points_order=descending)")
game5 = pyspiel.load_game("goofspiel", {
    "imp_info": True,
    "num_cards": 5,
    "points_order": "descending"
})
```
