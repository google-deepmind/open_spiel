# OpenSpiel game methods: max_utility and min_utility

[Back to Core API reference](../api_reference.md) \
<br>

`max_utility()` \
`min_utility()`

Returns the maximum and minimum achievable utility (return in any given episode)
in the game.

## Examples:

```python
import pyspiel

# Win/loss game
game = pyspiel.load_game("tic_tac_toe")
print(game.min_utility())    # Output: -1
print(game.max_utility())    # Output: 1

# Win/los/draw game (draw counts as 0).
game = pyspiel.load_game("chess")
print(game.min_utility())    # Output: -1
print(game.max_utility())    # Output: 1

# Money game.
game = pyspiel.load_game("leduc_poked")
print (game.num_distinct_actions())
print(game.min_utility())    # Output: -13
print(game.max_utility())    # Output: 13
```
