# OpenSpiel game methods: num_distinct_actions

[Back to Core API reference](../api_reference.md) \
<br>

`num_distinct_actions()`

Returns the number of state-independent actions in the game. Valid actions in a
game will always be between 0 and `num_distinct_actions() - 1`. This number can
be thought of as the fixed width of a policy head or Q-network. Legal actions
are always a subset of { 0, 1, ... , `num_distinct_actions() - 1` }.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
print(game.num_distinct_actions())    # Output: 9

game = pyspiel.load_game("go")
print (game.num_distinct_actions())   # Output: 362

game = pyspiel.load_game("chess")
print (game.num_distinct_actions())   # Output: 4672

game = pyspiel.load_game("leduc_poker")
print (game.num_distinct_actions())   # Output: 3
```
