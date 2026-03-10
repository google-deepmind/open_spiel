# OpenSpiel game methods: utility_sum

[Back to Core API reference](../api_reference.md) \
<br>

`utility_sum()`

Returns the total utility summed across all players, if this is a constant-sum
game. For zero-sum games, this returns 0. For general-sum games (where the total
utility varies across outcomes), this returns `None`.

## Examples:

```python
import pyspiel

# Zero-sum game
game = pyspiel.load_game("tic_tac_toe")
print(game.utility_sum())    # Output: 0.0

# Zero-sum game
game = pyspiel.load_game("leduc_poker")
print(game.utility_sum())    # Output: 0.0

# General-sum game
game = pyspiel.load_game("matrix_pd")
print(game.utility_sum())    # Output: None
```
