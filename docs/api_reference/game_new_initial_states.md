# OpenSpiel game methods: new_initial_states

[Back to Core API reference](../api_reference.md) \
<br>

`new_initial_states()`

Returns a list of new initial states. For most games, this returns a single
state (identical to calling `new_initial_state()`). For multi-population mean
field games, it returns one initial state per population, ordered from
population 0 to population N-1 (where N is the number of players/populations).

## Examples:

```python
import pyspiel

# Standard game: returns a single initial state
game = pyspiel.load_game("tic_tac_toe")
states = game.new_initial_states()
print(len(states))    # Output: 1

# Mean field game: may return multiple initial states
game = pyspiel.load_game("mfg_crowd_modelling_2d")
states = game.new_initial_states()
print(len(states))    # Output: 1
```
