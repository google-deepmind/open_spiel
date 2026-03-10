# OpenSpiel game methods: new_initial_state_for_population

[Back to Core API reference](../api_reference.md) \
<br>

`new_initial_state_for_population(population: int)`

Returns a new initial state for the given population. This is used for
multi-population mean field games, where each population has its own initial
state. The `population` parameter must be in the range `[0, num_players())`.

For single-population mean field games or standard games, use
`new_initial_state()` instead.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("mfg_crowd_modelling_2d")
state_pop0 = game.new_initial_state_for_population(0)
print(state_pop0)
```
