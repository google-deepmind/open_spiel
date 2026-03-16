# OpenSpiel state methods: mean_field_population

[Back to Core API reference](../api_reference.md) \
<br>

`mean_field_population()`

Returns the population that this state belongs to. This is relevant only for
mean field games. Returns 0 by default (single-population games).
Multi-population mean field games should override this to return the correct
population index.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("mfg_crowd_modelling_2d")
state = game.new_initial_state()
print(state.mean_field_population())    # Output: 0
```
