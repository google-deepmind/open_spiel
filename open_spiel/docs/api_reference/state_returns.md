# OpenSpiel state methods: returns

[Back to Core API reference](../api_reference.md) \
<br>

`returns()`

Returns the list of returns (cumulated reward from the start of the game): one
value per player.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

# Play out a win for 'x'.
state.apply_action(4)
state.apply_action(1)
state.apply_action(2)
state.apply_action(5)
state.apply_action(6)
print(state)
print(state.returns())

# Output:
# .ox
# .xo
# x..
# [1.0, -1.0]
```
