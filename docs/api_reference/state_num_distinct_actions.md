# OpenSpiel state methods: num_distinct_actions

[Back to Core API reference](../api_reference.md) \
<br>

`num_distinct_actions()`

Returns the number of distinct actions available in the game for any one player.
This is a state-independent value (the same for all states) and does not include
chance outcomes. It corresponds to the size of the action space, not the number
of currently legal actions. Equivalent to calling `game.num_distinct_actions()`.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.num_distinct_actions())   # Output: 9

state.apply_action(4)
# Still 9, even though only 8 actions are now legal
print(state.num_distinct_actions())   # Output: 9
print(len(state.legal_actions()))     # Output: 8
```
