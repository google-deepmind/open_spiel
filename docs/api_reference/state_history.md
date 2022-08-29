# OpenSpiel state methods: history

[Back to Core API reference](../api_reference.md) \
<br>

`history()`

Returns a list of actions taken by all players (including chance) from the
beginning of the game.

In simultaneous-move games, joint actions are written out sequentially in player
ID order.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("kuhn_poker")
state = game.new_initial_state()
state.apply_action(0)    # First player gets the Jack
state.apply_action(1)    # Second player gets the Queen
state.apply_action(0)    # First player passes (check)
state.apply_action(1)    # Second player bets (raise)

print(state.history())
# Output: [0, 1, 0, 1]

game = pyspiel.load_game("matrix_pd")
state = game.new_initial_state()
state.apply_actions([0, 1])   # Cooperate, Defect
print(state.history())
# Output: [0, 1]
```
