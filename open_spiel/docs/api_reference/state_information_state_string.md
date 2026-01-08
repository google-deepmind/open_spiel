# OpenSpiel state methods: information_state_string

[Back to Core API reference](../api_reference.md) \
<br>

1.  `information_state_string()`
2.  `information_state_string(player: int)`

Returns a string representation of the information state, for (1) the current
player, or (2) the specified player.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("kuhn_poker")
state = game.new_initial_state()
state.apply_action(0)    # Deal first player the Jack,
state.apply_action(1)    # and second player the Queen
state.apply_action(0)    # First player passes (check)
state.apply_action(1)    # Second player bets (raise)

# Player 0's turn.
print(state.information_state_string())
print(state.information_state_string(1))

# Output:
# 0pb
# 1pb
```
