# OpenSpiel state methods: information_state_tensor

[Back to Core API reference](../api_reference.md) \
<br>

1.  `information_state_tensor()`
2.  `information_state_tensor(player: int)`

Returns information state tensor (a list of values) for (1) the current player,
or (2) the specified player.

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
print(state.information_state_tensor())
print(state.information_state_tensor(1))

# Tensors differ in the observing player and the card obtained.
# Output:
# [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
# [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
```
