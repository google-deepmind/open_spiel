# OpenSpiel state methods: apply_action_with_legality_check

[Back to Core API reference](../api_reference.md) \
<br>

`apply_action_with_legality_check(action: int)`

Applies the specified action to the state, but first verifies that the action is
legal. If the action is not in the list of legal actions, an error is raised.
This is a safer (but slower) alternative to `apply_action()`, which skips
legality checks for performance.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

# Legal action: succeeds
state.apply_action_with_legality_check(4)  # Take center square

# Illegal action: raises an error (square already taken)
try:
    state.apply_action_with_legality_check(4)
except pyspiel.SpielError as e:
    print(e)  # Action is not legal
```
