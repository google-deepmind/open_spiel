# OpenSpiel state methods: full_history

[Back to Core API reference](../api_reference.md) \
<br>

`full_history()`

Returns the full history as a list of `PlayerAction` objects, where each entry
records both the player who acted and the action taken. This provides more
information than `history()`, which returns only the action integers. Each
`PlayerAction` has `.player` and `.action` attributes.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)   # Player 0 takes center
state.apply_action(0)   # Player 1 takes top-left

for pa in state.full_history():
    print(f"Player {pa.player}, Action {pa.action}")
# Output:
# Player 0, Action 4
# Player 1, Action 0

# Compare with history(), which returns only actions
print(state.history())   # Output: [4, 0]
```
