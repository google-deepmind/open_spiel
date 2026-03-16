# OpenSpiel state methods: legal_actions_mask

[Back to Core API reference](../api_reference.md) \
<br>

1.  `legal_actions_mask(player: int)`
2.  `legal_actions_mask()`

Returns a binary vector where 1 indicates a legal action and 0 indicates an
illegal action. The length is `game.num_distinct_actions()` for player nodes, or
`game.max_chance_outcomes()` for chance nodes. The parameterless version uses the
current player.

This is useful for masking illegal actions in neural network policy outputs.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.legal_actions_mask())
# Output: [1, 1, 1, 1, 1, 1, 1, 1, 1]  (all 9 squares available)

state.apply_action(4)   # Player 0 takes center
print(state.legal_actions_mask())
# Output: [1, 1, 1, 1, 0, 1, 1, 1, 1]  (center no longer available)

# Also works with explicit player argument
print(state.legal_actions_mask(1))
# Output: [1, 1, 1, 1, 0, 1, 1, 1, 1]
```
