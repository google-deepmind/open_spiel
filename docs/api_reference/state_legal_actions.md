# OpenSpiel state methods: legal_actions

[Back to Core API reference](../api_reference.md) \
<br>

1.  `legal_actions()`
2.  `legal_actions(player: int)`

Returns the list of legal actions (integers between 0 and
`game.num_distinct_actions() - 1`) for (1) the current player, or (2) the
specified player.

When called on a chance node, returns the legal chance outcomes without their
corresponding probabilities.

When called on a simultaneous node, returns the set of legal joint actions
represented as flat integers, which can then be passed to `apply_action`.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.legal_actions())
# Output: [0, 1, 2, 3, 4, 5, 6, 7, 8]

game = pyspiel.load_game("matrix_pd")
state = game.new_initial_state()
print(state.legal_actions(0))   # row player
print(state.legal_actions(1))   # column player
# Output:
# [0, 1]
# [0, 1]
```
