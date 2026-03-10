# OpenSpiel state methods: apply_actions_with_legality_checks

[Back to Core API reference](../api_reference.md) \
<br>

`apply_actions_with_legality_checks(actions: List[int])`

Applies the specified joint action (one action per player) in a
simultaneous-move game, but first verifies that each player's action is legal.
If any action is not legal, an error is raised. This is a safer (but slower)
alternative to `apply_actions()`, which skips legality checks for performance.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("matrix_pd")   # Prisoner's dilemma
state = game.new_initial_state()

# Legal joint action: both cooperate (action 0)
state.apply_actions_with_legality_checks([0, 0])
print(state.returns())    # Output: [-1.0, -1.0]
```
