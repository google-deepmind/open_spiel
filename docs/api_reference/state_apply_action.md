# OpenSpiel state methods: apply_action and apply_actions

[Back to Core API reference](../api_reference.md) \
<br>

1.  `apply_action(action: int)`
2.  `apply_actions(action: List[int])`

Apply the specified action in a turn-based game (1), or joint action (one action
per player) in a simultaneous-move game (2).

(1) must also be called to apply chance outcomes at chance nodes. (1) can also
be called on a simultaneous player state by passing in a flat integer (which was
obtained by `legal_actions()` on a simultaneous node).

In a simultaneous-move game, when a player has no legal actions, 0 must be
passed in for their action choice.

For performance reasons, legality of the actions are generally not checked and
applying an illegal action (or outcome at chance nodes) can fail in unspecified
ways.

## Examples:

```python
import pyspiel
import numpy as np

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)    # Player 0 takes the middle
state.apply_action(1)    # Player 1 takes the top

game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()
state.apply_action(0)    # First player gets the lowest card
state.apply_action(1)    # Second player gets the next lowest card
state.apply_action(1)    # First player checks

game = pyspiel.load_game("matrix_pd")   # Prisoner's dilemma
state = game.new_initial_state()
state.apply_actions([1, 1])    # Defect, Defect
```
