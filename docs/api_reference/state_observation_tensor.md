# OpenSpiel state methods: observation_tensor

[Back to Core API reference](../api_reference.md) \
<br>

1.  `observation_tensor()`
2.  `observation_tensor(player: int)`

Returns observation tensor (a list of values) for (1) the current player, or (2)
the specified player.

## Examples:

```python
import pyspiel
import numpy as np

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)    # Middle
state.apply_action(2)    # Top-right

# Player 0's turn.
shape = game.observation_tensor_shape()
print(state.observation_tensor())
print(state.observation_tensor(0))

# First dimension interpreted as selecting from 2D planes of { empty, O, X }.
print(np.reshape(np.asarray(state.observation_tensor()), shape))

# Output:
# [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
# [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
# [[[1. 1. 0.]
#   [1. 0. 1.]
#   [1. 1. 1.]]
#
#  [[0. 0. 1.]
#   [0. 0. 0.]
#   [0. 0. 0.]]
#
#  [[0. 0. 0.]
#   [0. 1. 0.]
#   [0. 0. 0.]]]
```
