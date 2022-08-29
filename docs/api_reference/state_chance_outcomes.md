# OpenSpiel state methods: chance_outcomes

[Back to Core API reference](../api_reference.md) \
<br>

`chance_outcomes()`

Returns a list of (action, probability) tuples representing the probability
distribution over chance outcomes.

## Examples:

```python
import pyspiel
import numpy as np

game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()

# First player's private card.
print(state.chance_outcomes())
# Output:
# [(0, 0.16666666666666666), (1, 0.16666666666666666), (2, 0.16666666666666666), (3, 0.16666666666666666), (4, 0.16666666666666666), (5, 0.16666666666666666)]
state.apply_action(0)

# Second player's private card.
outcomes = state.chance_outcomes()
print()
# Output:
# [(1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2), (5, 0.2)]

# Sampling an outcome and applying it.
action_list, prob_list = zip(*outcomes)
action = np.random.choice(action_list, p=prob_list)
state.apply_action(action)
```
