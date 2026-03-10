# OpenSpiel state methods: update_distribution

[Back to Core API reference](../api_reference.md) \
<br>

`update_distribution(distribution: List[float])`

Updates the state distribution at a mean field node. The `distribution` list
must correspond element-by-element to the states returned by
`distribution_support()`. After this call, the state transitions to a chance
node. This should only be called when `current_player()` returns
`PlayerId.MEAN_FIELD`.

This method is specific to mean field games.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("mfg_crowd_modelling_2d")
state = game.new_initial_state()

# Advance to a mean field node
while not state.is_mean_field_node():
    if state.is_chance_node():
        action = state.chance_outcomes()[0][0]
        state.apply_action(action)
    else:
        state.apply_action(state.legal_actions()[0])

support = state.distribution_support()
# Set a uniform distribution
dist = [1.0 / len(support)] * len(support)
state.update_distribution(dist)
```
