# OpenSpiel state methods: distribution_support

[Back to Core API reference](../api_reference.md) \
<br>

`distribution_support()`

Returns the support of the state distribution that needs to be updated at the
current mean field node. States in the support are identified by their string
representations. This should only be called when `current_player()` returns
`PlayerId.MEAN_FIELD`. Can return an empty list if the distribution is not needed
at this point.

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
print(type(support))    # Output: <class 'list'>
```
