# OpenSpiel state methods: is_simultaneous_node

[Back to Core API reference](../api_reference.md) \
<br>

`is_simultaneous_node()`

Returns True if the state represents a simultaneous player node (where all
players act simultaneously), False otherwise.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.is_simultaneous_node())    # Output: False

game = pyspiel.load_game("matrix_mp")
state = game.new_initial_state()
print(state.is_simultaneous_node())    # Output: True

# Simultaneous-move game that start at a chance node.
game = pyspiel.load_game("markov_soccer")
state = game.new_initial_state()
print(state.is_simultaneous_node())    # Output: False
print(state.legal_actions())
state.apply_action(state.legal_actions()[0])   # Apply first legal chance outcome.
print(state.is_simultaneous_node())    # Output: True

```
