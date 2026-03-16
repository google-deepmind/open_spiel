# OpenSpiel state methods: move_number

[Back to Core API reference](../api_reference.md) \
<br>

`move_number()`

Returns how many moves have been made so far in the game. Simultaneous moves
(where all players act at once) count as a single move. Chance transitions also
count as one move. Note that game transformations are not required to preserve
the move number.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.move_number())    # Output: 0

state.apply_action(4)
print(state.move_number())    # Output: 1

state.apply_action(0)
print(state.move_number())    # Output: 2
```
