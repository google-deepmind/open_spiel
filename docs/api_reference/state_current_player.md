# OpenSpiel state methods: current_player

[Back to Core API reference](../api_reference.md) \
<br>

`current_player()`

Returns the player ID of the acting player. Player IDs for actual players start
at 0 and end at `game.num_players() - 1`. There are some special player IDs that
represent the chance player, simultaneous-move nodes, and terminal states.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.current_player())    # Output: 0

game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()
print(state.current_player())    # Output: -1 (pyspiel.PlayerId.CHANCE)

game = pyspiel.load_game("matrix_rps")
state = game.new_initial_state()
print(state.current_player())    # Output: -2 (pyspiel.PlayerId.SIMULTANEOUS)
state.apply_actions([0, 0])      # I like to Rock! Oh yeah? Well.. so do I!
print(state.current_player())    # Output: -4 (pyspiel.PlayerId.TERMINAL)
```
