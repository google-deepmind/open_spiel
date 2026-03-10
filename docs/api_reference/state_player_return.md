# OpenSpiel state methods: player_return

[Back to Core API reference](../api_reference.md) \
<br>

`player_return(player: int)`

Returns the cumulated return (sum of all rewards) for the specified player from
the start of the game up to the current state. This is a single-player
convenience method; if you need returns for all players, use `returns()` instead
as it is more efficient.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

# Play out a win for player 0
state.apply_action(4)
state.apply_action(1)
state.apply_action(2)
state.apply_action(5)
state.apply_action(6)

print(state.player_return(0))     # Output: 1.0
print(state.player_return(1))     # Output: -1.0
print(state.returns())            # Output: [1.0, -1.0]
```
