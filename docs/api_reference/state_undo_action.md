# OpenSpiel state methods: undo_action

[Back to Core API reference](../api_reference.md) \
<br>

`undo_action(player: int, action: int)`

Undoes the last action applied to the state. Both the player who took the action
and the action itself must be supplied. This is a fast method for reverting
state, intended for algorithms that need efficient undo (e.g. minimax search).

Not all games implement this method. If a game does not support undo, calling
this will raise an error.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
state.apply_action(4)    # Player 0 takes center
print(state.current_player())    # Output: 1

state.undo_action(0, 4)
print(state.current_player())    # Output: 0
print(state.history())           # Output: []
```
