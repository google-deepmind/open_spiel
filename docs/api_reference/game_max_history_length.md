# OpenSpiel game methods: max_history_length

[Back to Core API reference](../api_reference.md) \
<br>

`max_history_length()`

Returns the maximum length of any history in the game. The value of
`len(state.history())` will never exceed this value. For sequential games, this
equals `max_game_length() + max_chance_nodes_in_history()`. For simultaneous-move
games, the history is flattened (one entry per player per joint move), so this
equals `max_game_length() * num_players() + max_chance_nodes_in_history()`.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
print(game.max_history_length())   # Output: 9
print(game.max_game_length())      # Output: 9

game = pyspiel.load_game("kuhn_poker")
print(game.max_history_length())   # Output: 15
```
