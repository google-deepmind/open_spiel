# OpenSpiel game methods: get_type

[Back to Core API reference](../api_reference.md) \
<br>

`get_type()`

Returns the `GameType` object describing the static properties of this game,
such as its dynamics (sequential or simultaneous), chance mode, information type,
utility structure, and which observation/information state representations it
supports. This matches the information provided when the game was registered.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
game_type = game.get_type()
print(game_type.short_name)           # Output: tic_tac_toe
print(game_type.dynamics)             # Output: GameType.Dynamics.SEQUENTIAL
print(game_type.chance_mode)          # Output: GameType.ChanceMode.DETERMINISTIC
print(game_type.information)          # Output: GameType.Information.PERFECT_INFORMATION
print(game_type.utility)              # Output: GameType.Utility.ZERO_SUM
print(game_type.max_num_players)      # Output: 2

game = pyspiel.load_game("leduc_poker")
game_type = game.get_type()
print(game_type.chance_mode)          # Output: GameType.ChanceMode.EXPLICIT_STOCHASTIC
print(game_type.information)          # Output: GameType.Information.IMPERFECT_INFORMATION
```
