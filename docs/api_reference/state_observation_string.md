# OpenSpiel state methods: observation_string

[Back to Core API reference](../api_reference.md) \
<br>

1.  `observation_string()`
2.  `observation_string(player: int)`

Returns a string representation of the observation, for (1) the current player,
or (2) the specified player.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("breakthrough")
state = game.new_initial_state()
print(state.action_to_string(0, 148))    # Output: e7f6
state.apply_action(148)

print(state.observation_string())
# Output:
# 8bbbbbbbb
# 7bbbb.bbb
# 6.....b..
# 5........
# 4........
# 3........
# 2wwwwwwww
# 1wwwwwwww
#  abcdefgh

# Perfect information game, same observation for both players.
print(state.observation_string(0))
# Output:
# 8bbbbbbbb
# 7bbbb.bbb
# 6.....b..
# 5........
# 4........
# 3........
# 2wwwwwwww
# 1wwwwwwww
#  abcdefgh
```
