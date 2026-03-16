# OpenSpiel state methods: string_to_action

[Back to Core API reference](../api_reference.md) \
<br>

1.  `string_to_action(player: int, action_str: str)`
2.  `string_to_action(action_str: str)`

Converts a string representation of an action back to its integer action ID.
This reverses the mapping done by `action_to_string()`. The parameterless
version uses the current player.

Note: the default implementation loops over all legal actions and compares
strings, so it can be slow for games with large action spaces.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

action_str = state.action_to_string(0, 4)
print(action_str)                            # Output: x(1,1)

action_id = state.string_to_action(0, action_str)
print(action_id)                             # Output: 4

# Parameterless version uses current player
action_id = state.string_to_action(action_str)
print(action_id)                             # Output: 4
```
