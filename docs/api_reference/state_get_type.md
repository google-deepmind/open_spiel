# OpenSpiel state methods: get_type

[Back to Core API reference](../api_reference.md) \
<br>

`get_type()`

Returns the type of the current state as a `StateType` enum value. Possible
values are:

- `StateType.CHANCE` — a chance node where nature acts
- `StateType.DECISION` — a decision node where one or more players act
- `StateType.TERMINAL` — the game has ended
- `StateType.MEAN_FIELD` — a mean field node (for mean field games)

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print(state.get_type())    # Output: StateType.DECISION

state.apply_action(4)
state.apply_action(1)
state.apply_action(2)
state.apply_action(5)
state.apply_action(6)
print(state.get_type())    # Output: StateType.TERMINAL

game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()
print(state.get_type())    # Output: StateType.CHANCE
```
