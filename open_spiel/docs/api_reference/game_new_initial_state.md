# OpenSpiel game methods: new_initial_state

[Back to Core API reference](../api_reference.md) \
<br>

`new_initial_state()`

Returns a new state object representing the first state of the game. Note, in
particular, this might be a chance node (where the current player is chance) in
games with chance events.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("hex")
state = game.new_initial_state()
print(state)

# Output:
# . . . . . . . . . . .
#  . . . . . . . . . . .
#   . . . . . . . . . . .
#    . . . . . . . . . . .
#     . . . . . . . . . . .
#      . . . . . . . . . . .
#       . . . . . . . . . . .
#        . . . . . . . . . . .
#         . . . . . . . . . . .
#          . . . . . . . . . . .
#           . . . . . . . . . . .
```
