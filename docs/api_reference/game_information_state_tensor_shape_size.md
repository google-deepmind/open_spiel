# OpenSpiel game methods: information_state_tensor_shape and information_state_tensor_size

[Back to Core API reference](../api_reference.md) \
<br>

1.  `information_state_tensor_shape()`
2.  `information_state_tensor_size()`

(1) Returns the information state tensor's shape: a list of integers
representing the size of each dimension.

(2) Returns the total number of values used to represent the information state
tensor.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("kuhn_poker")
print(game.information_state_tensor_shape())
print(game.information_state_tensor_size())

# Output:
# [11]
# 11
```
