# OpenSpiel game methods: observation_tensor_shape and observation_tensor_size

[Back to Core API reference](../api_reference.md) \
<br>

1.  `observation_tensor_shape()`
2.  `observation_tensor_size()`

(1) Returns the observation tensor's shape: a list of integers representing the
size of each dimension.

(2) Returns the total number of values used to represent the observation tensor.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
print(game.observation_tensor_shape())
print(game.observation_tensor_size())

# Output:
# [3, 3, 3]
# 27
```
