# OpenSpiel game methods: observation_tensor_layout

[Back to Core API reference](../api_reference.md) \
<br>

`observation_tensor_layout()`

Returns the tensor layout used for the observation tensor representation.
The layout is either `TensorLayout.CHW` (channels, height, width) or
`TensorLayout.HWC` (height, width, channels). Defaults to `CHW` for most games.

This is relevant when interpreting the flat vector returned by
`state.observation_tensor()` according to the shape given by
`game.observation_tensor_shape()`.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("leduc_poker")
print(game.observation_tensor_layout())
# Output: TensorLayout.CHW

print(game.observation_tensor_shape())
# Output: [16]
```
