# OpenSpiel game methods: information_state_tensor_layout

[Back to Core API reference](../api_reference.md) \
<br>

`information_state_tensor_layout()`

Returns the tensor layout used for the information state tensor representation.
The layout is either `TensorLayout.CHW` (channels, height, width) or
`TensorLayout.HWC` (height, width, channels). Defaults to `CHW` for most games.

This is relevant when interpreting the flat vector returned by
`state.information_state_tensor()` according to the shape given by
`game.information_state_tensor_shape()`.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("leduc_poker")
print(game.information_state_tensor_layout())
# Output: TensorLayout.CHW

print(game.information_state_tensor_shape())
# Output: [30]
```
