# OpenSpiel game methods: policy_tensor_shape

[Back to Core API reference](../api_reference.md) \
<br>

`policy_tensor_shape()`

Returns the shape of the policy tensor as a list of integers. This describes the
structure of a policy vector over actions. By default this is
`[num_distinct_actions()]`, representing a flat probability distribution over all
actions. Games may override this to provide a structured shape (e.g. a 2D grid
for board games).

## Examples:

```python
import pyspiel

game = pyspiel.load_game("tic_tac_toe")
print(game.policy_tensor_shape())        # Output: [9]
print(game.num_distinct_actions())       # Output: 9

game = pyspiel.load_game("leduc_poker")
print(game.policy_tensor_shape())        # Output: [4]
```
