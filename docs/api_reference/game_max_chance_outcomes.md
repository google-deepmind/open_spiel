# OpenSpiel game methods: max_chance_outcomes

[Back to Core API reference](../api_reference.md) \
<br>

`max_chance_outcomes`

Returns the maximum number of distinct chance outcomes at chance nodes in the
game.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("chess")
print(game.max_chance_outcomes())
# Outputs: 0   (no chance nodes in Chess)

game = pyspiel.load_game("markov_soccer")
print(game.max_chance_outcomes())
# Outputs: 4   (ball starting location, and who gets initiative)

game = pyspiel.load_game("leduc_poker")
print(game.max_chance_outcomes())
# Outputs: 6   (three cards in two suits)
```
