# OpenSpiel functions: registered_names

[Back to Core API reference](../api_reference.md) \
<br>

`registered_names()`

Returns a list of short names of all game in the library. These are names that
can be used when loading games in `load_game`.

## Examples:

```python
import pyspiel

# Print the name of all OpenSpiel games
for short_name in pyspiel.registered_names():
  print(short_name)
```
