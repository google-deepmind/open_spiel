# OpenSpiel game methods: make_observer

[Back to Core API reference](../api_reference.md) \
<br>

1.  `make_observer(iig_obs_type: IIGObservationType, params: dict)`
2.  `make_observer(params: dict)`

Creates an `Observer` object for obtaining observations of the game state. The
observer produces observation tensors and strings according to the requested
observation type. If `iig_obs_type` is omitted, the default observation type is
used.

The `IIGObservationType` controls what information is included in the
observation: public information, private information, and whether perfect recall
is used. See `observer.h` for details.

Returns `None` if the requested observation type is not supported by the game.

## Examples:

```python
import pyspiel

game = pyspiel.load_game("leduc_poker")
observer = game.make_observer(
    pyspiel.IIGObservationType(
        public_info=True,
        perfect_recall=False,
        private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER),
    {})
state = game.new_initial_state()
state.apply_action(0)  # Deal first card
state.apply_action(1)  # Deal second card
print(observer.string_from(state, player=0))

# Using default observation type
observer2 = game.make_observer({})
```
