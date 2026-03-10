# OpenSpiel state methods: resample_from_infostate

[Back to Core API reference](../api_reference.md) \
<br>

`resample_from_infostate(player_id: int, rng: callable)`

Resamples a new game history that is consistent with the specified player's
information state. The resampled state keeps `player_id`'s private information
and all public information unchanged, but resamples the private information of
other players. Chance outcomes are sampled uniformly from consistent
possibilities.

The `rng` parameter should be a callable that returns a float in `[0, 1)`,
used to sample from chance actions.

For perfect information games, this returns a clone of the current state. For
imperfect information games, the game must implement this method.

## Examples:

```python
import pyspiel
import random

game = pyspiel.load_game("kuhn_poker")
state = game.new_initial_state()
state.apply_action(0)   # Deal card 0 to player 0
state.apply_action(1)   # Deal card 1 to player 1

# Resample from player 0's perspective
resampled = state.resample_from_infostate(
    0, lambda: random.random())
# Player 0 still has card 0, but player 1's card may differ
```
