# pyspiel equivalence

`openspiel_equivalence_test.py` checks the JAX games move for move against a
reference implementation. Coverage, by game:

- checked against released OpenSpiel (`pyspiel.load_game`), every field
  below: bargaining, breakthrough, connect_four, dark_hex, dark_hex_ir, go,
  gomoku, hex (with the swap rule on), kuhn_poker, leduc_poker, liars_dice,
  lines_of_action, othello, phantom_ttt, phantom_ttt_ir, tic_tac_toe and
  heads-up no-limit hold'em (`universal_poker`);
- bridge: checked on every field for boards North deals; boards another
  player deals are checked only against an OpenSpiel with dealer-relative
  bridge observations (`bridge_reference.py` probes the installed pyspiel),
  because releases up to 2.0.2 place auction calls as if North always dealt
  while the JAX auction view is dealer-relative;
- gin_rummy: not checked against released OpenSpiel (`NEEDS_REFERENCE_BUILD`
  in the test says why); `run_playouts` with a loader for an OpenSpiel build
  that has the same rules checks it.

For each game the harness creates a pyspiel state and a JAX state, draws
a shared stream of actions from a seeded Python RNG, and applies each action to
both. A divergence fails with the game string, playout, step and field.

At every visited state it checks:

- `is_terminal` and `is_chance_node`;
- `current_player`, on non-terminal states;
- `returns`;
- at chance nodes, the set of chance outcomes and their probabilities;
- at player nodes, the legal actions, mapped in both directions through
  `action_to_os` and `action_from_os`;
- `observation_tensor` against `observation_from_os`, exactly;
- `information_state_tensor` against `information_state_from_os`, exactly,
  for games whose OpenSpiel counterpart provides one.

Every game registered under an OpenSpiel name is checked at OpenSpiel's
default parameters (the name with no parameters, or the defaults spelled out)
as well as at any other parameters it supports, and its action count,
player count and chance outcome count at the defaults are compared with
OpenSpiel's: the lockstep maps OpenSpiel's legal actions into the JAX mask, so
an extra action that is never legal would otherwise go unnoticed. A game that
refuses OpenSpiel's defaults is listed in `REFUSES_OPENSPIEL_DEFAULTS` with
the reason (`bridge`: OpenSpiel's default scores the auction with a
double-dummy solver), and the test checks that loading it by name raises.

Each game has its own playout budget: small games run hundreds to a
thousand playouts, heads-up no-limit hold'em 5000, and large boards fewer. The test
shards by the `TEST_TOTAL_SHARDS` and `TEST_SHARD_INDEX` environment variables.

No-limit hold'em playouts draw player actions from fixed strata rather than
uniformly, so betting trees develop past an early fold: some playouts fold,
most call, raise the minimum, raise a random amount or go all in. Targeted
tests cover an all-in run-out before the flop, minimum-raise chains, all-ins
after the flop, the per-round action limit, stack and pot invariants at every
state, and showdown returns for every hand class from high card to straight
flush, kickers included.

The reference games come from a `reference_loader`, `pyspiel.load_game` by
default. `run_playouts` takes
the loader as a parameter, so a test can check a game against another
OpenSpiel build.

Known limits:

- The player to move at a terminal state is game-specific in the JAX games, so
  `current_player` is compared on non-terminal states only.
- Long games may stop at their configured step cap. Truncation is explicit
  per game; every state before the cap is checked, and truncations are
  reported.
- OpenSpiel's gin rummy has no information-state tensor, and neither does
  the JAX one.

```bash
python -m open_spiel.python.jax.games.openspiel_equivalence_test
```
