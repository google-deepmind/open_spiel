# OpenSpiel games in JAX

This package reimplements OpenSpiel games in pure JAX. Each game is a set of pure
functions over a fixed-shape state pytree, so thousands of games run in
parallel on a GPU or TPU inside one compiled program, and a whole rollout can
be a `jax.lax.scan`. The games use OpenSpiel's names, parameters and action
ids, and a test harness checks most of them move for move against OpenSpiel;
`docs/pyspiel_equivalence.md` lists what each game is checked against and
which checks a game skips.

Acknowledgements. These JAX implementations of OpenSpiel games were inspired
by Pgx (https://github.com/sotetsuk/pgx), a collection of JAX-native game
environments. They are independent implementations written against
OpenSpiel's game definitions and checked against `pyspiel`; no Pgx code is
included.

## Install

The games come with OpenSpiel. Install OpenSpiel from source (see
`docs/install.md`), then its pinned JAX dependencies, from the root of the
checkout:

```bash
pip install .
source open_spiel/scripts/python_extra_deps.sh python3
echo "$OPEN_SPIEL_PYTHON_JAX_DEPS" | xargs pip install
```

Then `import open_spiel.python.jax.games`; the examples import it as
`games`. The games themselves need only JAX; OpenSpiel's `pyspiel` serves the
comparisons with OpenSpiel and the conversion of pyspiel states (`interop`).

## Quick start

```python
import jax
import jax.numpy as jnp
from open_spiel.python.jax import games

game = games.load_game("connect_four")
batch = 4096
states = jax.vmap(lambda _: game.new_initial_state())(jnp.arange(batch))

@jax.jit
def random_step(states, key):
  keys = jax.random.split(key, batch)
  def act(state, k):
    logits = jnp.where(game.legal_actions_mask(state), 0.0, -jnp.inf)
    return game.apply_action(state, jax.random.categorical(k, logits))
  return jax.vmap(act)(states, keys)

states = random_step(states, jax.random.PRNGKey(0))
print(jax.vmap(game.returns)(states).shape)          # (4096, 2)
```

## The Game interface

| Method | Meaning |
| --- | --- |
| `new_initial_state()` | The start of the game, possibly a chance node. |
| `apply_action(state, action)` | The next state; `action` is an OpenSpiel action id or chance outcome id. |
| `current_player(state)` | The player to move, or `CHANCE_PLAYER_ID`. |
| `is_terminal(state)`, `is_chance_node(state)` | bool scalars. |
| `legal_actions_mask(state)` | `bool[num_actions]`. |
| `chance_outcomes(state)` | `f32[max_chance_outcomes]`, the chance distribution indexed by outcome id. |
| `observation_tensor(state, player)` | What `player` sees now. |
| `information_state_tensor(state, player)` | Everything `player` has seen. |
| `returns(state)` | `f32[num_players]`. |

Every method is jittable and works under `vmap`. `interop`'s
`PyspielInterop`, which every registered game inherits, converts pyspiel
states into a game's tensors; the views (`breakthrough.egocentric_view`,
`hex.canonical_view`) do not inherit it.

## Package map

| Module | Contents |
| --- | --- |
| `games` (the package itself) | `load_game`, `load_game_module`, `register_game`, `parse_game_string`, `registered_names` (from `registry`), `Game`, `CHANCE_PLAYER_ID`, `TERMINAL_PLAYER_ID` (from `game`) |
| `game` | The `Game` interface. |
| `registry` | The game registry; a built-in game's module is imported when the game is first loaded, or by `load_game_module(name)`. |
| `interop` | `PyspielInterop`: conversions from pyspiel states. |
| one module per game | See the list below. |
| `benchmarks` | The throughput harness (`throughput_common`) and per-game benchmarks. |

## Extension points

| Hook | Use |
| --- | --- |
| `games.register_game(name, ctor, replace=False)` | Make a game loadable by name. |
| `breakthrough.egocentric_view(game)` | Breakthrough in each player's own orientation, with compact actions. |
| `hex.canonical_view(game)` | Hex transposed for white, so both players join top and bottom. |
| `gin_rummy.GinRummy(backend=...)` | Pick the meld-cache implementation: `"reference"` or `"dense_cache"`. |

## Games

A game registered under an OpenSpiel name plays OpenSpiel's game at the same
parameters, OpenSpiel's defaults included, or refuses parameters it does not
implement with a `ValueError`; `load_game(name)` with no parameters is
OpenSpiel's default game or an error, never another game.

| Name | OpenSpiel game string | Notes |
| --- | --- | --- |
| `bargaining` | `bargaining` | |
| `breakthrough` | `breakthrough` | `rows`, `columns` |
| `bridge` | `bridge(use_double_dummy_result=false)` | auction and card play; `random_dealer_vul` draws the dealer and vulnerability per board. OpenSpiel's default (`use_double_dummy_result=true`, scored by a double-dummy solver) is refused, so `load_game("bridge")` raises |
| `connect_four` | `connect_four` | `rows`, `columns` |
| `dark_hex` | `dark_hex(board_size=3)` | 3x3 only |
| `dark_hex_ir` | `dark_hex_ir` | 3x3 only; `gameversion` `cdh` (default) or `adh` |
| `gin_rummy` | `gin_rummy` | `oklahoma` and optional rule variants; one scoring rule differs from released OpenSpiel: equal deadwood after layoffs is an undercut (the defender scores the undercut bonus), where OpenSpiel scores the tie 0 for both, so it is not move-checked against stock OpenSpiel |
| `go` | `go` | `board_size`, `komi`, `max_game_length`; simple ko, and a repeated position ends the game as a draw (OpenSpiel's superko); no handicap |
| `gomoku` | `gomoku` | `board_size`, `connect` |
| `hex` | `hex` | `board_size`, `swap` (default off: `board_size**2` actions, one more with the swap rule) |
| `kuhn_poker` | `kuhn_poker` | two players |
| `leduc_poker` | `leduc_poker` | two players |
| `liars_dice` | `liars_dice` | `numdice` (default 1), `dice_sides` (default 6); two players, `reset-face` bidding |
| `lines_of_action` | `lines_of_action` | `max_game_length` |
| `no_limit_holdem` | `universal_poker(betting=nolimit,...)` | heads-up, 200-chip stacks; see the module |
| `othello` | `othello` | |
| `phantom_ttt` | `phantom_ttt` | classical variant |
| `phantom_ttt_ir` | `phantom_ttt_ir` | `gameversion` `classical` (default) or `abrupt` |
| `tic_tac_toe` | `tic_tac_toe` | |

## Testing

Each test file runs on its own, for example:

```bash
python -m open_spiel.python.jax.games.kuhn_poker_test
```

Under Bazel (`bazel test --config=jax`) and `ctest` (OpenSpiel built with
`OPEN_SPIEL_ENABLE_JAX=ON`), the fast tests run by default; the long ones
are tagged `manual` (Bazel runs them when named, ctest with
`-DOPEN_SPIEL_RUN_MANUAL_TESTS=ON`).

`openspiel_equivalence_test` plays the games in lockstep with OpenSpiel; see
`docs/pyspiel_equivalence.md` for the per-game coverage.

## Licence

Part of OpenSpiel, under its Apache-2.0 licence (the `LICENSE` file at the
repository root).
