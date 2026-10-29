# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Step a JAX game and check it move for move against OpenSpiel.

What it shows:
  * loading a game from the JAX games package by its OpenSpiel name and
    stepping it with pure, jittable functions (`new_initial_state`,
    `apply_action`, `legal_actions_mask`, `observation_tensor`, ...);
  * a cross-check against OpenSpiel itself (`pyspiel`): random playouts run
    in lockstep, and at every state the player to move, terminality, chance
    distribution, legal actions, observation and information-state tensors
    and returns must agree. Any difference stops the script with the game,
    playout and step. A JAX game's tensors may use their own layout (tic-tac-
    toe's is [row, column, plane], from the observer's point of view), so
    tensors are compared through the game's own converter from a pyspiel
    state (`observation_from_os`, `information_state_from_os`);
  * the reason the games are in JAX: `jax.vmap` plays thousands of games at
    once inside one compiled loop (`jax.lax.while_loop`).

  python -m open_spiel.python.scale.examples.jax_games_vs_openspiel
  python -m open_spiel.python.scale.examples.jax_games_vs_openspiel \\
      --games=kuhn_poker --playouts=5
"""

import time

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import registry
import pyspiel

_GAMES = flags.DEFINE_list(
    "games",
    ["tic_tac_toe", "connect_four", "kuhn_poker", "leduc_poker"],
    "OpenSpiel game names to check.",
)
_PLAYOUTS = flags.DEFINE_integer("playouts", 50, "Random playouts per game.")
_BATCH = flags.DEFINE_integer(
    "batch", 4096, "Games in the batched random rollout."
)
_SEED = flags.DEFINE_integer("seed", 0, "Seed of the playouts.")


class MismatchError(AssertionError):
  """The JAX game and OpenSpiel disagree."""


def check(
    ok: bool, game: str, playout: int, step: int, what: str, got, want
) -> None:
  """Raises MismatchError unless `ok`.

  Args:
    ok: Whether the two implementations agree.
    game: The game name.
    playout: The playout index.
    step: The step index.
    what: What was compared.
    got: The JAX game's value.
    want: OpenSpiel's value.

  Raises:
    MismatchError: `ok` is False.
  """
  if not ok:
    raise MismatchError(
        f"{game}: playout {playout}, step {step}: {what} differs:\n"
        f"  JAX game: {got}\n  OpenSpiel: {want}"
    )


def lockstep(name: str, playouts: int, seed: int) -> int:
  """Plays `playouts` random games in both implementations, comparing them.

  Args:
    name: The game name.
    playouts: The number of games.
    seed: The seed.

  Returns:
    The number of states compared.

  Raises:
    MismatchError: the sizes or a compared value differ.
  """
  game = registry.load_game(name)
  ref = pyspiel.load_game(name)
  if (
      game.num_actions != ref.num_distinct_actions()
      or game.num_players != ref.num_players()
  ):
    raise MismatchError(
        f"{name}: sizes differ: JAX ({game.num_actions} actions, "
        f"{game.num_players} players), OpenSpiel "
        f"({ref.num_distinct_actions()}, {ref.num_players()})"
    )
  init = jax.jit(game.new_initial_state)
  step = jax.jit(game.apply_action)
  view = jax.jit(
      lambda s: dict(
          player=game.current_player(s),
          terminal=game.is_terminal(s),
          chance=game.is_chance_node(s),
          legal=game.legal_actions_mask(s),
          returns=game.returns(s),
          chance_probs=game.chance_outcomes(s),
          obs=[
              game.observation_tensor(s, jnp.int32(p))
              for p in range(game.num_players)
          ],
          info=[
              game.information_state_tensor(s, jnp.int32(p))
              for p in range(game.num_players)
          ],
      )
  )
  rng = np.random.default_rng(seed)
  compared = 0
  for i in range(playouts):
    s, r = init(), ref.new_initial_state()
    t = 0
    while True:
      v = jax.device_get(view(s))
      check(
          bool(v["terminal"]) == r.is_terminal(),
          name,
          i,
          t,
          "is_terminal",
          v["terminal"],
          r.is_terminal(),
      )
      compared += 1
      if r.is_terminal():
        check(
            np.allclose(v["returns"], r.returns()),
            name,
            i,
            t,
            "returns",
            v["returns"],
            r.returns(),
        )
        break
      check(
          bool(v["chance"]) == r.is_chance_node(),
          name,
          i,
          t,
          "is_chance_node",
          v["chance"],
          r.is_chance_node(),
      )
      if r.is_chance_node():
        want = np.zeros(game.max_chance_outcomes)
        for a, p in r.chance_outcomes():
          want[a] = p
        check(
            np.allclose(v["chance_probs"], want, atol=1e-6),
            name,
            i,
            t,
            "chance_outcomes",
            v["chance_probs"],
            want,
        )
        actions, probs = zip(*r.chance_outcomes())
        a = int(rng.choice(actions, p=probs))
      else:
        check(
            int(v["player"]) == r.current_player(),
            name,
            i,
            t,
            "current_player",
            v["player"],
            r.current_player(),
        )
        legal = np.flatnonzero(v["legal"]).tolist()
        check(
            legal == r.legal_actions(),
            name,
            i,
            t,
            "legal_actions",
            legal,
            r.legal_actions(),
        )
        for p in range(ref.num_players()):
          for what, got, conv in (
              ("observation_tensor", v["obs"][p], game.observation_from_os),
              (
                  "information_state_tensor",
                  v["info"][p],
                  game.information_state_from_os,
              ),
          ):
            want = np.asarray(conv(r, p))
            check(
                got.shape == want.shape and np.array_equal(got, want),
                name,
                i,
                t,
                f"{what}[player {p}]",
                got,
                want,
            )
        a = int(rng.choice(legal))
      s = step(s, jnp.int32(a))
      r.apply_action(a)
      t += 1
  return compared


def batched_rollout(
    name: str, batch: int, seed: int
) -> tuple[float, float, np.ndarray]:
  """Plays `batch` uniform-random games at once.

  Args:
    name: The game name.
    batch: The number of games.
    seed: The seed.

  Returns:
    (compile s, run s, mean returns).

  Raises:
    RuntimeError: a game did not finish within the length bound.
  """
  game = registry.load_game(name)
  # The longest game: OpenSpiel's bound on player moves plus chance moves.
  # Finished games stay terminal; the loop stops once every game has finished.
  ref = pyspiel.load_game(name)
  max_len = ref.max_game_length() + ref.max_chance_nodes_in_history()

  def act(state, key):
    """Plays one uniformly random action; a no-op at a terminal state.

    Args:
      state: The state.
      key: The PRNG key.

    Returns:
      The next state.
    """
    k_chance, k_move = jax.random.split(key)
    if game.max_chance_outcomes:
      chance_logits = jnp.log(game.chance_outcomes(state))
    else:
      chance_logits = jnp.zeros((1,))
    move_logits = jnp.where(game.legal_actions_mask(state), 0.0, -jnp.inf)
    a = jnp.where(
        game.is_chance_node(state),
        jax.random.categorical(k_chance, chance_logits),
        jax.random.categorical(k_move, move_logits),
    ).astype(jnp.int32)
    nxt = game.apply_action(state, a)
    return jax.tree_util.tree_map(
        lambda new, old: jnp.where(game.is_terminal(state), old, new),
        nxt,
        state,
    )

  @jax.jit
  def play(key):
    """Plays the batch to the end.

    Args:
      key: The PRNG key.

    Returns:
      (returns [batch, players], done bool[batch]).
    """
    states = jax.vmap(lambda _: game.new_initial_state())(jnp.arange(batch))

    def unfinished(carry):
      """Returns whether to keep playing.

      Args:
        carry: (states, key, ply).

      Returns:
        True while a game is unfinished and the length bound is not reached.
      """
      states, _, t = carry
      return (t < max_len) & ~jnp.all(jax.vmap(game.is_terminal)(states))

    def body(carry):
      """Plays one ply in every game.

      Args:
        carry: (states, key, ply).

      Returns:
        The next carry.
      """
      states, key, t = carry
      key, sub = jax.random.split(key)
      return jax.vmap(act)(states, jax.random.split(sub, batch)), key, t + 1

    states, _, _ = jax.lax.while_loop(
        unfinished, body, (states, key, jnp.int32(0))
    )
    done = jax.vmap(game.is_terminal)(states)
    return jax.vmap(game.returns)(states), done

  t0 = time.time()
  play.lower(jax.random.key(seed)).compile()
  t1 = time.time()
  returns, done = jax.block_until_ready(play(jax.random.key(seed)))
  t2 = time.time()
  if not bool(np.all(done)):
    raise RuntimeError(
        f"{name}: {int(np.sum(~np.asarray(done)))} games unfinished "
        f"after {max_len} plies"
    )
  return t1 - t0, t2 - t1, np.asarray(returns).mean(axis=0)


def main(argv):
  """Runs the comparison and the batched rollout for each game.

  Args:
    argv: The command line; no positional arguments.

  Raises:
    app.UsageError: there are positional arguments.
  """
  if len(argv) > 1:
    raise app.UsageError(f"unexpected arguments {argv[1:]}")
  for name in _GAMES.value:
    t0 = time.time()
    n = lockstep(name, _PLAYOUTS.value, _SEED.value)
    print(
        f"{name}: {_PLAYOUTS.value} random playouts agree with OpenSpiel at "
        f"all {n} states ({time.time() - t0:.1f} s)",
        flush=True,
    )
    compile_s, run_s, mean = batched_rollout(name, _BATCH.value, _SEED.value)
    print(
        f"{name}: {_BATCH.value} random games in one jitted vmap loop: "
        f"{run_s:.3f} s (compile {compile_s:.1f} s); mean returns "
        f"[{', '.join(f'{x:+.3f}' for x in mean)}]",
        flush=True,
    )


if __name__ == "__main__":
  app.run(main)
