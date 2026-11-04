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

"""Test helper: random pyspiel games replayed through an Env's raw layer.

A game is played in pyspiel (the reference rules) with uniform-random chance
and player choices; every chance outcome and action is replayed through the
env's RAW layer (root / apply / apply_chance), and at every resolved node
(decision or terminal) `assert_same_node` checks that the env agrees with
pyspiel: terminal flag, actor (-1 at terminal), the legal set, and returns
at terminal. This is what serving needs (serve/wire.check_replayed_node
checks actor, legal set and terminal on every request) plus the returns.
"""

from typing import Any, Iterator, NamedTuple, Optional

import jax
import numpy as np


class Node(NamedTuple):
  """One resolved node of a replayed game."""

  ref: object  # the pyspiel state (do not mutate)
  ours: object  # the env state
  moves: int  # player decisions played so far


class _Fns:

  def __init__(self, env):
    self.root = jax.jit(env.root)
    self.apply = jax.jit(env.apply)
    self.apply_chance = jax.jit(env.apply_chance)
    self.node = jax.jit(
        lambda s: (
            env.is_chance(s),
            env.terminal(s),
            env.actor(s),
            env.legal_mask(s),
            env.returns(s),
        )
    )


_CACHE: dict[int, tuple[Any, "_Fns"]] = {}


def _fns(env) -> _Fns:
  """The jitted helpers of `env`, built once per env.

  Args:
    env: The env.

  Returns:
    The env's _Fns.
  """
  hit = _CACHE.get(id(env))
  if hit is None or hit[0] is not env:
    hit = (env, _Fns(env))
    _CACHE[id(env)] = hit
  return hit[1]


def replay_random_games(
    env, game, num_games: int, seed: int, max_moves: Optional[int] = None
) -> Iterator[Node]:
  """Yields every resolved node of num_games random pyspiel games of `game`.

  (a loaded pyspiel game), with the env state replayed alongside. A game is cut
  after max_moves player decisions (None: played to the end).

  Args:
    env: The env.
    game: The loaded pyspiel game.
    num_games: Number of games.
    seed: The seed.
    max_moves: Player decisions per game (None: to the end).

  Yields:
    The Nodes, with the env state replayed alongside.
  """
  f = _fns(env)
  rng = np.random.default_rng(seed)
  for _ in range(num_games):
    st = game.new_initial_state()
    ours = f.root()
    moves = 0
    while True:
      is_chance = bool(f.node(ours)[0])
      if st.is_chance_node():
        assert (
            is_chance
        ), f"pyspiel at a chance node, the env is not: {st.history()}"
        outs = st.chance_outcomes()
        ids = np.array([o for o, _ in outs])
        probs = np.array([p for _, p in outs], np.float64)
        o = int(rng.choice(ids, p=probs / probs.sum()))
        st.apply_action(o)
        ours = f.apply_chance(ours, np.int32(o))
        continue
      assert (
          not is_chance
      ), f"the env is at a chance node, pyspiel is not: {st.history()}"
      yield Node(st, ours, moves)
      if st.is_terminal() or (max_moves is not None and moves >= max_moves):
        break
      a = int(rng.choice(st.legal_actions()))
      st.apply_action(a)
      ours = f.apply(ours, np.int32(a))
      moves += 1


def assert_same_node(env, node: Node) -> None:
  """Raises AssertionError unless the env node is pyspiel's node.

  Args:
    env: The env.
    node: The node.

  Raises:
    AssertionError: The env node differs from pyspiel's.
  """
  _, term, actor, legal, rets = (
      np.asarray(x) for x in _fns(env).node(node.ours)
  )
  st = node.ref
  where = f"after {node.moves} moves, history {st.history()}"
  assert (
      bool(term) == st.is_terminal()
  ), f"terminal {bool(term)} != pyspiel {st.is_terminal()} {where}"
  if st.is_terminal():
    assert int(actor) == -1, f"actor {int(actor)} at terminal {where}"
    assert not legal.any(), f"non-empty legal mask at terminal {where}"
    np.testing.assert_array_equal(
        rets, np.asarray(st.returns(), np.float32), err_msg=where
    )
    return
  assert (
      int(actor) == st.current_player()
  ), f"actor {int(actor)} != pyspiel {st.current_player()} {where}"
  want = np.zeros(env.num_actions, np.bool_)
  want[st.legal_actions()] = True
  if not np.array_equal(legal, want):
    raise AssertionError(
        f"legal set differs {where}: env-only "
        f"{np.flatnonzero(legal & ~want)[:10]}, "
        f"pyspiel-only {np.flatnonzero(want & ~legal)[:10]}"
    )
  np.testing.assert_array_equal(
      rets, np.zeros(env.num_players, np.float32), err_msg=where
  )


def check_random_games(
    env, game, num_games: int, seed: int, max_moves: Optional[int] = None
) -> dict[str, Any]:
  """assert_same_node at every node of num_games random games.

  Returns {"nodes", "terminals", "outcomes": {returns tuple: count}} so callers
  can assert the games were not vacuous.

  Args:
    env: The env.
    game: The loaded pyspiel game.
    num_games: Number of games.
    seed: The seed.
    max_moves: Player decisions per game (None: to the end).

  Returns:
    {"nodes", "terminals", "outcomes": {returns tuple: count}}.
  """
  nodes = terminals = 0
  outcomes: dict[tuple[float, ...], int] = {}
  for node in replay_random_games(env, game, num_games, seed, max_moves):
    assert_same_node(env, node)
    nodes += 1
    if node.ref.is_terminal():
      terminals += 1
      r = tuple(node.ref.returns())
      outcomes[r] = outcomes.get(r, 0) + 1
  return {"nodes": nodes, "terminals": terminals, "outcomes": outcomes}
