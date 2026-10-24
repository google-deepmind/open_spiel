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

"""breakthrough, gomoku, hex, othello against pyspiel (open_spiel 1.6.12).

On random pyspiel playouts the env, driven through its raw layer with pyspiel's
action ids, has pyspiel's actor, legal set and terminal flag at every node and
pyspiel's returns at the end. Default sizes and a small size per game.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import registry

pyspiel = pytest.importorskip("pyspiel")

CASES = [
    # (pyspiel game string, env game, env options, playouts)
    ("breakthrough", "breakthrough", {}, 12),
    (
        "breakthrough(rows=6,columns=5)",
        "breakthrough",
        {"rows": 6, "columns": 5},
        60,
    ),
    ("gomoku", "gomoku", {}, 6),
    ("gomoku(size=7,connect=4)", "gomoku", {"board_size": 7, "connect": 4}, 60),
    ("hex", "hex", {}, 6),
    ("hex(board_size=5)", "hex", {"board_size": 5}, 60),
    ("othello", "othello", {}, 20),
]


def _check_node(env_node, st, where):
  actor, legal, term = (np.asarray(x) for x in env_node)
  assert bool(term) == st.is_terminal(), where
  if st.is_terminal():
    assert int(actor) == -1, where
    assert not legal.any(), where
    return
  assert int(actor) == st.current_player(), where
  assert np.flatnonzero(legal).tolist() == sorted(st.legal_actions()), where


@pytest.mark.parametrize(
    "game_string,name,kwargs,playouts", CASES, ids=[c[0] for c in CASES]
)
def test_random_playouts_match_pyspiel(game_string, name, kwargs, playouts):
  """Random playouts match pyspiel.

  Args:
    game_string: The parametrized case.
    name: The parametrized case.
    kwargs: The parametrized case.
    playouts: The parametrized case.
  """
  game = pyspiel.load_game(game_string)
  env = registry.make(name, **kwargs)
  assert env.num_actions == game.num_distinct_actions()
  assert env.max_len <= game.max_game_length()
  apply = jax.jit(env.apply)
  node = jax.jit(lambda s: (env.actor(s), env.legal_mask(s), env.terminal(s)))
  rets = jax.jit(env.returns)
  rng = np.random.default_rng(0)
  outcomes = set()
  longest = 0
  for g in range(playouts):
    st, s = game.new_initial_state(), env.root()
    assert not bool(env.is_chance(s))
    ply = 0
    while True:
      _check_node(node(s), st, (g, ply, st.history()))
      if st.is_terminal():
        break
      a = int(rng.choice(st.legal_actions()))
      st.apply_action(a)
      s = apply(s, jnp.int32(a))
      ply += 1
    np.testing.assert_array_equal(
        np.asarray(rets(s)), np.asarray(st.returns(), np.float32)
    )
    outcomes.add(tuple(st.returns()))
    longest = max(longest, ply)
  assert longest <= env.max_len
  assert (
      len(outcomes) >= 2
  ), outcomes  # both seats win somewhere (not a vacuous check)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
