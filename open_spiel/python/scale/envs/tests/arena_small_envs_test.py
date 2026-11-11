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

"""Integration: the core Arena over the small envs and the board games.

(each passes check_env alone) with the uniform policy (passes check_policy
alone).

Known answers: uniform-vs-uniform E[return of seat 0] for leduc and
tic_tac_toe, computed exactly by enumerating the pyspiel game tree; every hand
is zero-sum and completes; in phantom_ttt no seat ever sees the other seat's
move (Env.visible_action through the Arena).
"""

import functools
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import breakthrough
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import gomoku
from open_spiel.python.scale.envs import hex as hex_env
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs import othello
from open_spiel.python.scale.envs import phantom_ttt
from open_spiel.python.scale.envs import tic_tac_toe
from open_spiel.python.scale.policies import uniform

ENVS = {
    "leduc_poker": leduc.LeducEnv,
    "tic_tac_toe": tic_tac_toe.TicTacToeEnv,
    "connect_four": connect_four.ConnectFourEnv,
    "phantom_ttt": phantom_ttt.PhantomTTTEnv,
    "breakthrough": breakthrough.BreakthroughEnv,
    "gomoku": lambda: gomoku.GomokuEnv(board_size=9),
    "hex": lambda: hex_env.HexEnv(board_size=7),
    "othello": othello.OthelloEnv,
}


def _exact_uniform_value(game_name):
  """E[seat-0 return] under uniform-legal play, by exhaustive recursion.

  Args:
    game_name: The pyspiel game.

  Returns:
    The expected seat-0 return.
  """
  pyspiel = pytest.importorskip("pyspiel")
  game = pyspiel.load_game(game_name)

  @functools.cache
  def value(hist):
    """Exact uniform-play value for seat 0 after `hist`.

    Args:
      hist: The action history.

    Returns:
      E[return of seat 0].
    """
    st = game.new_initial_state()
    for a in hist:
      st.apply_action(a)
    if st.is_terminal():
      return st.returns()[0]
    if st.is_chance_node():
      return sum(p * value(hist + (a,)) for a, p in st.chance_outcomes())
    acts = st.legal_actions()
    return sum(value(hist + (a,)) for a in acts) / len(acts)

  return value(())


def _collect(env, b, t, seed, all_seats=False):
  """Uniform self-play rows.

  Args:
    env: The env.
    b: The rows.
    t: The plies.
    seed: The seed.
    all_seats: Whether the table covers every seat.

  Returns:
    The arena's collect output.
  """
  pol = uniform.UniformPolicy(env.name, env.num_actions, env.num_players)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  table = arena_lib.self_play_table(b, 2, all_seats=all_seats)
  dk, pk = arena_lib.independent_keys(jax.random.key(seed), b, 2)
  params = [
      jax.tree_util.tree_map(
          lambda x: x[None], pol.init_params(jax.random.key(0))
      )
  ]
  return ar.collect(params, table, dk, pk, t)


@pytest.mark.parametrize("game", sorted(ENVS))
def test_uniform_self_play_hands_complete_and_zero_sum(game):
  """Uniform self play hands complete and zero sum.

  Args:
    game: The parametrized case.
  """
  env = ENVS[game]()
  _, batch, log = _collect(env, 64, 3 * (env.max_len + 1), seed=1)
  done = np.asarray(log.done)
  assert done.sum() >= 64
  r = np.asarray(log.rewards)[done]
  np.testing.assert_allclose(r.sum(-1), 0.0)
  assert not np.asarray(log.bad).any()
  if game != "leduc_poker":
    assert np.isin(r[:, 0], [-1.0, 0.0, 1.0]).all()
  assert batch.action.dtype == np.int32


@pytest.mark.parametrize(
    "game_name,ctor",
    [
        ("leduc_poker", leduc.LeducEnv),
        ("tic_tac_toe", tic_tac_toe.TicTacToeEnv),
    ],
)
def test_uniform_self_play_matches_exact_value(game_name, ctor):
  """Uniform self play matches exact value.

  Args:
    game_name: The parametrized case.
    ctor: The parametrized case.
  """
  exact = _exact_uniform_value(game_name)
  env = ctor()
  _, _, log = _collect(env, 2048, 4 * (env.max_len + 1), seed=2)
  r0 = np.asarray(log.rewards)[..., 0][np.asarray(log.done)]
  assert r0.size > 4000
  se = r0.std() / np.sqrt(r0.size)
  assert abs(r0.mean() - exact) < 4 * se, (r0.mean(), exact, se)
  if game_name == "tic_tac_toe":
    assert abs(exact - (0.5849 - 0.2881)) < 1e-3  # known: X 58.49%, O 28.81%


def test_phantom_moves_are_hidden_from_the_other_seat():
  """Phantom moves are hidden from the other seat."""
  env = phantom_ttt.PhantomTTTEnv()
  _, batch, _ = _collect(env, 32, 2 * (env.max_len + 1), seed=3, all_seats=True)
  st = batch.steps
  seat = np.asarray(st.seat)
  la, lact = np.asarray(st.last_action), np.asarray(st.last_actor)
  other = (lact >= 0) & (lact != seat)
  own = lact == seat
  assert other.any() and own.any()
  assert (la[other] == types.NO_ACTION).all()
  assert (la[own] >= 0).all()


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
