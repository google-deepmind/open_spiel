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

"""GoEnv tests.

The Env contract, known answers (Tromp-Taylor area score with komi, capture,
suicide), agreement with pyspiel `go` on random games (every node: actor, legal
set, terminal, returns), and the perfect-information world_from_info round trip.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import go
from open_spiel.python.scale.envs.tests import differential_vs_openspiel as pd
from open_spiel.python.scale.testing import check_env


@pytest.fixture(name="env9", scope="module")
def _env9_fixture():
  """A 9x9 go env.

  Returns:
    The env.
  """
  return go.GoEnv()


@pytest.fixture(name="env5", scope="module")
def _env5_fixture():
  """A 5x5 go env.

  Returns:
    The env.
  """
  return go.GoEnv(board_size=5)


def _play(env, actions):
  """Plays legal `actions` from the initial state.

  Args:
    env: The env.
    actions: The actions.

  Returns:
    The state.
  """
  s = env.init(jax.random.key(0))
  for a in actions:
    assert not bool(env.terminal(s)), actions
    assert bool(env.legal_mask(s)[a]), (actions, a)
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  return s


def test_defaults_and_sizes(env9):
  """Defaults and sizes.

  Args:
    env9: The module `env9` fixture.
  """
  assert (env9.board_size, env9.komi) == (9, 7.5)
  assert (
      env9.num_actions == 82 and env9.max_len == 162 and env9.num_players == 2
  )
  assert env9.pass_action == 81
  big = go.GoEnv(board_size=19)
  assert big.num_actions == 362 and big.max_len == 722
  for bad in (1, 0, 9.0, True, "9"):
    with pytest.raises((ValueError, TypeError)):
      go.GoEnv(board_size=bad)
  for bad in (float("nan"), float("inf"), "7.5", None):
    with pytest.raises((ValueError, TypeError)):
      go.GoEnv(komi=bad)


def test_check_env_9x9(env9):
  """Check env 9x9.

  Args:
    env9: The module `env9` fixture.
  """
  check_env.check_env(env9, num_playouts=16)


def test_check_env_5x5(env5):
  """Check env 5x5.

  Args:
    env5: The module `env5` fixture.
  """
  check_env.check_env(env5, num_playouts=64)


@pytest.mark.parametrize(
    "komi,actions,expected",
    [
        (7.5, (81, 81), [-1.0, 1.0]),  # empty board: white wins by komi
        (0.0, (81, 81), [0.0, 0.0]),  # empty board, no komi: a draw
        # one black stone owns the board: 81 - 7.5
        (7.5, (40, 81, 81), [1.0, -1.0]),
        (-0.5, (81, 81), [1.0, -1.0]),  # negative komi favours black
    ],
)
def test_known_terminal_returns(komi, actions, expected):
  """Known terminal returns.

  Args:
    komi: The parametrized case.
    actions: The parametrized case.
    expected: The parametrized case.
  """
  env = go.GoEnv(board_size=9, komi=komi)
  s = _play(env, actions)
  assert bool(env.terminal(s))
  assert int(env.actor(s)) == -1
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), expected)
  assert not bool(env.terminal(_play(env, actions[:-1])))


def test_capture_and_suicide(env5):
  """Capture and suicide.

  Args:
    env5: The module `env5` fixture.
  """
  # 5x5: B 1, W 0 (corner), B 5 captures the white corner stone.
  s = _play(env5, (1, 0, 5))
  info = env5.info_state(s, 1)
  b = np.asarray(info.board)
  assert b[0] == 0 and b[1] > 0 and b[5] > 0
  assert int(env5.actor(s)) == 1
  # The corner is now suicide for white (both neighbours black, not in atari).
  assert not bool(env5.legal_mask(s)[0])
  assert bool(env5.legal_mask(s)[env5.pass_action])
  # Black stones: 1, 5; the corner is black territory; white has nothing.
  s = _play(env5, (1, 0, 5, 25, 25))
  assert bool(env5.terminal(s))
  np.testing.assert_array_equal(np.asarray(env5.returns(s)), [1.0, -1.0])


def test_max_length_ends_the_game(env5):
  """A 5x5 game ends at move 2 * 25 = 50 without two passes, and is scored."""
  assert env5.max_len == 50
  s = _play(env5, (12,))  # one black stone in the centre, white to move
  s = s._replace(move_count=jnp.int32(48))  # as if 48 moves were played
  s = env5.step(s, jnp.int32(0), jax.random.key(0))  # move 49: white corner
  assert not bool(env5.terminal(s))
  s = env5.step(s, jnp.int32(24), jax.random.key(0))  # move 50: black corner
  assert bool(env5.terminal(s)) and int(env5.actor(s)) == -1
  # Black 2 stones, white 1, the rest is neutral (touches both): 2 - 1 - 7.5 <
  # 0.
  np.testing.assert_array_equal(np.asarray(env5.returns(s)), [-1.0, 1.0])


@pytest.mark.parametrize("board_size,games,seed", [(9, 24, 0), (5, 64, 1)])
def test_agrees_with_pyspiel_on_random_games(board_size, games, seed):
  """Agrees with pyspiel on random games.

  Args:
    board_size: The parametrized case.
    games: The parametrized case.
    seed: The parametrized case.
  """
  pyspiel = pytest.importorskip("pyspiel")
  env = go.GoEnv(board_size=board_size)
  game = pyspiel.load_game("go", {"board_size": board_size, "komi": 7.5})
  assert game.max_game_length() == env.max_len
  assert game.num_distinct_actions() == env.num_actions
  stats = pd.check_random_games(env, game, games, seed)
  assert stats["terminals"] == games
  assert len(stats["outcomes"]) == 2, stats  # both seats win some games


def test_world_from_info_round_trips(env5):
  """World from info round trips.

  Args:
    env5: The module `env5` fixture.
  """
  rng = np.random.default_rng(3)
  step = jax.jit(env5.step)
  s = env5.init(jax.random.key(0))
  states = [s]
  while not bool(env5.terminal(s)):
    legal = np.flatnonzero(np.asarray(env5.legal_mask(s)))
    s = step(s, jnp.int32(rng.choice(legal)), jax.random.key(0))
    states.append(s)
  states += [_play(env5, (1, 0, 5)), _play(env5, (1, 0, 5, 25, 25))]
  w_fn = jax.jit(env5.world_from_info)
  for st in states:
    for seat in (0, 1):
      w = w_fn(env5.info_state(st, seat))
      # The superko history is rebuilt from the position alone (the
      # InfoState has no earlier positions): every row up to move_count
      # holds the current position's hash. Every other field is exact.
      n = int(st.move_count)
      hist = np.asarray(w.position_hashes)
      assert w.position_hashes.dtype == st.position_hashes.dtype
      np.testing.assert_array_equal(hist[n], np.asarray(st.position_hashes)[n])
      np.testing.assert_array_equal(
          hist[: n + 1], np.broadcast_to(hist[n], (n + 1, 2))
      )
      assert not np.any(hist[n + 1 :])
      w = w._replace(position_hashes=st.position_hashes)
      for x, y in zip(
          jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(st)
      ):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_world_from_info_of_a_superko_end_is_over_and_a_draw():
  """World from info of a superko end is over and a draw."""
  # The engine's go_test 4x4 game whose 16th move recreates an earlier
  # position: OpenSpiel's positional superko ends it at once as a draw. The
  # InfoState says so (superko), and the rebuilt world is that terminal.
  env = go.GoEnv(board_size=4, komi=0.5)
  s = _play(env, (1, 11, 9, 7, 8, 3, 6, 12, 5, 10, 4, 15, 14, 13, 14))
  assert not bool(s.superko) and int(env.info_state(s, 0).superko) == 0
  s = env.step(s, jnp.int32(12), jax.random.key(0))
  assert bool(s.superko) and bool(env.terminal(s))
  assert int(s.consecutive_passes) < 2 and int(s.move_count) < env.max_len
  w_fn = jax.jit(env.world_from_info)
  for seat in (0, 1):
    info = env.info_state(s, seat)
    assert int(info.superko) == 1
    w = w_fn(info)
    assert bool(env.terminal(w)) and bool(w.superko) and int(w.winner) == -1
    assert not np.asarray(env.legal_mask(w)).any()
    np.testing.assert_array_equal(np.asarray(env.returns(w)), [0.0, 0.0])
    w = w._replace(position_hashes=s.position_hashes)
    for x, y in zip(jax.tree_util.tree_leaves(w), jax.tree_util.tree_leaves(s)):
      assert x.dtype == y.dtype
      np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_info_state_dtypes(env9):
  """Info state dtypes.

  Args:
    env9: The module `env9` fixture.
  """
  info = env9.info_state(_play(env9, (40,)), 1)
  for leaf in jax.tree_util.tree_leaves(info):
    assert leaf.dtype == jnp.int32
  assert (
      info.board.shape == (81,)
      and int(info.seat) == 1
      and int(info.move_count) == 1
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
