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

"""LinesOfActionEnv tests.

The Env contract, known answers, and agreement with pyspiel `lines_of_action`
(OpenSpiel's rules) on random games.

The env applies three OpenSpiel rules where the JAX engine differs; each
has a known-answer test here:
  * recreating a board seen before (the initial board included) is a draw;
  * reaching max_game_length is a draw even when the last move connects;
  * connectivity is a full flood fill (the engine's runs 8 steps, so a long
    connected snake is not seen as a win).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.jax.games import lines_of_action as loa_engine
from open_spiel.python.scale.envs import lines_of_action as loa
from open_spiel.python.scale.envs.tests import differential_vs_openspiel as pd
from open_spiel.python.scale.testing import check_env

# (from_row, from_col, to_row, to_col, capture) -> action id
A = loa.encode_action  # pylint: disable=invalid-name
# Black (0, 1) down to (2, 1), white (1, 0) across to (1, 2), both back: the
# board after move 4 is the initial board.
REPEAT = (
    A(0, 1, 2, 1, 0),
    A(1, 0, 1, 2, 0),
    A(2, 1, 0, 1, 0),
    A(1, 2, 1, 0, 0),
)
# Black out and back, white (1, 7) to (1, 5), black out again, white back: the
# board after move 6 is the board after move 2 (not the initial board).
REPEAT_MID = (
    A(0, 1, 2, 1, 0),
    A(1, 0, 1, 2, 0),
    A(2, 1, 0, 1, 0),
    A(1, 7, 1, 5, 0),
    A(0, 1, 2, 1, 0),
    A(1, 5, 1, 7, 0),
)


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A lines of action env.

  Returns:
    The env.
  """
  return loa.LinesOfActionEnv()


@pytest.fixture(name="short", scope="module")
def _short_fixture():
  """A lines of action env with a 40-ply limit.

  Returns:
    The env.
  """
  return loa.LinesOfActionEnv(max_game_length=40)


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


def test_defaults(env):
  """Defaults.

  Args:
    env: The module `env` fixture.
  """
  assert env.max_game_length == 1000 and env.max_len == 1000
  assert env.num_actions == 8192 and env.num_players == 2
  for bad in (0, -1, 10.0, True):
    with pytest.raises((ValueError, TypeError)):
      loa.LinesOfActionEnv(max_game_length=bad)


def test_encode_action_is_pyspiel_mixed_base():
  """Encode action is pyspiel mixed base."""
  assert A(0, 0, 0, 0, 0) == 0 and A(0, 0, 0, 0, 1) == 1
  assert A(1, 7, 1, 6, 0) == (((1 * 8 + 7) * 8 + 1) * 8 + 6) * 2
  assert A(7, 7, 7, 7, 1) == 8191


def test_check_env_short(short):
  """Check env short.

  Args:
    short: The module `short` fixture.
  """
  check_env.check_env(short, num_playouts=32)


def test_check_env_default(env):
  """Check env default.

  Args:
    env: The module `env` fixture.
  """
  check_env.check_env(env, num_playouts=4)


def test_repeated_board_is_a_draw(env):
  """Repeated board is a draw.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, REPEAT)
  assert bool(env.terminal(s)) and int(env.actor(s)) == -1
  assert not bool(env.legal_mask(s).any())
  np.testing.assert_array_equal(np.asarray(env.returns(s)), [0.0, 0.0])
  assert not bool(env.terminal(_play(env, REPEAT[:3])))


def test_repeating_a_later_board_is_a_draw(env):
  """Repeating a later board is a draw.

  Args:
    env: The module `env` fixture.
  """
  s = _play(env, REPEAT_MID)
  assert bool(env.terminal(s))
  np.testing.assert_array_equal(np.asarray(env.returns(s)), [0.0, 0.0])
  assert not bool(env.terminal(_play(env, REPEAT_MID[:5])))


def test_max_game_length_is_a_draw():
  """Max game length is a draw."""
  env = loa.LinesOfActionEnv(max_game_length=3)
  assert env.max_len == 3
  s = _play(env, REPEAT[:3])
  assert bool(env.terminal(s))
  np.testing.assert_array_equal(np.asarray(env.returns(s)), [0.0, 0.0])
  assert not bool(env.terminal(_play(env, REPEAT[:2])))


def _snake_state(env, max_game_length=None):
  """A position one move from a 12-piece black connection.

  Black: row 0 cols 0-5, col 5 rows 2-6, and one piece at (1, 7); white
  at (7, 0) and (7, 7). Black to move (1, 7) -> (1, 6) joins the snake: 12
  connected black pieces, 11 king steps from the first to the last.

  Args:
    env: The lines_of_action env.
    max_game_length: The game's move limit, to set the position one move
      before it; None: move 10.

  Returns:
    The position.
  """
  board = np.full(64, -1, np.int32)
  for c in range(6):
    board[c] = 0
  for r in range(2, 7):
    board[r * 8 + 5] = 0
  board[1 * 8 + 7] = 0
  board[56] = board[63] = 1
  assert (board == 0).sum() == 12
  info = loa.LinesOfActionInfo(
      seat=jnp.int32(0),
      board=jnp.asarray(board),
      move_count=jnp.int32(
          10 if max_game_length is None else max_game_length - 1
      ),
  )
  return env.world_from_info(info)


def test_long_snake_connects_and_wins(env):
  """Long snake connects and wins.

  Args:
    env: The module `env` fixture.
  """
  s = _snake_state(env)
  assert not bool(env.terminal(s)) and int(env.actor(s)) == 0
  a = A(1, 7, 1, 6, 0)
  assert bool(env.legal_mask(s)[a])
  s2 = env.step(s, jnp.int32(a), jax.random.key(0))
  assert bool(env.terminal(s2))
  np.testing.assert_array_equal(np.asarray(env.returns(s2)), [1.0, -1.0])
  # The engine alone sees the same win.
  raw = loa_engine.LinesOfAction()
  g2 = raw.apply_action(s.game, jnp.int32(a))
  assert bool(g2.is_terminal) and int(g2.winner) == 0


def test_connecting_on_the_last_move_is_still_a_draw():
  """Connecting on the last move is still a draw."""
  env = loa.LinesOfActionEnv(max_game_length=11)
  s = _snake_state(env, max_game_length=11)
  s2 = env.step(s, jnp.int32(A(1, 7, 1, 6, 0)), jax.random.key(0))
  assert bool(env.terminal(s2))
  np.testing.assert_array_equal(np.asarray(env.returns(s2)), [0.0, 0.0])


def _pyspiel_game(max_game_length):
  """The pyspiel lines_of_action game at max_game_length, or None.

  When this pyspiel has no max_game_length parameter (1.6.12 does not; its limit
  is 1000).

  Args:
    max_game_length: The game's move limit.

  Returns:
    The pyspiel game, or None.
  """
  pyspiel = pytest.importorskip("pyspiel")
  try:
    return pyspiel.load_game(
        "lines_of_action", {"max_game_length": max_game_length}
    )
  except pyspiel.SpielError:
    if max_game_length == loa.DEFAULT_MAX_GAME_LENGTH:
      return pyspiel.load_game("lines_of_action")
    return None


def test_known_answers_agree_with_pyspiel():
  """Known answers agree with pyspiel."""
  for seq in (REPEAT, REPEAT_MID):
    st = _pyspiel_game(1000).new_initial_state()
    for a in seq:
      assert not st.is_terminal()
      st.apply_action(a)
    assert st.is_terminal() and st.returns() == [0.0, 0.0]
  game = _pyspiel_game(3)
  if game is not None:
    st = game.new_initial_state()
    for a in REPEAT[:3]:
      st.apply_action(a)
    assert st.is_terminal() and st.returns() == [0.0, 0.0]


@pytest.mark.parametrize(
    "max_game_length,games,seed", [(1000, 24, 0), (40, 48, 1)]
)
def test_agrees_with_pyspiel_on_random_games(max_game_length, games, seed):
  """Agrees with pyspiel on random games.

  Args:
    max_game_length: The parametrized case.
    games: The parametrized case.
    seed: The parametrized case.
  """
  env = loa.LinesOfActionEnv(max_game_length=max_game_length)
  game = _pyspiel_game(max_game_length)
  max_moves = None
  if (
      game is None
  ):  # compare the nodes before the env's limit with pyspiel's 1000
    game, max_moves = _pyspiel_game(1000), max_game_length - 1
  else:
    assert game.max_game_length() == env.max_len
  assert game.num_distinct_actions() == env.num_actions
  stats = pd.check_random_games(env, game, games, seed, max_moves=max_moves)
  assert stats["nodes"] > 20 * games
  if max_moves is None:
    assert stats["terminals"] == games
    if max_game_length >= 1000:  # random 40-move games all reach the limit
      assert len(stats["outcomes"]) >= 2, stats


def test_world_from_info_round_trips(short):
  """World from info round trips.

  Args:
    short: The module `short` fixture.
  """
  rng = np.random.default_rng(5)
  step = jax.jit(short.step)
  w_fn = jax.jit(short.world_from_info)
  info_fn = jax.jit(short.info_state)
  s = short.init(jax.random.key(0))
  n = 0
  while not bool(short.terminal(s)):
    for seat in (0, 1):
      info = info_fn(s, jnp.int32(seat))
      w = w_fn(info)

      # The world is the same position; the repetition history before it is
      # not part of the InfoState, so only the position must round-trip.
      def position(g):
        """The game without its board history.

        Args:
          g: The game state.

        Returns:
          The game.
        """
        # Reaches private _replace of the code under test.
        return g._replace(board_history=None)  # pylint: disable=protected-access

      for x, y in zip(
          jax.tree_util.tree_leaves(position(w.game)),
          jax.tree_util.tree_leaves(position(s.game)),
      ):
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
      for x, y in zip(
          jax.tree_util.tree_leaves(info_fn(w, jnp.int32(seat))),
          jax.tree_util.tree_leaves(info),
      ):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    legal = np.flatnonzero(np.asarray(short.legal_mask(s)))
    s = step(s, jnp.int32(rng.choice(legal)), jax.random.key(0))
    n += 1
  assert n > 5


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
