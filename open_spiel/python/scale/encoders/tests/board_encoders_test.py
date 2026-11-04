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

"""Tests for the breakthrough / gomoku / hex / othello research encoders.

They are f32[H, W, 3] HWC planes [the seat's pieces, the opponent's, a plane
filled with the seat id] (the JAX engine's observation layout, absolute board
orientation for both seats). Known answers on hand-built positions; one encoder
per registered size; the batched form equals the unbatched one; a wrong-size
board fails loud.
"""

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import board_games
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import breakthrough
from open_spiel.python.scale.envs import gomoku
from open_spiel.python.scale.envs import hex as hex_env
from open_spiel.python.scale.envs import othello

F32 = np.float32


@functools.cache
def _jit(env):
  """Jitted (init, step, legal_mask, terminal) of an env (compiled once).

  Args:
    env: The env.

  Returns:
    (init, step, legal_mask, terminal), jitted.
  """
  return (
      jax.jit(env.init),
      jax.jit(env.step),
      jax.jit(env.legal_mask),
      jax.jit(env.terminal),
  )


def _play(env, actions):
  """Plays `actions` from the initial state.

  Args:
    env: The env.
    actions: The actions.

  Returns:
    The state.
  """
  init, step, _, _ = _jit(env)
  s = init(jax.random.key(0))
  for a in actions:
    s = step(s, jnp.int32(a), jax.random.key(0))
  return s


def test_registered_names_and_shapes():
  """Registered names and shapes."""
  want = {
      "breakthrough_planes": (8, 8, 3),
      "breakthrough_planes@columns=6,rows=6": (6, 6, 3),
      "gomoku_planes": (15, 15, 3),
      "gomoku_planes@board_size=9": (9, 9, 3),
      "gomoku_planes@board_size=7": (7, 7, 3),
      "hex_planes": (11, 11, 3),
      "hex_planes@board_size=9": (9, 9, 3),
      "hex_planes@board_size=7": (7, 7, 3),
      "hex_planes@board_size=5": (5, 5, 3),
      "othello_planes": (8, 8, 3),
  }
  for name, shape in want.items():
    enc = registry.get(name)
    assert enc.shape == shape and enc.dtype == jnp.float32, name
    assert enc.game == name.split("_", maxsplit=1)[0], name
  for game in ("breakthrough", "gomoku", "hex", "othello"):
    # one name per encoding
    assert set(registry.names(game=game)) == {f"{game}_planes"}
    assert f"{game}_planes" in registry.families()


def test_encoder_name_for_an_env():
  """Encoder name for an env."""
  assert board_games.planes_name(gomoku.GomokuEnv()) == "gomoku_planes"
  assert board_games.planes_name(gomoku.GomokuEnv(board_size=9)) == (
      "gomoku_planes@board_size=9"
  )
  assert board_games.planes_name(gomoku.GomokuEnv(board_size=9, connect=4)) == (
      "gomoku_planes@board_size=9"
  )
  assert (
      board_games.planes_name(hex_env.HexEnv(board_size=4))
      == "hex_planes@board_size=4"
  )
  assert board_games.planes_name(othello.OthelloEnv()) == "othello_planes"
  assert registry.get("hex_planes@board_size=4").shape == (4, 4, 3)


def test_hex_known_planes():
  """Hex known planes."""
  env = hex_env.HexEnv(board_size=5)
  s = _play(env, (0, 6, 24))  # x a1, o b2, x e5
  enc = registry.get("hex_planes@board_size=5")
  for seat in (0, 1):
    out = np.asarray(enc(env.info_state(s, seat)))
    mine, theirs = np.zeros((5, 5), F32), np.zeros((5, 5), F32)
    (mine if seat == 0 else theirs)[[0, 4], [0, 4]] = 1
    (theirs if seat == 0 else mine)[1, 1] = 1
    np.testing.assert_array_equal(out[..., 0], mine)
    np.testing.assert_array_equal(out[..., 1], theirs)
    np.testing.assert_array_equal(out[..., 2], np.full((5, 5), seat, F32))


def test_gomoku_known_planes():
  """Gomoku known planes."""
  env = gomoku.GomokuEnv(board_size=7, connect=4)
  s = _play(env, (8, 48))  # black (1, 1), white (6, 6)
  out = np.asarray(
      registry.get("gomoku_planes@board_size=7")(env.info_state(s, 1))
  )
  assert out[6, 6, 0] == 1 and out[..., 0].sum() == 1
  assert out[1, 1, 1] == 1 and out[..., 1].sum() == 1
  assert (out[..., 2] == 1).all()


def test_breakthrough_known_planes():
  """Breakthrough known planes."""
  env = breakthrough.BreakthroughEnv()
  s = _play(env, (breakthrough.action_id(1, 0, 1),))
  out = np.asarray(registry.get("breakthrough_planes")(env.info_state(s, 0)))
  mine = np.zeros((8, 8), F32)
  mine[0] = 1
  mine[1, 1:] = 1
  mine[2, 0] = 1
  theirs = np.zeros((8, 8), F32)
  theirs[6:] = 1
  np.testing.assert_array_equal(out[..., 0], mine)
  np.testing.assert_array_equal(out[..., 1], theirs)
  assert (out[..., 2] == 0).all()


def test_othello_known_planes():
  """Othello known planes."""
  env = othello.OthelloEnv()
  s = _play(env, (19,))  # black d3 flips d4
  out = np.asarray(registry.get("othello_planes")(env.info_state(s, 1)))
  black = {(2, 3), (3, 3), (3, 4), (4, 3)}
  white = {(4, 4)}
  assert {tuple(x) for x in np.argwhere(out[..., 1] == 1)} == black
  assert {tuple(x) for x in np.argwhere(out[..., 0] == 1)} == white
  assert (out[..., 2] == 1).all()


@pytest.mark.parametrize(
    "name,ctor",
    [
        (
            "breakthrough_planes@columns=6,rows=6",
            lambda: breakthrough.BreakthroughEnv(rows=6, columns=6),
        ),
        ("gomoku_planes@board_size=9", lambda: gomoku.GomokuEnv(board_size=9)),
        ("hex_planes@board_size=7", lambda: hex_env.HexEnv(board_size=7)),
        ("othello_planes", othello.OthelloEnv),
    ],
)
def test_batch_equals_unbatched_and_planes_are_one_hot(name, ctor):
  """Batch equals unbatched and planes are one hot."""
  env = ctor()
  enc = registry.get(name)
  s = env.init(jax.random.key(0))
  states = [s]
  for i in range(6):
    m = np.flatnonzero(np.asarray(env.legal_mask(s)))
    s = env.step(s, jnp.int32(m[(7 * i) % len(m)]), jax.random.key(0))
    states.append(s)
  infos = []
  for x in states:
    infos.extend(env.info_state(x, q) for q in (0, 1))
  batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *infos)
  out = np.asarray(enc.batch(batched))
  assert out.shape == (len(infos),) + enc.shape and out.dtype == np.float32
  for i, info in enumerate(infos):
    np.testing.assert_array_equal(out[i], np.asarray(enc(info)))
  assert (out[..., 0] + out[..., 1] <= 1).all()


def test_wrong_size_board_fails_loud():
  """Wrong size board fails loud."""
  info = hex_env.HexEnv(board_size=5).info_state(
      _play(hex_env.HexEnv(board_size=5), ()), 0
  )
  with pytest.raises((TypeError, ValueError)):
    registry.get("hex_planes")(info)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
