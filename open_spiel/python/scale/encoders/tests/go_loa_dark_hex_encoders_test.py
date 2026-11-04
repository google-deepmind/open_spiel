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

"""Tests for the Go, Lines of Action and Dark Hex encoders.

The per-encoder contract (declared shape and dtype under x64, batch ==
per-example, jit == eager, finite), hand-computed known answers, and agreement
with pyspiel's own tensors on positions replayed from random pyspiel games.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import dark_hex
from open_spiel.python.scale.envs import go
from open_spiel.python.scale.envs import lines_of_action as loa
from open_spiel.python.scale.envs.tests import differential_vs_openspiel as pd

ENCODERS = {
    "go_planes@board_size=5": lambda: go.GoEnv(board_size=5),
    "go_planes": go.GoEnv,
    "lines_of_action_planes": lambda: loa.LinesOfActionEnv(max_game_length=30),
    "dark_hex_info_state": dark_hex.DarkHexEnv,
    "dark_hex_observation_tensor": dark_hex.DarkHexEnv,
    "dark_hex_ir_view": dark_hex.DarkHexIREnv,
    "dark_hex_ir_observation_tensor": dark_hex.DarkHexIREnv,
}


def _play(env, actions):
  """Plays `actions` from the initial state.

  Args:
    env: The env.
    actions: The actions.

  Returns:
    The state.
  """
  s = env.init(jax.random.key(0))
  for a in actions:
    s = env.step(s, jnp.int32(a), jax.random.key(0))
  return s


def _random_infos(env, n_states=24, seed=0):
  """InfoStates of both seats along one random playout (stacked).

  Args:
    env: The env.
    n_states: Plies to record at most.
    seed: The seed.

  Returns:
    The stacked InfoStates.
  """
  rng = np.random.default_rng(seed)
  step = jax.jit(env.step)
  s = env.init(jax.random.key(0))
  infos = []
  while not bool(env.terminal(s)) and len(infos) < 2 * n_states:
    infos += [env.info_state(s, jnp.int32(q)) for q in (0, 1)]
    legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
    s = step(s, jnp.int32(rng.choice(legal)), jax.random.key(0))
  return jax.tree.map(lambda *x: jnp.stack(x), *infos)


def test_registered():
  """Registered."""
  assert all(registry.known(n) for n in ENCODERS)

  def go_name(n):
    """The go encoder name of board size `n`.

    Args:
      n: The board size.

    Returns:
      The name.
    """
    return "go_planes" if n == 9 else f"go_planes@board_size={n}"

  assert {registry.get(go_name(n)).shape for n in (5, 7, 9, 13, 19)} == {
      (n, n, 6) for n in (5, 7, 9, 13, 19)
  }
  # one name per encoding
  assert set(registry.names(game="go")) == {"go_planes"}
  assert set(registry.names(game="dark_hex")) == {
      "dark_hex_info_state",
      "dark_hex_observation_tensor",
  }
  assert set(registry.names(game="dark_hex_ir")) == {
      "dark_hex_ir_view",
      "dark_hex_ir_observation_tensor",
  }
  assert registry.get("go_planes@board_size=5").shape == (5, 5, 6)


@pytest.mark.parametrize("name", sorted(ENCODERS))
def test_contract(name):
  """Contract."""
  assert jax.config.jax_enable_x64
  enc = registry.get(name)
  infos = _random_infos(ENCODERS[name]())
  batch = enc.batch(infos)
  assert batch.dtype == enc.dtype == jnp.float32
  assert batch.shape[1:] == enc.shape
  assert np.isfinite(np.asarray(batch)).all()
  jitted = jax.jit(enc.fn)
  for i in (0, 1, batch.shape[0] - 1):
    one = jax.tree.map(lambda x, i=i: x[i], infos)
    np.testing.assert_array_equal(np.asarray(enc(one)), np.asarray(batch[i]))
    np.testing.assert_array_equal(np.asarray(jitted(one)), np.asarray(batch[i]))


def test_go_planes_known_answer():
  """Go planes known answer."""
  env = go.GoEnv(board_size=5)
  # B 1, W 0, B 5 captures, W passes; black to move
  s = _play(env, (1, 0, 5, 25))
  enc = registry.get("go_planes@board_size=5")
  x = np.asarray(enc(env.info_state(s, 0)))
  mine = np.zeros(25)
  mine[[1, 5]] = 1
  np.testing.assert_array_equal(x[..., 0].reshape(-1), mine)
  np.testing.assert_array_equal(x[..., 1], 0)
  np.testing.assert_array_equal(x[..., 2].reshape(-1), 1 - mine)
  np.testing.assert_array_equal(x[..., 3], 0)  # B 5 was not a ko capture
  np.testing.assert_array_equal(x[..., 4], 1)  # seat 0 is black
  np.testing.assert_array_equal(x[..., 5], 1)  # white just passed
  y = np.asarray(enc(env.info_state(s, 1)))
  np.testing.assert_array_equal(y[..., 1].reshape(-1), mine)
  np.testing.assert_array_equal(y[..., 4], 0)


def test_go_ko_plane():
  """Go ko plane.

  Plane 3 is one-hot at the state's ko point wherever one is set (random
  5x5 games that never pass while a stone is legal, so kos arise).
  """
  env = go.GoEnv(board_size=5)
  rng = np.random.default_rng(11)
  step = jax.jit(env.step)
  enc = registry.get("go_planes@board_size=5")
  found = 0
  for _ in range(40):
    s = env.init(jax.random.key(0))
    while not bool(env.terminal(s)):
      info = env.info_state(s, 0)
      ko = int(info.ko_point)
      plane = np.asarray(enc(info))[..., 3].reshape(-1)
      want = np.zeros(25)
      if ko >= 0:
        want[ko] = 1
        found += 1
      np.testing.assert_array_equal(plane, want)
      legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
      stones = legal[legal != env.pass_action]
      s = step(
          s,
          jnp.int32(rng.choice(stones if stones.size else legal)),
          jax.random.key(0),
      )
  assert found > 0, "no ko arose in 40 random 5x5 games"


def test_go_planes_match_pyspiel_stones():
  """Go planes match pyspiel stones."""
  pyspiel = pytest.importorskip("pyspiel")
  env = go.GoEnv(board_size=9)
  game = pyspiel.load_game("go", {"board_size": 9})
  enc = registry.get("go_planes")
  n = 0
  for node in pd.replay_random_games(env, game, 4, seed=2):
    if node.ref.is_terminal():
      continue
    seat = node.ref.current_player()
    ref = np.asarray(node.ref.observation_tensor(seat), np.float32).reshape(
        4, 9, 9
    )
    x = np.asarray(enc(env.info_state(node.ours, seat)))
    mine, theirs = (ref[0], ref[1]) if seat == 0 else (ref[1], ref[0])
    np.testing.assert_array_equal(x[..., 0], mine)
    np.testing.assert_array_equal(x[..., 1], theirs)
    np.testing.assert_array_equal(x[..., 2], ref[2])
    n += 1
  assert n > 20


def test_lines_of_action_planes_match_pyspiel():
  """Lines of action planes match pyspiel."""
  pyspiel = pytest.importorskip("pyspiel")
  env = loa.LinesOfActionEnv()
  game = pyspiel.load_game("lines_of_action")
  enc = registry.get("lines_of_action_planes")
  n = 0
  for node in pd.replay_random_games(env, game, 2, seed=3, max_moves=60):
    if node.ref.is_terminal():
      continue
    for seat in (0, 1):
      ref = np.asarray(node.ref.observation_tensor(seat), np.float32).reshape(
          3, 8, 8
      )
      x = np.asarray(enc(env.info_state(node.ours, seat)))
      mine, theirs = (ref[0], ref[1]) if seat == 0 else (ref[1], ref[0])
      np.testing.assert_array_equal(x[..., 0], mine)
      np.testing.assert_array_equal(x[..., 1], theirs)
      np.testing.assert_array_equal(x[..., 2], float(seat))
      n += 1
  assert n > 100


def test_dark_hex_known_vectors():
  """Dark hex known vectors."""
  env = dark_hex.DarkHexEnv()
  # black 4; white tries 4 (fails), white 0; black to move
  s = _play(env, (4, 4, 0))
  i1 = env.info_state(s, 1)
  x = np.asarray(registry.get("dark_hex_info_state")(i1))
  view = np.zeros((9, 9), np.float32)
  for cell, v in enumerate([-1, 0, 0, 0, 1, 0, 0, 0, 0]):
    view[cell, v + 4] = 1
  attempts = np.zeros((9, 9), np.float32)
  attempts[0, 4] = attempts[1, 0] = 1
  np.testing.assert_array_equal(
      x, np.concatenate([view.ravel(), attempts.ravel(), [0, 1]])
  )
  np.testing.assert_array_equal(
      np.asarray(registry.get("dark_hex_observation_tensor")(i1)), view.ravel()
  )


def test_dark_hex_ir_known_vectors():
  """Dark hex ir known vectors."""
  env = dark_hex.DarkHexIREnv()
  # black 4; white tries 4 (fails, turn passes); black 0
  s = _play(env, (4, 4, 0))
  i0 = env.info_state(s, 0)
  x = np.asarray(registry.get("dark_hex_ir_view")(i0))
  empty = np.ones(9)
  empty[[0, 4]] = 0
  black = 1 - empty
  np.testing.assert_array_equal(
      x, np.concatenate([empty, black, np.zeros(9), [1, 0]])
  )
  p = np.asarray(registry.get("dark_hex_ir_observation_tensor")(i0)).reshape(
      9, 9
  )
  want = np.zeros((9, 9))
  want[np.arange(9), np.where(black > 0, 5, 4)] = 1
  np.testing.assert_array_equal(p, want)


@pytest.mark.parametrize(
    "name,params", [("dark_hex", {}), ("dark_hex_ir", {"gameversion": "adh"})]
)
def test_dark_hex_encoders_match_pyspiel_tensors(name, params):
  """Dark hex encoders match pyspiel tensors.

  dark_hex: info_state[:162] == pyspiel information_state_tensor and the
  observation encoder == observation_tensor at every node; dark_hex_ir: the
  observation encoder == observation_tensor at every decision node.

  Args:
    name: the game name.
    params: its pyspiel parameters.
  """
  pyspiel = pytest.importorskip("pyspiel")
  env = dark_hex.DarkHexEnv() if name == "dark_hex" else dark_hex.DarkHexIREnv()
  game = pyspiel.load_game(name, params)
  n = 0
  for node in pd.replay_random_games(env, game, 60, seed=4):
    for seat in (0, 1):
      info = env.info_state(node.ours, seat)
      obs = np.asarray(node.ref.observation_tensor(seat), np.float32)
      if name == "dark_hex":
        ist = np.asarray(node.ref.information_state_tensor(seat), np.float32)
        x = np.asarray(registry.get("dark_hex_info_state")(info))
        np.testing.assert_array_equal(x[:162], ist)
        np.testing.assert_array_equal(x[162:], np.eye(2)[seat])
        np.testing.assert_array_equal(
            np.asarray(registry.get("dark_hex_observation_tensor")(info)), obs
        )
      elif not node.ref.is_terminal():
        np.testing.assert_array_equal(
            np.asarray(registry.get("dark_hex_ir_observation_tensor")(info)),
            obs,
        )
      n += 1
  assert n > 500


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
