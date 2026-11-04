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

"""Tests for the encoder registry and the per-encoder contract.

The contract: declared shape and dtype (under jax_enable_x64), batch ==
per-example, jit == eager, deterministic, finite and seat-dependent; families
name their members canonically.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs import phantom_ttt
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs import tic_tac_toe
from open_spiel.python.scale.testing import check_env

ENVS = {
    "leduc_poker": leduc.LeducEnv,
    "tic_tac_toe": tic_tac_toe.TicTacToeEnv,
    "connect_four": connect_four.ConnectFourEnv,
    "phantom_ttt": phantom_ttt.PhantomTTTEnv,
}
EXPECTED = {
    "leduc_poker_info_state",
    "leduc_poker_observation_tensor",
    "tic_tac_toe_planes",
    "tic_tac_toe_observation_tensor",
    "connect_four_planes",
    "connect_four_player_planes",
    "connect_four_flat_planes",
    "phantom_ttt_info_state",
    "phantom_ttt_observation",
}


def test_x64_is_on():
  """X64 is on."""
  assert jax.config.jax_enable_x64, "run with JAX_ENABLE_X64=1"


def test_builtin_names_per_game():
  """Builtin names per game."""
  assert EXPECTED <= set(registry.names())
  assert set(registry.names(game="connect_four")) == {
      "connect_four_planes",
      "connect_four_player_planes",
      "connect_four_flat_planes",
  }
  assert {"go_planes", "gomoku_planes", "bargaining_info_state"} <= set(
      registry.families()
  )


def test_registry_rejects_duplicate_and_unknown_and_checks_dtype():
  """Registry rejects duplicate and unknown and checks dtype."""
  with pytest.raises(ValueError, match="already registered"):
    registry.register(registry.get("tic_tac_toe_planes"))
  with pytest.raises(KeyError, match="unknown encoder"):
    registry.get("no_such_encoder")
  bad = registry.Encoder(
      name="bad_dtype",
      game="g",
      shape=(1,),
      dtype=jnp.float32,
      fn=lambda info: jnp.zeros((1,), jnp.float64),
  )
  with pytest.raises(AssertionError, match="declared"):
    bad(None)
  with pytest.raises(ValueError, match="shape"):
    registry.Encoder(
        name="x", game="g", shape=(0,), dtype=jnp.float32, fn=lambda i: i
    )


def test_family_names_are_canonical():
  """Family names are canonical."""
  assert (
      registry.family_name("go_planes", env_registry.spec("go")) == "go_planes"
  )
  spec = env_registry.spec("go", board_size=13, komi=6.5)
  assert registry.family_name("go_planes", spec) == "go_planes@board_size=13"
  assert registry.get("go_planes@board_size=13").shape == (13, 13, 6)
  assert registry.get("go_planes").shape == (9, 9, 6)
  assert registry.get("gomoku_planes@board_size=9").shape == (9, 9, 3)
  assert registry.get("gomoku_planes") is registry.get("gomoku_planes")
  b = registry.get("breakthrough_planes@columns=5,rows=6")
  assert b.shape == (6, 5, 3) and b.game == "breakthrough"
  with pytest.raises(KeyError, match="not canonical"):
    registry.get("breakthrough_planes@rows=6,columns=5")
  with pytest.raises(KeyError, match="depends only on"):
    registry.get("go_planes@komi=6.5")
  assert registry.known("go_planes@board_size=13") and not registry.known(
      "chess_planes"
  )


def test_family_member_encodes_its_env():
  """Family member encodes its env."""
  env = env_registry.make("bargaining", max_turns=12)
  enc = registry.get("bargaining_info_state@max_turns=12")
  s = env.init(jax.random.key(0))
  out = enc(env.info_state(s, 0))
  assert out.shape == enc.shape == (12 + 2 + 24 + 33 + 12 * 24 + 2,)


@pytest.fixture(name="states", scope="module")
def _states_fixture():
  """Every state of random hands of each game.

  Returns:
    {game: (env, states)}.
  """
  # Reaches private _playout of the code under test.
  # pylint: disable=protected-access
  out = {}
  for game, ctor in ENVS.items():
    env = ctor()
    kd, kp = jax.random.split(jax.random.key(5))
    traj = jax.jit(
        jax.vmap(lambda d, q, env=env: check_env._playout(env, d, q))
    )(jax.random.split(kd, 32), jax.random.split(kp, 32))
    flat = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), traj
    )
    out[game] = (env, flat)
  return out


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_encoder_contract(states, name):
  """Encoder contract."""
  enc = registry.get(name)
  env, flat = states[enc.game]
  n = jax.tree_util.tree_leaves(flat)[0].shape[0]
  for seat in (0, 1):
    info = jax.vmap(env.info_state)(flat, jnp.full((n,), seat, jnp.int32))
    eager = enc.batch(info)
    jitted = jax.jit(enc.batch)(info)
    assert eager.shape == (n,) + enc.shape and eager.dtype == enc.dtype
    np.testing.assert_array_equal(
        np.asarray(eager, np.float32), np.asarray(jitted, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(jitted, np.float32),
        np.asarray(jax.jit(enc.batch)(info), np.float32),
    )
    assert np.isfinite(np.asarray(eager, np.float32)).all()
    one = enc(jax.tree_util.tree_map(lambda x: x[3], info))
    np.testing.assert_array_equal(
        np.asarray(one, np.float32), np.asarray(eager[3], np.float32)
    )
  # Every encoder here depends on the seat: the two seats' encodings differ.
  info0 = jax.vmap(env.info_state)(flat, jnp.zeros((n,), jnp.int32))
  info1 = jax.vmap(env.info_state)(flat, jnp.ones((n,), jnp.int32))
  assert not np.array_equal(
      np.asarray(enc.batch(info0), np.float32),
      np.asarray(enc.batch(info1), np.float32),
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
