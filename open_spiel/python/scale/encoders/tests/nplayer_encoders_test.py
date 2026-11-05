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

"""Encoders of the N-player test games.

The kuhn_nplayer tensors equal pyspiel kuhn_poker(players=N)'s on every node of
random games, family names and shapes follow the options, and the matrix seat
encoder is one_hot(seat).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import kuhn_nplayer
from open_spiel.python.scale.envs import nplayer_matrix
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs.tests import differential_vs_openspiel


def _enc(family, game, **kw):
  """The registered encoder of `family` for game `game`.

  Args:
    family: The encoder family.
    game: The env name.
    **kw: The env options.

  Returns:
    The Encoder.
  """
  return registry.get(
      registry.family_name(family, env_registry.spec(game, **kw))
  )


def test_family_names_and_shapes():
  """Family names and shapes."""
  assert _enc("kuhn_nplayer_info_state_tensor", "kuhn_nplayer").name == (
      "kuhn_nplayer_info_state_tensor"
  )
  e4 = _enc("kuhn_nplayer_info_state_tensor", "kuhn_nplayer", players=4)
  assert e4.name == "kuhn_nplayer_info_state_tensor@players=4" and e4.shape == (
      23,
  )
  assert _enc(
      "kuhn_nplayer_observation_tensor", "kuhn_nplayer", players=4
  ).shape == (13,)
  assert registry.get("kuhn_nplayer_observation_tensor").shape == (10,)
  m = _enc("nplayer_matrix_seat", "nplayer_matrix", variant="team_pennies_2v2")
  assert (
      m.name == 'nplayer_matrix_seat@variant="team_pennies_2v2"'
      and m.shape == (4,)
  )
  assert registry.get("nplayer_matrix_seat").shape == (3,)


def test_kuhn_three_player_known_answer():
  """Kuhn three player known answer."""
  env = kuhn_nplayer.KuhnNPlayerEnv(3)
  # p, b
  s = env.apply(env.apply(env.deal(2, 0, 3), jnp.int32(0)), jnp.int32(1))
  x = np.asarray(
      registry.get("kuhn_nplayer_info_state_tensor")(env.info_state(s, 2))
  )
  np.testing.assert_array_equal(
      x, [0, 0, 1, 0, 0, 0, 1] + [1, 0, 0, 1] + [0] * 6
  )
  o = np.asarray(
      registry.get("kuhn_nplayer_observation_tensor")(env.info_state(s, 2))
  )
  np.testing.assert_array_equal(o, [0, 0, 1, 0, 0, 0, 1, 1, 2, 1])


def test_an_encoder_refuses_another_player_count():
  """An encoder refuses another player count."""
  info = kuhn_nplayer.KuhnNPlayerEnv(4).info_state(
      kuhn_nplayer.KuhnNPlayerEnv(4).deal(0, 1, 2, 3), jnp.int32(0)
  )
  with pytest.raises(ValueError, match="bets of shape"):
    registry.get("kuhn_nplayer_info_state_tensor")(info)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_kuhn_tensors_equal_pyspiel(n):
  """Kuhn tensors equal pyspiel."""
  pyspiel = pytest.importorskip("pyspiel")
  env = kuhn_nplayer.KuhnNPlayerEnv(n)
  game = pyspiel.load_game("kuhn_poker", {"players": n})
  info_enc = _enc("kuhn_nplayer_info_state_tensor", "kuhn_nplayer", players=n)
  obs_enc = _enc("kuhn_nplayer_observation_tensor", "kuhn_nplayer", players=n)
  assert info_enc.shape == (game.information_state_tensor_size(),)
  assert obs_enc.shape == (game.observation_tensor_size(),)
  seats = jnp.arange(n, dtype=jnp.int32)
  both = jax.jit(
      lambda s: (
          info_enc.batch(jax.vmap(lambda q: env.info_state(s, q))(seats)),
          obs_enc.batch(jax.vmap(lambda q: env.info_state(s, q))(seats)),
      )
  )
  nodes = 0
  for node in differential_vs_openspiel.replay_random_games(
      env, game, 150, seed=10 + n
  ):
    xi, xo = (np.asarray(x) for x in both(node.ours))
    for q in range(n):
      where = f"seat {q} history {node.ref.history()}"
      np.testing.assert_array_equal(
          xi[q], node.ref.information_state_tensor(q), err_msg=where
      )
      np.testing.assert_array_equal(
          xo[q], node.ref.observation_tensor(q), err_msg=where
      )
    nodes += 1
  assert nodes > 300


def test_matrix_seat_encoder():
  """Matrix seat encoder."""
  env = nplayer_matrix.NPlayerMatrixEnv("odd_one_out_4p")
  enc = _enc("nplayer_matrix_seat", "nplayer_matrix", variant="odd_one_out_4p")
  s = env.apply(env.root(), jnp.int32(1))
  for q in range(4):
    np.testing.assert_array_equal(
        np.asarray(enc(env.info_state(s, jnp.int32(q)))),
        np.eye(4, dtype=np.float32)[q],
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
