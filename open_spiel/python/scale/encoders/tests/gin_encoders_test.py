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

"""Tests for gin_rummy_observation_tensor.

Shape, dtype under x64, known answers, and a
transcript error poisoning the encoding.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import x64
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.tests import gin_cards

P0 = "As 2s 3s 4c 5c 6c 7d 8d 9d Kh"
P1 = "6s 8s Ts Qs 2c 8c Tc Kc 3d 5d"
UPCARD = "Td"
O644 = "gin_rummy_observation_tensor"


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A gin env with Oklahoma rules.

  Returns:
    The env.
  """
  return gin.GinEnv(oklahoma_frac=1.0)


@pytest.fixture(name="deal", scope="module")
def _deal_fixture(env):
  """The state after dealing P0, P1 and the upcard.

  Args:
    env: The gin env.

  Returns:
    The state.
  """
  outcomes = (
      [0] + gin_cards.cards(P0) + gin_cards.cards(P1) + [gin_cards.card(UPCARD)]
  )
  return env.deal(jnp.asarray(outcomes, jnp.int32))


def test_registry_name_dims_and_game():
  """Registry name dims and game."""
  enc = registry.get(O644)
  assert enc.shape == (644,) and enc.game == "gin_rummy" and enc.name == O644


def test_dtype_shape_jit_vmap_under_x64(env, deal):
  """Dtype shape jit vmap under x64."""
  enc = registry.get(O644)
  info = env.info_state(deal, jnp.int32(0))
  x = enc(info)
  assert x.dtype == jnp.float32 and x.shape == (enc.shape[0],)
  batch = jax.tree_util.tree_map(lambda a: jnp.stack([a, a, a]), info)
  xb = jax.jit(enc.batch)(batch)
  assert xb.dtype == jnp.float32 and xb.shape == (3, enc.shape[0])
  np.testing.assert_array_equal(np.asarray(xb[1]), np.asarray(x))


def test_obs644_known_answer_at_the_known_deal(env, deal):
  """Obs644 known answer at the known deal."""
  x = np.asarray(registry.get(O644)(env.info_state(deal, jnp.int32(0))))
  want = np.zeros(644, np.float32)
  want[0] = 1  # player plane: seat 0
  want[2:54] = gin_cards.hand_mask(P0)  # own hand; seat 1's block stays zero
  want[106] = 1  # current player 0
  want[108:118] = 1  # knock card 10: thermometer of 10
  want[118 + gin_cards.card(UPCARD)] = 1
  want[222 : 222 + 31] = 1  # stock size 31
  np.testing.assert_array_equal(x, want)
  y = np.asarray(registry.get(O644)(env.info_state(deal, jnp.int32(1))))
  assert y[1] == 1 and y[0] == 0 and y[106] == 1
  np.testing.assert_array_equal(y[54:106], gin_cards.hand_mask(P1))
  assert not y[2:54].any()


if __name__ == "__main__":
  # The package runs with x64 on, as its entry points and conftest.py set
  # it; set here for `python <file>`, never at import (which changes no
  # process-global state).
  x64.enable_x64()
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
