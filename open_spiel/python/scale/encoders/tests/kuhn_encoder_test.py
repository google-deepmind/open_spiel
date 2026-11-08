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

"""Tests for Kuhn encoders.

Known answers of the observation tensor and the
information-state index.
"""

import sys

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import kuhn


def test_kuhn_observation_known_answers():
  """Kuhn observation known answers."""
  env = kuhn.KuhnEnv()
  enc = registry.get("kuhn_poker_observation_tensor")
  s = env.deal(2, 0)  # P0 K, P1 J
  np.testing.assert_array_equal(
      np.asarray(enc(env.info_state(s, 0))), [1, 0, 0, 0, 1, 1, 1]
  )
  s = env.game.apply_action(s, jnp.int32(1))  # P0 bets
  np.testing.assert_array_equal(
      np.asarray(enc(env.info_state(s, 1))), [0, 1, 1, 0, 0, 2, 1]
  )
  s = env.game.apply_action(
      env.game.apply_action(env.deal(1, 2), jnp.int32(0)), jnp.int32(1)
  )
  np.testing.assert_array_equal(
      np.asarray(enc(env.info_state(s, 0))), [1, 0, 0, 1, 0, 1, 2]
  )


def test_kuhn_info_index_encoder_is_one_hot_of_info_index():
  """Kuhn info index encoder is one hot of info index."""
  env = kuhn.KuhnEnv()
  enc = registry.get("kuhn_poker_info_index")
  info = env.info_state(env.deal(1, 0), jnp.int32(1))
  x = np.asarray(enc(info))
  assert x.dtype == np.float32 and x.sum() == 1.0
  assert int(np.argmax(x)) == int(kuhn.info_index(info)) == 0 * 4 + 0


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
