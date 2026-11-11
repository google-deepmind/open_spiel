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

"""The gin card-token encoder (encoders.gin_card_tokens) and its layout.

Its transcript codes equal the gin env's, it is registered at 799 floats,
and on real play it is finite, splits into the layout's blocks the torso
reads, and carries the observer's hand, the upcard and the event count.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.encoders import card_tokens_layout as ct
from open_spiel.python.scale.encoders import gin_card_tokens as enc
from open_spiel.python.scale.encoders import registry
from open_spiel.python.scale.envs import gin


def test_codes_and_sizes_equal_the_env():
  assert enc.EVENTS_MAX == gin.EVENTS_MAX == ct.EVENTS_MAX
  assert (enc.ACTOR_PAD, enc.ACTOR_SELF, enc.ACTOR_OPP) == (
      gin.ACTOR_PAD,
      gin.ACTOR_SELF,
      gin.ACTOR_OPP,
  )
  assert (enc.EVT_DRAW_STOCK, enc.EVT_KNOCK) == (
      gin.EVT_DRAW_STOCK,
      gin.EVT_KNOCK,
  )
  assert enc.DIM == ct.dim(enc.CARD_FIELDS, enc.GLOBAL_FIELDS) == 799
  e = registry.get(enc.NAME)
  assert e.shape == (799,) and e.game == "gin_rummy" and e.fn is enc.encode


def test_real_play_encodes_finite_and_splits():
  env = gin.GinEnv(oklahoma_frac=1.0)
  key = jax.random.key(3)
  s = env.init(key)
  for t in range(30):
    if bool(env.terminal(s)):
      break
    seat = int(env.actor(s))
    info = env.info_state(s, seat)
    x = registry.get(enc.NAME)(info)
    blind = enc.encode_opp_blind(info)
    assert bool(jnp.all(jnp.isfinite(x)))
    cards, glob, touch = ct.split(x[None], enc.CARD_FIELDS, enc.GLOBAL_FIELDS)
    hand = np.asarray(info.hand, np.float32)
    np.testing.assert_array_equal(cards[0, :, 0], hand)
    assert float(glob[0, -1]) == float(np.sum(np.asarray(info.ev_actor) != 0))
    assert int(touch[0, :, 1].max()) < ct.NUM_KINDS
    opp_col = np.asarray(blind[: 52 * 4]).reshape(52, 4)[:, 2]
    np.testing.assert_array_equal(opp_col, 0)
    legal = np.flatnonzero(np.asarray(env.legal_mask(s)))
    key, ka, kc = jax.random.split(key, 3)
    a = int(jax.random.choice(ka, legal))
    s = env.step(s, jnp.int32(a), kc)
  assert t > 3


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
