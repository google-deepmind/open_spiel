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

"""Under jax_enable_x64 the small envs keep their declared dtypes.

InfoState leaves i32, actor i32, legal bool, returns f32, and the engine state's
dtypes are unchanged by a step (scan carries).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import bargaining
from open_spiel.python.scale.envs import breakthrough
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import gomoku
from open_spiel.python.scale.envs import hex as hex_env
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs import liars_dice
from open_spiel.python.scale.envs import othello
from open_spiel.python.scale.envs import phantom_ttt
from open_spiel.python.scale.envs import phantom_ttt_ir
from open_spiel.python.scale.envs import tic_tac_toe

ENVS = [
    leduc.LeducEnv,
    tic_tac_toe.TicTacToeEnv,
    connect_four.ConnectFourEnv,
    phantom_ttt.PhantomTTTEnv,
    breakthrough.BreakthroughEnv,
    gomoku.GomokuEnv,
    hex_env.HexEnv,
    othello.OthelloEnv,
    liars_dice.LiarsDiceEnv,
    phantom_ttt_ir.PhantomTTTIREnv,
    bargaining.BargainingEnv,
]


def test_x64_is_on():
  """X64 is on."""
  assert jax.config.jax_enable_x64, "run with JAX_ENABLE_X64=1"


@pytest.mark.parametrize("ctor", ENVS, ids=lambda c: c.__name__)
def test_declared_dtypes_survive_x64(ctor):
  """Declared dtypes survive x64.

  Args:
    ctor: The parametrized case.
  """
  env = ctor()
  s = env.init(jax.random.key(0))
  a = jnp.argmax(env.legal_mask(s))  # an int64 action under x64
  s2 = env.step(s, a, jax.random.key(1))
  for x, y in zip(jax.tree_util.tree_leaves(s), jax.tree_util.tree_leaves(s2)):
    assert x.dtype == y.dtype and x.shape == y.shape
  for st in (s, s2):
    for seat in (0, 1):
      for leaf in jax.tree_util.tree_leaves(env.info_state(st, seat)):
        assert leaf.dtype == jnp.int32, leaf.dtype
    assert env.actor(st).dtype == jnp.int32
    assert env.legal_mask(st).dtype == jnp.bool_
    assert env.returns(st).dtype == jnp.float32
    assert env.visible_action(st, a, 0).dtype == jnp.int32
  assert np.asarray(env.legal_mask(s)).any()


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
