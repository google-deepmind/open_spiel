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

"""Core fixtures and policies keep their declared dtypes under x64.

Carries, logits and params never promote to f64.
"""

import sys

import jax
import jax.numpy as jnp
import pytest

from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import fixtures
from open_spiel.python.scale.testing import steps as steps_lib

KUHN_INDEX = f"{kuhn.__name__}:info_index"


def _float_dtypes(tree):
  """The floating dtypes in a tree.

  Args:
    tree: A pytree.

  Returns:
    The dtype names.
  """
  return {
      str(x.dtype)
      for x in jax.tree_util.tree_leaves(tree)
      if jnp.issubdtype(x.dtype, jnp.floating)
  }


def test_policies_keep_f32_under_x64():
  """The fixture policies keep f32 under x64."""
  assert jax.config.jax_enable_x64
  env = kuhn.KuhnEnv()
  s1, s2 = steps_lib.two_ply_steps(env, 8, seat=1)
  pols = [
      uniform.UniformPolicy("kuhn_poker", 2, 2),
      tabular.TabularPolicy(
          "kuhn_poker", kuhn.NUM_INFO_STATES, 2, 2, KUHN_INDEX
      ),
      fixtures.HistoryPolicy("kuhn_poker", 2, 2),
      fixtures.HistoryPolicy("kuhn_poker", 2, 2, cross_hand=True),
  ]
  for pol in pols:
    params = pol.init_params(jax.random.key(0))
    assert _float_dtypes(params) <= {"float32"}, (
        pol.spec.arch_name,
        _float_dtypes(params),
    )
    c = pol.init_carry(params, 8, s1.seat, jax.random.key(1))
    for st in (s1, s2, s1):
      c = pol.observe(params, c, st)
      assert _float_dtypes(c) <= {"float32"}, (
          pol.spec.arch_name,
          _float_dtypes(c),
      )
      out = pol.evaluate(params, c, st, jax.random.key(2))
      assert out.logits.dtype == jnp.float32 and out.value.dtype == jnp.float32


def test_two_ply_steps_dtypes():
  """two_ply_steps keeps its declared dtypes."""
  env = kuhn.KuhnEnv()
  for st in steps_lib.two_ply_steps(env, 4, seat=0):
    assert st.seat.dtype == jnp.int32 and st.last_action.dtype == jnp.int32
    assert st.to_move.dtype == jnp.int32 and st.legal_mask.dtype == jnp.bool_


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
