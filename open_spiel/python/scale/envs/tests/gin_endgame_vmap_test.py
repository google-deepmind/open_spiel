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

"""The scripted gin endgame under the Arena's per-row vmap.

`actions` is a per-row function whose batch-level lax.conds (the planner only
when some row plans, its 1,024-subset form only when some planning row needs it,
simple_bot only when some row needs it) are what keep it cheap. Vmapped one row
per call they would turn into selects of every branch on every row;
`rowwise_actions` runs the vmapped calls as one batched call instead. Known
answers: the per-row vmapped policy equals the batched `actions` on every
endgame node of 64 games and on non-endgame rows, and the vmapped program keeps
its conds.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.tests import gin_games

tree_map = jax.tree_util.tree_map


@pytest.fixture(name="infos", scope="module")
def _infos_fixture():
  """InfoStates of endgame and other decision states.

  Returns:
    The batched InfoStates.
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  g = gin_games.games(64, 4)
  dec = gin_games.decision_mask(env, g)
  phase = np.asarray(g.states.game["phase"])
  end = gin_games.flat_states(g, dec & np.isin(phase, [4, 5, 6]))
  mid = gin_games.flat_states(g, dec & ~np.isin(phase, [4, 5, 6]))
  mid = tree_map(lambda x: x[:200], mid)
  st = tree_map(lambda a, b: jnp.concatenate([a, b]), end, mid)
  return jax.vmap(env.info_state)(st, jax.vmap(env.actor)(st))


def _one(info1):
  """The rowwise endgame action of one InfoState.

  Args:
    info1: One unbatched GinInfo.

  Returns:
    The action.
  """
  return gin_endgame.rowwise_actions()(tree_map(lambda x: x[None], info1))[0]


def test_vmapped_rows_equal_the_batched_actions(infos):
  """Vmapped rows equal the batched actions.

  Args:
    infos: The module `infos` fixture.
  """
  want = np.asarray(jax.jit(gin_endgame.actions)(infos))
  got = np.asarray(jax.jit(jax.vmap(_one))(infos))
  np.testing.assert_array_equal(got, want)
  assert np.asarray(gin_endgame.in_domain(infos)).sum() > 200


def test_policy_evaluate_vmapped_equals_batched(infos):
  """Policy evaluate vmapped equals batched.

  Args:
    infos: The module `infos` fixture.
  """
  pol = gin_endgame.GinScriptedEndgamePolicy()
  b = infos.phase.shape[0]
  step = type("S", (), {})()
  step.info = infos

  def ev(info):
    """The policy's logits on `info`.

    Args:
      info: A GinInfo.

    Returns:
      The logits.
    """
    s = step.__class__()
    s.info = info
    return pol.evaluate({}, (), s, None).logits

  batched = np.asarray(jax.jit(ev)(infos))
  per_row = np.asarray(
      jax.jit(jax.vmap(lambda i: ev(tree_map(lambda x: x[None], i))[0]))(infos)
  )
  np.testing.assert_array_equal(np.isnan(per_row), np.isnan(batched))
  np.testing.assert_array_equal(np.nan_to_num(per_row), np.nan_to_num(batched))
  assert batched.shape[0] == b


def test_vmapped_program_keeps_its_conds(infos):
  """Vmapped program keeps its conds.

  Args:
    infos: The module `infos` fixture.
  """
  jaxpr = jax.make_jaxpr(jax.vmap(_one))(infos)
  prims = set()

  def walk(j):
    for e in j.eqns:
      prims.add(e.primitive.name)
      for sub in _subjaxprs(e.params):
        walk(sub)

  walk(jaxpr.jaxpr)
  assert "cond" in prims, sorted(prims)


def _subjaxprs(params):
  for v in params.values():
    for x in v if isinstance(v, (tuple, list)) else (v,):
      if hasattr(x, "eqns"):
        yield x
      elif hasattr(getattr(x, "jaxpr", None), "eqns"):
        yield x.jaxpr


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
