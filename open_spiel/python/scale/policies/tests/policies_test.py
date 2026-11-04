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

"""Tests for the uniform and tabular policies and the test fixtures.

They pass check_policy; known answers for the tabular lookup and the
registry.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import fixtures
from open_spiel.python.scale.testing import steps as steps_lib

KUHN_INDEX = f"{kuhn.__name__}:info_index"


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Kuhn env.

  Returns:
    The env.
  """
  return kuhn.KuhnEnv()


@pytest.fixture(name="steps", scope="module")
def _steps_fixture(env):
  """Two-ply Kuhn SeatSteps for seat 1.

  Args:
    env: The Kuhn env.

  Returns:
    The SeatSteps.
  """
  return steps_lib.two_ply_steps(env, batch=16, seat=1)


def _tabular(env):
  """A Kuhn tabular policy.

  Args:
    env: The Kuhn env.

  Returns:
    The policy.
  """
  return tabular.TabularPolicy(
      "kuhn_poker",
      kuhn.NUM_INFO_STATES,
      env.num_actions,
      env.num_players,
      KUHN_INDEX,
  )


def test_uniform(env, steps):
  """Uniform."""
  pol = uniform.UniformPolicy("kuhn_poker", env.num_actions, env.num_players)
  check_policy.check_policy(pol, pol.init_params(), pol.init_params(), *steps)


def test_tabular(env, steps):
  """Tabular."""
  pol = _tabular(env)
  pa = pol.params_from_logits(jnp.zeros((12, 2)))
  pb = pol.params_from_logits(jnp.ones((12, 2)) * jnp.arange(2.0))
  check_policy.check_policy(pol, pa, pb, *steps)


def test_tabular_known_answer(env, steps):
  """Tabular known answer."""
  pol = _tabular(env)
  table = jnp.arange(24.0).reshape(12, 2)
  out = pol.evaluate(pol.params_from_logits(table), (), steps[1], None)
  idx = kuhn.info_index(steps[1].info)
  assert jnp.array_equal(out.logits, table[idx])


def test_tabular_out_of_range_index_reads_nan_not_another_row(env, steps):
  """An index outside [0, N) is never clamped onto a real row."""
  pol = _tabular(env)
  info = steps[1].info._replace(card=steps[1].info.card + 3)  # index += 12
  out = pol.evaluate(
      pol.params_from_logits(jnp.ones((12, 2))),
      (),
      steps[1]._replace(info=info),
      None,
  )
  assert np.isnan(np.asarray(out.logits)).all()
  info = steps[1].info._replace(card=steps[1].info.card - 3)  # negative index
  out = pol.evaluate(
      pol.params_from_logits(jnp.ones((12, 2))),
      (),
      steps[1]._replace(info=info),
      None,
  )
  assert np.isnan(np.asarray(out.logits)).all()


def test_history_policy(env, steps):
  """History policy."""
  pol = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  check_policy.check_policy(
      pol, pol.init_params(), pol.make_params(b=jnp.ones(2)), *steps
  )


def test_cross_hand_history_policy(env, steps):
  """Cross hand history policy."""
  pol = fixtures.HistoryPolicy(
      "kuhn_poker", env.num_actions, env.num_players, cross_hand=True
  )
  pa, pb = pol.init_params(), pol.make_params(d=jnp.ones(2))
  check_policy.check_policy(pol, pa, pb, *steps, cross_hand_state=True)
  with pytest.raises(AssertionError, match="new_hand"):
    check_policy.check_policy(pol, pa, pb, *steps)


def test_callback_tabular_policy(env, steps):
  """Callback tabular policy."""
  del env  # Unused.
  pol = fixtures.CallbackTabularPolicy(
      "kuhn_poker", kuhn.NUM_INFO_STATES, 2, 2, KUHN_INDEX
  )
  pol.set_host_tables(np.arange(48.0).reshape(2, 12, 2))
  check_policy.check_policy(
      pol, {"member": jnp.int32(0)}, {"member": jnp.int32(1)}, *steps
  )
  out = pol.evaluate({"member": jnp.int32(1)}, (), steps[1], None)
  idx = np.asarray(kuhn.info_index(steps[1].info))
  np.testing.assert_array_equal(
      np.asarray(out.logits), np.arange(48.0).reshape(2, 12, 2)[1][idx]
  )
  # Under vmap (how the Arena calls it, batch of 1 per row) the callback sees
  # all rows at once and matches the direct call.
  per_row = jax.vmap(
      lambda st: pol.evaluate(
          {"member": jnp.int32(1)},
          (),
          jax.tree_util.tree_map(lambda x: x[None], st),
          None,
      ).logits[0]
  )(steps[1])
  np.testing.assert_array_equal(np.asarray(per_row), np.asarray(out.logits))


def test_registry_builds_every_architecture_from_its_spec(env):
  """Registry builds every architecture from its spec."""
  pols = [
      uniform.UniformPolicy("kuhn_poker", 2, 2),
      _tabular(env),
      fixtures.HistoryPolicy("kuhn_poker", 2, 2, cross_hand=True),
      fixtures.CallbackTabularPolicy("kuhn_poker", 12, 2, 2, KUHN_INDEX),
  ]
  for pol in pols:
    rebuilt = registry.build_from_spec(pol.spec)
    assert type(rebuilt) is type(pol) and rebuilt.spec == pol.spec
  assert {"uniform", "tabular"} <= set(registry.registered())


def test_registry_rejects_duplicates_and_unknown_names():
  """Registry rejects duplicates and unknown names."""
  with pytest.raises(ValueError, match="already registered"):
    registry.register("uniform", uniform.UniformPolicy)
  with pytest.raises(KeyError, match="unknown architecture"):
    registry.build("no_such_arch", {})


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
