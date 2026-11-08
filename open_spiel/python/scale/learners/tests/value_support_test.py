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

"""The HL-Gauss support is configured once: on the value head.

The loss configs carry no hl_gauss_* fields; a learner takes the support from
its policy's value head (policy.value_support()) and hands it to the loss, so
the loss always trains on exactly the support the network decodes with.
"""

import dataclasses
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.learners import losses as loss_lib
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.learners.tests import pg_fixtures as fx


def _pg(pg_loss):
  """A PPO PGConfig with `pg_loss`.

  Args:
    pg_loss: The PGLossConfig.

  Returns:
    The PGConfig.
  """
  return pg.PGConfig(
      loss="ppo",
      num_updates=2,
      pg=pg_loss,
      kl_coef=schedules.Schedule("constant", 0.0),
  )


def test_loss_configs_have_no_support_fields():
  """Loss configs have no support fields."""
  for cls in (loss_lib.PGLossConfig, loss_lib.DistillLossConfig):
    names = {f.name for f in dataclasses.fields(cls)}
    assert not any(n.startswith("hl_gauss") for n in names), (
        cls.__name__,
        sorted(names),
    )


def test_network_policy_reports_its_support():
  """Network policy reports its support."""
  pol = fx.make_policy(network=fx.mlp(value="hl_gauss"))
  s = pol.value_support()
  assert (
      s.num_bins == 11
      and float(s.edges[0]) == -3.0
      and float(s.edges[-1]) == 3.0
  )
  assert fx.make_policy().value_support() is None


def test_learners_take_the_heads_support():
  """Learners take the heads support."""
  pol = fx.make_policy(network=fx.mlp(value="hl_gauss"))
  lrn = pg.PGLearner(pol, _pg(loss_lib.PGLossConfig(value_head="hl_gauss")))
  assert lrn.value_support is pol.value_support() or (
      lrn.value_support.num_bins == 11
      and np.array_equal(lrn.value_support.edges, pol.value_support().edges)
  )
  with pytest.raises(ValueError, match="scalar"):
    pg.PGLearner(
        fx.make_policy(), _pg(loss_lib.PGLossConfig(value_head="hl_gauss"))
    )
  b = bc.BCLearner(
      pol,
      bc.BCConfig(
          loss=loss_lib.DistillLossConfig(policy="kl_target", value="hl_gauss")
      ),
  )
  assert b.value_support.num_bins == 11
  with pytest.raises(ValueError, match="scalar"):
    bc.BCLearner(
        fx.make_policy(),
        bc.BCConfig(loss=loss_lib.DistillLossConfig(value="hl_gauss")),
    )
  # a non-HL-Gauss value loss needs no support
  assert (
      bc.BCLearner(
          pol,
          bc.BCConfig(
              loss=loss_lib.DistillLossConfig(policy="kl_target", value="none")
          ),
      ).value_support
      is None
  )


def test_loss_requires_a_support_for_hl_gauss():
  """Loss requires a support for hl gauss."""
  n, a, nb = 4, 3, 11
  sup = loss_lib.hl_gauss_support(-3.0, 3.0, nb)
  kw = dict(
      logits=jnp.zeros((n, a), jnp.float32),
      legal=jnp.ones((n, a), bool),
      mask=jnp.ones((n,), bool),
      target_policy=jnp.full((n, a), 1 / a, jnp.float32),
      value_logits=jnp.zeros((n, nb), jnp.float32),
      target_value=jnp.asarray([-3.0, 0.0, 1.0, 9.0], jnp.float32),
  )
  cfg = loss_lib.DistillLossConfig(policy="ce_target", value="hl_gauss")
  with pytest.raises(AssertionError, match="value_support"):
    loss_lib.distill_loss(cfg, **kw)
  _, m = loss_lib.distill_loss(cfg, value_support=sup, **kw)
  want = jnp.mean(
      loss_lib.hl_gauss_cross_entropy(
          kw["value_logits"], kw["target_value"], sup
      )
  )
  np.testing.assert_allclose(float(m["loss_value"]), float(want), rtol=1e-6)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
