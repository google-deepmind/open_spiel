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

"""BCConfig.policy_mask and the reserved aux namespace.

Known answer: a uniform tabular student on the hand-built two-ply batch of
bc_fixtures (one acting ply per row, target [0.25, 0.75]): with every row
decided the cross-entropy is log 2; rows marked undecided give exactly zero
policy gradient and leave the loss of the decided rows unchanged.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners.tests import bc_fixtures
from open_spiel.python.scale.policies import operator_vocab
from open_spiel.python.scale.sources.tests import helpers


def _learner(pol, mask="valid&search_decided"):
  """A target-distillation BC learner with policy mask `mask`.

  Args:
    pol: The policy.
    mask: The policy mask.

  Returns:
    The learner.
  """
  return bc.BCLearner(
      pol,
      bc_fixtures.cfg(
          loss=losses.DistillLossConfig(policy="ce_target", value="none"),
          policy_mask=mask,
      ),
  )


def test_default_mask_is_valid_and_configs_refuse_unknown():
  """Default mask is valid and configs refuse unknown."""
  assert bc.BCConfig().policy_mask == "valid"
  with pytest.raises(ValueError, match="policy_mask"):
    bc.BCConfig(policy_mask="decided")


def test_missing_search_decided_raises_key_error():
  """Missing search decided raises key error."""
  env = kuhn.KuhnEnv()
  pol = helpers.kuhn_tabular(env)
  b = bc_fixtures.two_ply_batch(env, [0.25, 0.75])
  with pytest.raises(KeyError, match="absent never means decided"):
    _learner(pol).loss(pol.init_params(), b)


def test_undecided_rows_give_zero_policy_gradient():
  """Undecided rows give zero policy gradient."""
  env = kuhn.KuhnEnv()
  pol = helpers.kuhn_tabular(env)
  b = bc_fixtures.two_ply_batch(env, [0.25, 0.75])
  p = pol.init_params()
  all_dec = jnp.ones((4, 2), jnp.bool_)
  b_all = b._replace(aux={operator_vocab.AUX_SEARCH_DECIDED: all_dec})
  _, m = _learner(pol).loss(p, b_all)
  np.testing.assert_allclose(float(m["loss_policy"]), np.log(2.0), rtol=1e-6)
  # The "valid" default on the same batch is the same number (decided = valid).
  _, m0 = _learner(pol, "valid").loss(p, b)
  np.testing.assert_allclose(
      float(m0["loss_policy"]), float(m["loss_policy"]), rtol=0
  )
  # Rows 2, 3 undecided: their targets do not matter (zero gradient from them).
  some = all_dec.at[2:].set(False)
  b_some = b._replace(aux={operator_vocab.AUX_SEARCH_DECIDED: some})
  junk = b_some._replace(
      target_policy=b_some.target_policy.at[2:, 0].set(
          jnp.asarray([0.9, 0.1], jnp.float32)
      )
  )
  g1 = jax.grad(lambda q: _learner(pol).loss(q, b_some)[0])(p)
  g2 = jax.grad(lambda q: _learner(pol).loss(q, junk)[0])(p)
  helpers.assert_tree_equal(g1, g2)
  # All undecided: no policy gradient at all.
  none = b._replace(
      aux={operator_vocab.AUX_SEARCH_DECIDED: jnp.zeros((4, 2), jnp.bool_)}
  )
  g0 = jax.grad(lambda q: _learner(pol).loss(q, none)[0])(p)
  assert all(not np.asarray(x).any() for x in jax.tree_util.tree_leaves(g0))


def test_reserved_aux_term_names_are_refused():
  """Reserved aux term names are refused."""
  bad = losses.DistillLossConfig(
      aux_terms=(losses.AuxTerm("search_value", "mse", 1.0),)
  )
  with pytest.raises(ValueError, match="reserved"):
    bc.BCConfig(loss=bad)
  bc.BCConfig(
      loss=losses.DistillLossConfig(
          aux_terms=(losses.AuxTerm("opp_card", "softmax_ce", 1.0),)
      )
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
