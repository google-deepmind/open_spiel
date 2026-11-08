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

"""Search target builders and the one-writer rule, alone.

Search target builders (with_search_value_targets, with_q_softmax_targets)
and the one-writer rule, alone.

Known answers on the hand-built sequence of targets_test (perspective seat
0, gamma 1 here): rewards [0, 0, 0, 3, ...], valid plies 0, 2 (hand ends at
ply 3) and 5, 7 (unfinished): returns 3 at plies 0 and 2, complete
[1, 0, 1, 0, 0, 0, 0, 0].
"""

import sys

import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.policies import operator_vocab
from open_spiel.python.scale.sources import targets
from open_spiel.python.scale.sources.tests import helpers

R = [[0, 0, 0, 3, 0, 0, 0, 0]]
V = [[1, 0, 1, 0, 0, 1, 0, 1]]
D = [[0, 0, 0, 1, 0, 0, 0, 0]]
TO_MOVE = [[0, 1, 0, 1, -1, 0, 1, 0]]


def _batch(**kw):
  """The fake batch of the test's rewards.

  Args:
    **kw: More fake_batch fields.

  Returns:
    The batch.
  """
  return helpers.fake_batch(R, V, D, seat=0, to_move=TO_MOVE, **kw)


def test_the_label_constant_is_core_types_one():
  """The label constant is core types one."""
  assert (
      targets.TARGET_VALUE_VALID
      == types.TARGET_VALUE_VALID
      == "target_value_valid"
  )


def test_value_scale_none_leaves_returns_in_env_points():
  """Value scale none leaves returns in env points."""
  a = targets.with_mc_value_targets(_batch(), gamma=1.0)
  b = targets.with_mc_value_targets(_batch(), gamma=1.0, value_scale=None)
  helpers.assert_tree_equal(a, b)
  np.testing.assert_array_equal(
      np.asarray(a.target_value[..., 0]), [[3, 0, 3, 0, 0, 0, 0, 0]]
  )


def test_value_scale_divides_the_returns_known_answer():
  """Value scale divides the returns known answer."""
  out = targets.with_mc_value_targets(_batch(), gamma=1.0, value_scale=50.0)
  np.testing.assert_allclose(
      np.asarray(out.target_value[..., 0]),
      np.float32([[0.06, 0, 0.06, 0, 0, 0, 0, 0]]),
      rtol=0,
      atol=0,
  )
  assert out.target_value.dtype == jnp.float32
  with pytest.raises(ValueError, match="value_scale"):
    targets.with_mc_value_targets(_batch(), gamma=1.0, value_scale=0.0)


@pytest.mark.parametrize("builder", ["mc", "search"])
def test_every_writer_refuses_a_batch_with_value_targets(builder):
  """Every writer refuses a batch with value targets."""
  b = _batch(labels={types.TARGET_VALUE_VALID: jnp.asarray(V, bool)})
  b = b._replace(
      aux={operator_vocab.AUX_SEARCH_VALUE: jnp.zeros((1, 8), jnp.float32)}
  )
  with pytest.raises(ValueError, match="written once"):
    if builder == "mc":
      targets.with_mc_value_targets(b, gamma=1.0)
    else:
      targets.with_search_value_targets(b, lam=0.5, value_scale=1.0)


def test_search_value_targets_known_answer():
  """Search value targets known answer."""
  sv = jnp.asarray([[0.5, 0, -1.0, 0, 0, 0.25, 0, 0.75]], jnp.float32)
  b = _batch()._replace(aux={operator_vocab.AUX_SEARCH_VALUE: sv})
  out = targets.with_search_value_targets(b, lam=0.25, value_scale=2.0)
  # (1 - 0.25) * 3 / 2 + 0.25 * sv on complete plies 0 and 2; 0 elsewhere.
  want = np.float32(
      [[0.75 * 1.5 + 0.25 * 0.5, 0, 0.75 * 1.5 + 0.25 * -1.0, 0, 0, 0, 0, 0]]
  )
  np.testing.assert_allclose(
      np.asarray(out.target_value[..., 0]), want, rtol=0, atol=1e-7
  )
  np.testing.assert_array_equal(
      np.asarray(out.labels[types.TARGET_VALUE_VALID]),
      [[1, 0, 1, 0, 0, 0, 0, 0]],
  )
  # lam 1: the search value itself (on complete plies).
  out1 = targets.with_search_value_targets(b, lam=1.0, value_scale=2.0)
  np.testing.assert_array_equal(
      np.asarray(out1.target_value[..., 0]),
      np.float32([[0.5, 0, -1.0, 0, 0, 0, 0, 0]]),
  )


def test_search_value_targets_refusals():
  """Search value targets refusals."""
  b = _batch()
  with pytest.raises(KeyError, match="search_value"):
    targets.with_search_value_targets(b, lam=0.5, value_scale=1.0)
  b = b._replace(
      aux={operator_vocab.AUX_SEARCH_VALUE: jnp.zeros((1, 8), jnp.float32)}
  )
  with pytest.raises(ValueError, match="lam"):
    targets.with_search_value_targets(b, lam=0.0, value_scale=1.0)
  with pytest.raises(ValueError, match="value_scale"):
    targets.with_search_value_targets(b, lam=0.5, value_scale=None)
  bad = b._replace(
      aux={operator_vocab.AUX_SEARCH_VALUE: jnp.zeros((1, 8), jnp.float64)}
  )
  with pytest.raises(TypeError, match="float32"):
    targets.with_search_value_targets(bad, lam=0.5, value_scale=1.0)


def test_q_softmax_targets_known_answer():
  """Q softmax targets known answer."""
  q = np.zeros((1, 8, 2), np.float32)
  q[0, 0] = [0.0, np.log(3.0)]  # T = 1: softmax = [1/4, 3/4]
  q[0, 2] = [2.0, 0.0]  # T = 2 below: [e/(e+1), 1/(e+1)]
  b = _batch()._replace(aux={operator_vocab.AUX_SEARCH_Q: jnp.asarray(q)})
  out = targets.with_q_softmax_targets(b, temperature=1.0)
  tp = np.asarray(out.target_policy)
  assert out.target_policy.dtype == jnp.float32
  np.testing.assert_allclose(tp[0, 0], [0.25, 0.75], atol=1e-6)
  np.testing.assert_array_equal(tp[0, 1], 0.0)  # not a valid ply
  out2 = targets.with_q_softmax_targets(b, temperature=2.0)
  e = np.e
  np.testing.assert_allclose(
      np.asarray(out2.target_policy)[0, 2],
      [e / (e + 1), 1 / (e + 1)],
      atol=1e-6,
  )
  # Illegal actions get 0.
  legal = np.asarray(b.steps.legal_mask).copy()
  legal[0, 0, 0] = False
  b3 = b._replace(steps=b.steps._replace(legal_mask=jnp.asarray(legal)))
  np.testing.assert_array_equal(
      np.asarray(
          targets.with_q_softmax_targets(b3, temperature=1.0).target_policy
      )[0, 0],
      [0.0, 1.0],
  )
  with pytest.raises(ValueError):
    targets.with_q_softmax_targets(b, temperature=0.0)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
