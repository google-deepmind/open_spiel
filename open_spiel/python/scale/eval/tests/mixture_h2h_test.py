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

"""Tests of Evaluator.mixture_head_to_head.

Evaluator.mixture_head_to_head: population a under meta-strategy wa
against population b under wb, the per-game meta-mixture that is
realization-equivalent to the mixed strategy.

Known answers:
  * one-hot weights reduce to head_to_head of that member pair: the payoffs
    are bit for bit head_to_head's under the play key
    (jax.random.split(key)[0]; the member draws use split(key)[1]);
  * on kuhn with tabular members the mean payoff estimates the exact
    w_a^T M w_b (M[i, j] = league.mixture.symmetric_payoff of member tables,
    f64) within 4 SE over 4,096 deal pairs;
  * the member drawn for deal i is the same on both seat-swapped rows, the
    draw frequencies follow the weights, and a zero-weight member is never
    drawn;
  * malformed weights and mixed rules raise.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.eval import exploitability as expl
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import mixture
from open_spiel.python.scale.testing import populations


def _pops():
  """The Kuhn populations.

  Returns:
    The populations.
  """
  return populations.kuhn_populations()


def test_one_hot_weights_equal_head_to_head():
  """One-hot meta-strategies give the head-to-head of the chosen members."""
  env = kuhn.KuhnEnv()
  ev = match.Evaluator(env)
  a, b = _pops()
  key = jax.random.key(3)
  mm = ev.mixture_head_to_head(
      a, [0.0, 1.0, 0.0], b, [1.0, 0.0], 64, key, a_name="A", b_name="B"
  )
  ref = ev.head_to_head(a[1], b[0], 64, jax.random.split(key)[0])
  np.testing.assert_array_equal(mm.result.a_payoff, ref.a_payoff)
  np.testing.assert_array_equal(mm.result.b_payoff, ref.b_payoff)
  assert (mm.a_members == 1).all() and (mm.b_members == 0).all()
  assert mm.result.a == "A" and mm.result.b == "B" and mm.result.n_deals == 64


def test_mean_payoff_is_the_exact_mixture_payoff():
  """The mean payoff estimates the exact mixture payoff."""
  env = kuhn.KuhnEnv()
  tree = expl.GameTree(env)
  a, b = _pops()
  wa, wb = np.array([0.5, 0.3, 0.2]), np.array([0.6, 0.4])
  ta = [tree.policy_table(e.policy, e.params, types.SAMPLE) for e in a]
  tb = [tree.policy_table(e.policy, e.params, types.SAMPLE) for e in b]
  m = np.array([[mixture.symmetric_payoff(tree, x, y) for y in tb] for x in ta])
  exact = float(wa @ m @ wb)
  mm = match.Evaluator(env).mixture_head_to_head(
      a, wa, b, wb, 4096, jax.random.key(7)
  )
  est = mm.result.mean
  assert abs(est.mean - exact) < 4.0 * est.se, (est.mean, est.se, exact)
  # The draws follow the weights (4,096 deals: 4 binomial SDs).
  for k, w in enumerate(wa):
    f = float(np.mean(mm.a_members == k))
    assert abs(f - w) < 4.0 * np.sqrt(w * (1 - w) / 4096), (k, f, w)


def test_zero_weight_member_is_never_drawn_and_deals_pair():
  """A zero-weight member is never drawn, and deals stay paired."""
  env = kuhn.KuhnEnv()
  a, b = _pops()
  mm = match.Evaluator(env).mixture_head_to_head(
      a, [0.5, 0.0, 0.5], b, [0.0, 1.0], 512, jax.random.key(1)
  )
  assert not (mm.a_members == 1).any() and (mm.b_members == 1).all()
  assert mm.a_members.shape == (512,) and mm.a_members.dtype == np.int32
  assert set(np.unique(mm.a_members)) == {0, 2}


def test_validation():
  """Bad populations and weights are refused."""
  env = kuhn.KuhnEnv()
  ev = match.Evaluator(env)
  a, b = _pops()
  k = jax.random.key(0)
  with pytest.raises(ValueError, match="weights"):
    ev.mixture_head_to_head(a, [0.5, 0.5, 0.5], b, [0.5, 0.5], 8, k)
  with pytest.raises(ValueError, match="weights"):
    ev.mixture_head_to_head(a, [0.5, 0.5], b, [0.5, 0.5], 8, k)
  with pytest.raises(ValueError, match="weights"):
    ev.mixture_head_to_head(a, [1.5, -0.5, 0.0], b, [0.5, 0.5], 8, k)
  mixed = [
      a[0],
      populations.entrant("x", np.zeros(12), types.ARGMAX, policy=a[0].policy),
  ]
  with pytest.raises(ValueError, match="rule"):
    ev.mixture_head_to_head(mixed, [0.5, 0.5], b, [0.5, 0.5], 8, k)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
