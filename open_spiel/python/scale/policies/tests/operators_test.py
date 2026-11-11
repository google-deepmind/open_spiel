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

"""Tests for policies.operators.

Operator facts read from the class, value_bias computed
from the config, the operator id, BehaviorMixture's plain and mixture
behaviours, and the refusal to publish a training-only wrapper.
"""

import hashlib
import json
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.belief import perfect_information
from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import search as gin_search
from open_spiel.python.scale.policies import operators
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.policies.tests import search_fixtures
from open_spiel.python.scale.testing.tests import infoset_invariance_fixtures


def test_facts_of_search_operators():
  """Facts of search operators."""
  _, base, sp = infoset_invariance_fixtures.leduc_search()
  assert operators.decider_of(sp) == "search"
  assert operators.prior_dependence_of(sp) == "puct_prior+value"
  # Leduc PIMC over a belief: strategy fusion below the root.
  assert operators.value_bias_of(sp) == "fused_upper_bound"
  # Connect four: a perfect-information sampler and no determinization ->
  # unbiased.
  env4 = connect_four.ConnectFourEnv()
  c4 = search.SearchPolicy(
      search_fixtures.c4_net(),
      env4,
      perfect_information.PerfectInformationSampler(env4),
      search_fixtures.cfg(num_simulations=4),
  )
  assert operators.value_bias_of(c4) == "unbiased"
  # Gin PIMC (determinized) through the routed composition.
  genv = gin.GinEnv(oklahoma_frac=1.0)
  gp = gin_search.make(
      infoset_invariance_fixtures.gin_net(genv),
      genv,
      gin_search.mcts_config(num_simulations=4, num_worlds=2, value_scale=50.0),
  )
  assert operators.value_bias_of(gp) == "fused_upper_bound"
  assert operators.decider_of(gp) == "search"
  with pytest.raises(ValueError, match="not a search operator"):
    operators.prior_dependence_of(base)
  with pytest.raises(ValueError, match="not a search operator"):
    operators.value_bias_of(base)


def test_describe_reports_a_non_operator_and_refuses_a_misdeclared_one():
  """Describe reports a non operator and refuses a misdeclared one."""
  plain = uniform.UniformPolicy("kuhn_poker", 2, 2)
  d = operators.describe(plain)
  assert (
      d["decider"] == "scripted"
      and d["prior_dependence"] is None
      and d["value_bias"] is None
  )

  class Misdeclared(uniform.UniformPolicy):
    """A policy that declares a prior_dependence outside the vocabulary."""

    prior_dependence = "not_in_the_vocabulary"

  with pytest.raises(ValueError):
    operators.describe(Misdeclared("kuhn_poker", 2, 2))


def test_operator_id_is_sha_of_canonical_json():
  """Operator id is sha of canonical json."""
  _, _, sp = infoset_invariance_fixtures.leduc_search()
  kw = dict(sp.spec.arch_kwargs)
  want = hashlib.sha256(
      json.dumps(kw, sort_keys=True, separators=(",", ":")).encode()
  ).hexdigest()
  assert operators.operator_id(sp.spec) == want == operators.operator_id(kw)
  assert operators.operator_id({"b": 1, "a": 2}) == operators.operator_id(
      {"a": 2, "b": 1}
  )


def test_parse_behavior():
  """Parse behavior."""
  assert operators.parse_behavior("search") == ("search", 1.0)
  assert operators.parse_behavior("plain") == ("plain", 0.0)
  assert operators.parse_behavior("mixture(0.25)") == ("mixture", 0.25)
  for bad in ("mixture(0)", "mixture(1.0)", "mix(0.5)", "greedy", "mixture()"):
    with pytest.raises(ValueError):
      operators.parse_behavior(bad)


def _step(env, n=8, seed=0):
  """SeatSteps of n initial states for their actors.

  Args:
    env: The env.
    n: The states.
    seed: The seed.

  Returns:
    The SeatStep.
  """
  s = jax.vmap(env.init)(jax.random.split(jax.random.key(seed), n))
  actor = jax.vmap(env.actor)(s)
  return plies.seat_step(
      env, s, actor, True, True, types.NO_ACTION, types.NO_ACTOR
  )


def test_plain_behavior_plays_the_base_and_labels_with_the_operator():
  """Plain behavior plays the base and labels with the operator."""
  env, base, sp = infoset_invariance_fixtures.leduc_search()
  bm = operators.BehaviorMixture(base, sp, "plain")
  bp = base.init_params(jax.random.key(1))
  params = bm.make_params(bp, {"base": bp, "sampler": {}})
  step = _step(env)
  carry = bm.observe(
      params, bm.init_carry(params, 8, step.seat, jax.random.key(2)), step
  )
  out = bm.evaluate(params, carry, step, jax.random.key(3))
  b_out = base.evaluate(bp, (), step, jax.random.key(4))
  np.testing.assert_array_equal(
      np.asarray(out.logits), np.asarray(b_out.logits)
  )
  assert not np.asarray(out.aux[operators.AUX_OPERATOR_ACTED]).any()
  # The targets are the operator's, under the keys BehaviorMixture derives.
  ko = jax.random.split(jax.random.key(3), 3)[2]
  sc = sp.observe(
      params["operator"],
      sp.init_carry(
          params["operator"],
          8,
          step.seat,
          jax.random.split(jax.random.key(2))[1],
      ),
      step,
  )
  s_out = sp.evaluate(params["operator"], sc, step, ko)
  np.testing.assert_array_equal(
      np.asarray(out.aux[types.AUX_TARGET_POLICY]),
      np.asarray(s_out.aux[types.AUX_TARGET_POLICY]),
  )
  assert bm.spec.arch_name == operators.ARCH_BEHAVIOR_MIXTURE
  assert operators.decider_of(bm) == "search"  # its operator's
  with pytest.raises(ValueError, match="search"):
    operators.BehaviorMixture(base, sp, "search")


def test_mixture_behavior_plays_the_operator_on_beta_of_rows():
  """Mixture behavior plays the operator on beta of rows."""
  env, base, sp = infoset_invariance_fixtures.leduc_search()
  bm = operators.BehaviorMixture(base, sp, "mixture(0.3)")
  bp = base.init_params(jax.random.key(1))
  params = bm.make_params(bp, {"base": bp, "sampler": {}})
  step = _step(env, n=400, seed=5)
  carry = bm.observe(
      params, bm.init_carry(params, 400, step.seat, jax.random.key(2)), step
  )
  out = bm.evaluate(params, carry, step, jax.random.key(3))
  coin = np.asarray(out.aux[operators.AUX_OPERATOR_ACTED])
  assert abs(coin.mean() - 0.3) < 0.08, coin.mean()
  base_lg = np.asarray(base.evaluate(bp, (), step, jax.random.key(0)).logits)
  np.testing.assert_array_equal(np.asarray(out.logits)[~coin], base_lg[~coin])


def test_behavior_mixture_is_never_published(tmp_path):
  """Behavior mixture is never published."""
  env, base, sp = infoset_invariance_fixtures.leduc_search()
  bm = operators.BehaviorMixture(base, sp, "plain")
  with pytest.raises(ValueError, match="never published"):
    operators.check_publishable(bm)
  operators.check_publishable(sp)
  bp = base.init_params(jax.random.key(1))
  with pytest.raises(artifact.ArtifactError, match="never published"):
    artifact.write_artifact(
        str(tmp_path / "a"),
        1,
        bm,
        bm.make_params(bp, {"base": bp, "sampler": {}}),
        rules_sha="r",
        config_hash="c",
        algo="t",
        env_name=env.name,
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
