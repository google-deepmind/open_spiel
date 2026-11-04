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

"""Tests for LeducBeliefSampler.

Known-answer posteriors over the opponent's card given the public history and
an opponent model pi:

  P(c | history) ∝ 1[c not own card, c not public card]
                   * prod_t pi(a_t | I_opp_t(c))

over the opponent's decisions t (the deal prior is uniform over the cards the
seat cannot rule out, and the public card is independent of strategies).
Every expected value below is worked out by hand from the tables here.
"""

import fractions
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.belief.tests import leduc_fixtures
from open_spiel.python.scale.belief.tests import named_fns
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs.leduc import belief as leduc_belief
from open_spiel.python.scale.policies import search
from open_spiel.python.scale.policies import tabular
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import fixtures

Fr = fractions.Fraction

CALL, RAISE = leduc_fixtures.CALL, leduc_fixtures.RAISE
J0, J1, Q0, Q1, K0, K1 = range(6)


def _info(env, s, seat):
  """The InfoState of `seat` at `s`, with a batch axis of 1.

  Args:
    env: The env.
    s: The state.
    seat: The seat.

  Returns:
    The InfoState.
  """
  info = env.info_state(s, jnp.int32(seat))
  return jax.tree_util.tree_map(lambda x: x[None], info)


def _normalise(weights):
  """Normalises card weights into a 6-card distribution.

  Args:
    weights: {card: Fraction weight}.

  Returns:
    f64[6], the probabilities.
  """
  total = sum(weights.values())
  return np.array([float(weights.get(c, Fr(0)) / total) for c in range(6)])


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Leduc env.

  Returns:
    The env.
  """
  return leduc.LeducEnv()


def test_uniform_model_round1_is_card_removal(env):
  """Uniform model round1 is card removal."""
  pol = uniform.UniformPolicy("leduc_poker", 3, 2)
  sampler = leduc_belief.LeducBeliefSampler(env, pol)
  s = leduc_fixtures.state(env, (J0, K1), ())
  post = sampler.posterior(sampler.init_params(), _info(env, s, 0))
  assert post.dtype == jnp.float32 and post.shape == (1, 6)
  np.testing.assert_allclose(
      np.asarray(post[0]), [0, 0.2, 0.2, 0.2, 0.2, 0.2], atol=1e-6
  )


def test_uniform_model_round2_removes_public_card(env):
  """Uniform model round2 removes public card."""
  pol = uniform.UniformPolicy("leduc_poker", 3, 2)
  sampler = leduc_belief.LeducBeliefSampler(env, pol)
  # seat 1 holds Q1; round 1 check-check; public Q0. Uniform actions carry no
  # information: 1/4 on each of J0, J1, K0, K1.
  s = leduc_fixtures.state(env, (K0, Q1), (CALL, CALL), pub=Q0)
  post = sampler.posterior({}, _info(env, s, 1))
  np.testing.assert_allclose(
      np.asarray(post[0]), [0.25, 0.25, 0, 0, 0.25, 0.25], atol=1e-6
  )


def test_known_answer_one_opponent_raise(env):
  """Known answer one opponent raise."""
  pol, params = leduc_fixtures.table_policy()
  sampler = leduc_belief.LeducBeliefSampler(env, pol)
  # seat 0 holds J0 and checks; seat 1 raises (facing no bet: fold illegal, so
  # P(raise | rank) = r / (c + r) = .1, .5, .9 for J, Q, K).
  s = leduc_fixtures.state(env, (J0, K0), (CALL, RAISE))
  assert np.asarray(
      env.legal_mask(leduc_fixtures.state(env, (J0, K0), (CALL,)))
  ).tolist() == [False, True, True]
  want = _normalise(
      {J1: Fr(1, 10), Q0: Fr(1, 2), Q1: Fr(1, 2), K0: Fr(9, 10), K1: Fr(9, 10)}
  )
  post = sampler.posterior(params, _info(env, s, 0))
  np.testing.assert_allclose(np.asarray(post[0]), want, atol=1e-6)
  assert abs(want[J1] - 1 / 29) < 1e-12  # 0.1 / 2.9


def test_known_answer_two_opponent_decisions_then_public_card(env):
  """Known answer two opponent decisions then public card."""
  pol, params = leduc_fixtures.table_policy()
  sampler = leduc_belief.LeducBeliefSampler(env, pol)
  # Round 1: s0 check, s1 raise, s0 re-raise (second raise: the cap), s1 call.
  # s1's second decision faces a bet with raise capped: legal {fold, call},
  # P(call | rank) = c / (f + c) = .3/.9, .5/.8, .2/.3.
  hist = (CALL, RAISE, RAISE, CALL)
  facing = leduc_fixtures.state(env, (J0, K0), hist[:3])
  assert np.asarray(env.legal_mask(facing)).tolist() == [True, True, False]
  like = {
      J1: Fr(1, 10) * Fr(1, 3),
      Q0: Fr(1, 2) * Fr(5, 8),
      Q1: Fr(1, 2) * Fr(5, 8),
      K0: Fr(9, 10) * Fr(2, 3),
      K1: Fr(9, 10) * Fr(2, 3),
  }
  s = leduc_fixtures.state(env, (J0, K0), hist, pub=K1)
  # Round 2, seat 0 to act first: the public K1 removes K1.
  post = sampler.posterior(params, _info(env, s, 0))
  np.testing.assert_allclose(
      np.asarray(post[0]),
      _normalise({c: w for c, w in like.items() if c != K1}),
      atol=1e-6,
  )
  # Round 2: s0 checks, s1 raises (no bet facing in round 2: r / (c + r) again).
  s2 = leduc_fixtures.state(env, (J0, K0), hist, pub=K1, round2=(CALL, RAISE))
  like2 = {
      c: w * {J1: Fr(1, 10), Q0: Fr(1, 2), Q1: Fr(1, 2), K0: Fr(9, 10)}[c]
      for c, w in like.items()
      if c != K1
  }
  post2 = sampler.posterior(params, _info(env, s2, 0))
  np.testing.assert_allclose(np.asarray(post2[0]), _normalise(like2), atol=1e-6)


def test_posterior_does_not_depend_on_the_hidden_card(env):
  """Posterior does not depend on the hidden card."""
  pol, params = leduc_fixtures.table_policy()
  sampler = leduc_belief.LeducBeliefSampler(env, pol)
  posts = [
      np.asarray(
          sampler.posterior(
              params,
              _info(
                  env, leduc_fixtures.state(env, (J0, opp), (CALL, RAISE)), 0
              ),
          )[0]
      )
      for opp in (J1, Q0, K1)
  ]
  for p in posts[1:]:
    np.testing.assert_array_equal(p, posts[0])


def test_sample_worlds_consistent_and_distributed_as_posterior(env):
  """Sample worlds consistent and distributed as posterior."""
  pol, params = leduc_fixtures.table_policy()
  sampler = leduc_belief.LeducBeliefSampler(env, pol)
  s = leduc_fixtures.state(env, (J0, K0), (CALL, RAISE))
  info = _info(env, s, 0)
  k = 4000
  worlds, logw = sampler.sample_worlds(
      params,
      sampler.init_carry(
          params, 1, jnp.zeros((1,), jnp.int32), jax.random.key(0)
      ),
      info,
      k,
      jax.random.key(7),
  )
  np.testing.assert_array_equal(np.asarray(logw), np.zeros((1, k)))
  for leaf in jax.tree_util.tree_leaves(worlds):
    assert leaf.shape[:2] == (1, k)
  flat = jax.tree_util.tree_map(lambda x: x[0], worlds)
  # Every world agrees with the seat's InfoState exactly, at the same node.
  infos = jax.vmap(env.info_state, in_axes=(0, None))(flat, jnp.int32(0))
  for got, want in zip(
      jax.tree_util.tree_leaves(infos), jax.tree_util.tree_leaves(info)
  ):
    np.testing.assert_array_equal(
        np.asarray(got), np.broadcast_to(np.asarray(want), got.shape)
    )
  assert (np.asarray(jax.vmap(env.actor)(flat)) == 0).all()
  opp = np.asarray(flat.private_cards[:, 1])
  freq = np.bincount(opp, minlength=6) / k
  want = np.asarray(sampler.posterior(params, info)[0])
  np.testing.assert_allclose(freq, want, atol=0.03)
  assert freq[J0] == 0.0


def test_batched_rows_are_independent(env):
  """Batched rows are independent."""
  pol, params = leduc_fixtures.table_policy()
  sampler = leduc_belief.LeducBeliefSampler(env, pol)
  infos = [
      _info(env, leduc_fixtures.state(env, (J0, K0), (CALL, RAISE)), 0),
      _info(env, leduc_fixtures.state(env, (K0, Q1), (CALL, CALL), pub=Q0), 1),
  ]
  both = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), *infos)
  post = np.asarray(sampler.posterior(params, both))
  for i, inf in enumerate(infos):
    np.testing.assert_allclose(
        post[i], np.asarray(sampler.posterior(params, inf)[0]), atol=1e-7
    )


def test_stateful_opponent_model_is_refused(env):
  """Stateful opponent model is refused."""
  with pytest.raises(ValueError, match="stateless"):
    leduc_belief.LeducBeliefSampler(
        env, fixtures.HistoryPolicy("leduc_poker", 3, 2)
    )


def test_rebuild_from_kwargs(env):
  """Rebuild from kwargs."""
  pol, _ = leduc_fixtures.table_policy()
  sampler = leduc_belief.LeducBeliefSampler(env, pol)
  again = leduc_belief.LeducBeliefSampler(env, **sampler.kwargs)
  assert again.kwargs == sampler.kwargs
  assert again.opponent.spec == pol.spec


def test_card_removal_sampler_is_the_uniform_model_posterior(env):
  """Card removal sampler is the uniform model posterior.

  LeducCardRemovalSampler: the card-removal prior
  whatever the opponent did -- test_uniform_model_round1_is_card_removal's
  posterior, and an opponent raise (which moves the table model's posterior,
  test_known_answer_one_opponent_raise) moves nothing.

  Args:
    env: the env fixture.
  """
  sampler = leduc_belief.LeducCardRemovalSampler(env)
  assert not sampler.kwargs and not sampler.init_params()
  s = leduc_fixtures.state(env, (J0, K1), ())
  np.testing.assert_allclose(
      np.asarray(sampler.posterior({}, _info(env, s, 0))[0]),
      [0, 0.2, 0.2, 0.2, 0.2, 0.2],
      atol=1e-6,
  )
  s = leduc_fixtures.state(env, (J0, K1), (CALL, RAISE))
  np.testing.assert_allclose(
      np.asarray(sampler.posterior({}, _info(env, s, 0))[0]),
      [0, 0.2, 0.2, 0.2, 0.2, 0.2],
      atol=1e-6,
  )
  pol, params = leduc_fixtures.table_policy()
  steered = leduc_belief.LeducBeliefSampler(env, pol).posterior(
      params, _info(env, s, 0)
  )
  assert not np.allclose(
      np.asarray(steered[0]), [0, 0.2, 0.2, 0.2, 0.2, 0.2], atol=1e-3
  )
  again = leduc_belief.LeducCardRemovalSampler(env, **sampler.kwargs)
  assert again.opponent.spec == sampler.opponent.spec


def test_search_refuses_a_model_that_cannot_take_the_searched_params(env):
  """Search refuses a model that cannot take the searched params.

  params_source "own": SearchPolicy feeds the sampler the base's params,
  so the opponent model must have the base's param tree.

  Args:
    env: the env fixture.
  """
  pol, _ = leduc_fixtures.table_policy()
  cfg = search.MCTSConfig(num_simulations=2, num_worlds=2)
  search.SearchPolicy(pol, env, leduc_belief.LeducBeliefSampler(env, pol), cfg)
  other = tabular.TabularPolicy(
      game="leduc_poker",
      num_states=3,
      num_actions=3,
      num_players=2,
      index_fn=named_fns.LEDUC_INDEX_FN,
  )
  with pytest.raises(ValueError, match="params_source 'own'"):
    search.SearchPolicy(
        pol, env, leduc_belief.LeducBeliefSampler(env, other), cfg
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
