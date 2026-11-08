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

"""Tests for EnumerationSampler.

The exact posterior over an information set's
histories. Known answers on kuhn (a card-dependent opponent) and leduc
(against LeducBeliefSampler's independent replay posterior), empirical draw
frequencies within 4 standard errors over 2^16 draws, and unmatched rows.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.belief import enumeration
from open_spiel.python.scale.belief import registry
from open_spiel.python.scale.belief.tests import leduc_fixtures
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs.leduc import belief as leduc_belief
from open_spiel.python.scale.eval import exploitability
from open_spiel.python.scale.policies import uniform

DRAWS = 1 << 16
#: P(bet | own card) of the kuhn opponent below at its first decision.
BET = (0.2, 0.5, 0.9)


class CardBettor(types.Policy):
  """Kuhn opponent model that bets with probability BET[card].

  At the first decision it bets with probability BET[card]; uniform
  elsewhere. Stateless.

  Attributes:
    spec: the PolicySpec.
  """

  def __init__(self):
    self.spec = types.PolicySpec(
        game="kuhn_poker",
        obs_format="kuhn_info",
        num_actions=2,
        num_players=2,
        stateful=False,
        capability=types.Capability.NATIVE_JAX,
        framework="jax",
        arch_name="testing.card_bettor",
        arch_kwargs={},
    )

  def init_params(self, key=None):
    """No params.

    Args:
      key: unused.

    Returns:
      An empty dict.
    """
    del key  # Unused.
    return {}

  def encode(self, info):
    """The InfoState's card.

    Args:
      info: the InfoState, leaves [B, ...].

    Returns:
      The card of each row.
    """
    return info.card

  def init_carry(self, params, batch, seat, key):
    """No carry.

    Args:
      params: unused.
      batch: unused.
      seat: unused.
      key: unused.

    Returns:
      An empty tuple.
    """
    del params, batch, seat, key  # Unused.
    return ()

  def observe(self, params, carry, step):
    """The carry unchanged.

    Args:
      params: unused.
      carry: the carry.
      step: unused.

    Returns:
      `carry`.
    """
    del params, step  # Unused.
    return carry

  def evaluate(self, params, carry, step, key):
    """Bet logits at the first decision, uniform elsewhere.

    Args:
      params: unused.
      carry: unused.
      step: the SeatStep, leaves [B, ...].
      key: unused.

    Returns:
      The PolicyOutput (zero values, no aux).
    """
    del params, carry, key  # Unused.
    info = step.info
    p_bet = jnp.asarray(BET, jnp.float32)[jnp.maximum(info.card, 0)]
    first = info.num_bets == 0
    p_bet = jnp.where(first, p_bet, jnp.float32(0.5))
    logits = jnp.log(jnp.stack([1.0 - p_bet, p_bet], axis=-1)).astype(
        jnp.float32
    )
    b = logits.shape[0]
    return types.PolicyOutput(
        logits=logits, value=jnp.zeros((b, 2), jnp.float32), aux={}
    )


def _batch(info):
  """Adds a leading batch axis of 1 to `info`.

  Args:
    info: An unbatched InfoState.

  Returns:
    The batched InfoState.
  """
  return jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], info)


def _opp_card_frequencies(env, worlds, seat, n_cards):
  """The frequency of each opponent card over the sampled worlds.

  Args:
    env: The env.
    worlds: The sampled worlds, leaves [1, K, ...].
    seat: The observing seat.
    n_cards: The deck size.

  Returns:
    f64[n_cards], the frequencies.
  """
  flat = jax.tree_util.tree_map(lambda x: x[0], worlds)
  opp = np.asarray(
      jax.vmap(lambda s: env.labels(s, jnp.int32(seat))["opp_card"])(flat)
  )
  return np.bincount(opp, minlength=n_cards) / opp.size


def _within_4se(freq, p, n):
  se = np.sqrt(np.maximum(p * (1 - p), 1e-12) / n)
  np.testing.assert_array_less(np.abs(freq - p), 4 * se + 1e-12)


@pytest.fixture(name="kuhn_sampler", scope="module")
def _kuhn_sampler_fixture():
  """Kuhn and its enumeration sampler.

  Returns:
    (env, sampler).
  """
  env = kuhn.KuhnEnv()
  return env, enumeration.EnumerationSampler(env)


def test_kuhn_posterior_is_bayes_rule_over_the_opponents_bet(kuhn_sampler):
  """Kuhn posterior is bayes rule over the opponents bet."""
  env, sampler = kuhn_sampler
  params = sampler.table_for(CardBettor(), {}, types.SAMPLE)
  s = env.deal(1, 0)  # the true opponent card is irrelevant to the posterior
  s = env.apply(s, jnp.int32(1))  # seat 0 bets
  info = _batch(env.info_state(s, jnp.int32(1)))
  carry = sampler.init_carry(
      params, 1, jnp.zeros((1,), jnp.int32) + 1, jax.random.key(0)
  )
  # seat 1 holds card 0
  want = np.array([0.0, BET[1], BET[2]]) / (BET[1] + BET[2])
  worlds, logw = jax.jit(sampler.sample_worlds, static_argnums=3)(
      params, carry, info, DRAWS, jax.random.key(1)
  )
  np.testing.assert_array_equal(np.asarray(logw), np.zeros((1, DRAWS)))
  _within_4se(_opp_card_frequencies(env, worlds, 1, 3), want, DRAWS)


def test_kuhn_uniform_opponent_is_card_removal(kuhn_sampler):
  """Kuhn uniform opponent is card removal."""
  env, sampler = kuhn_sampler
  params = sampler.init_params()
  s = env.deal(2, 0)
  info = _batch(env.info_state(s, jnp.int32(0)))
  carry = sampler.init_carry(
      params, 1, jnp.zeros((1,), jnp.int32), jax.random.key(0)
  )
  worlds, _ = sampler.sample_worlds(
      params, carry, info, DRAWS, jax.random.key(2)
  )
  _within_4se(
      _opp_card_frequencies(env, worlds, 0, 3), np.array([0.5, 0.5, 0.0]), DRAWS
  )


@pytest.fixture(name="leduc_case", scope="module")
def _leduc_case_fixture():
  """Leduc and its enumeration sampler.

  Returns:
    (env, sampler).
  """
  env = leduc.LeducEnv()
  return env, enumeration.EnumerationSampler(env)


@pytest.mark.parametrize("model", ["uniform", "table"])
def test_leduc_posterior_matches_the_replay_posterior(leduc_case, model):
  """Leduc posterior matches the replay posterior."""
  # Reaches private _states of the code under test.
  # pylint: disable=protected-access
  env, sampler = leduc_case
  if model == "uniform":
    pol, pparams = uniform.UniformPolicy("leduc_poker", 3, 2), {}
  else:
    pol, pparams = leduc_fixtures.table_policy()
  params = sampler.table_for(pol, pparams, types.SAMPLE)
  replay = leduc_belief.LeducBeliefSampler(env, pol)
  call, rais = leduc_fixtures.CALL, leduc_fixtures.RAISE
  s = leduc_fixtures.state(
      env, (0, 4), (call, rais, call), pub=2, round2=(rais,)
  )
  # the seat to move: its information set is a decision node
  seat = int(env.actor(s))
  assert seat == 1
  info = _batch(env.info_state(s, jnp.int32(seat)))
  want = np.asarray(replay.posterior(pparams, info))[0]
  carry = sampler.init_carry(
      params, 1, jnp.full((1,), seat, jnp.int32), jax.random.key(0)
  )
  worlds, _ = sampler.sample_worlds(
      params, carry, info, DRAWS, jax.random.key(3)
  )
  _within_4se(_opp_card_frequencies(env, worlds, seat, 6), want, DRAWS)
  # The exact posterior over members, mapped to the opponent's card.
  logp = np.asarray(sampler.log_posterior(params, carry, info))[0]
  nodes = np.asarray(sampler.members_of(info, carry))[0]
  states = jax.tree_util.tree_map(
      lambda x: x[np.maximum(nodes, 0)],
      # Test access to private members.
      sampler._states,
  )  # pylint: disable=protected-access
  opp = np.asarray(
      jax.vmap(lambda st: env.labels(st, jnp.int32(seat))["opp_card"])(states)
  )
  real = nodes >= 0  # members_of pads with -1
  p = np.exp(logp[real] - np.max(logp[real]))
  got = np.bincount(opp[real], weights=p, minlength=6) / p.sum()
  np.testing.assert_allclose(got, want, atol=1e-5)


def test_rows_that_are_not_an_infoset_of_the_seat_get_minus_inf(leduc_case):
  """Rows that are not an infoset of the seat get minus inf."""
  env, sampler = leduc_case
  params = sampler.init_params()
  s = leduc_fixtures.state(env, (0, 4), ())
  info = _batch(env.info_state(s, jnp.int32(1)))  # seat 1 is not to move here
  carry = sampler.init_carry(
      params, 1, jnp.ones((1,), jnp.int32), jax.random.key(0)
  )
  assert int(sampler.infoset_index(info, carry)[0]) == -1
  _, logw = sampler.sample_worlds(params, carry, info, 2, jax.random.key(0))
  assert np.all(np.isneginf(np.asarray(logw)))


def test_registered_and_rebuilt_from_kwargs(leduc_case):
  """Registered and rebuilt from kwargs."""
  env, sampler = leduc_case
  assert registry.name_of(sampler) == "enumeration"
  assert sampler.params_source == "opponent"
  assert not sampler.is_perfect_information
  assert registry.build("enumeration", env, **sampler.kwargs).num_infosets == (
      sampler.num_infosets
  )


def test_too_large_a_tree_raises():
  """Too large a tree raises."""
  with pytest.raises(exploitability.TooLargeError):
    enumeration.EnumerationSampler(leduc.LeducEnv(), max_nodes=100)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
