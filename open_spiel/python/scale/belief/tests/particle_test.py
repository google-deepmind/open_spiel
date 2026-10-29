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

"""Tests for ParticleSampler.

On leduc its belief is within total variation 0.02 of the
exact posterior at 4096 particles, every sampled world agrees with the seat's
InfoState, and a history the opponent model cannot produce marks the row
degenerate after the restarts fail.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.belief import registry
from open_spiel.python.scale.belief.tests import leduc_fixtures
from open_spiel.python.scale.core import plies
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs.leduc import belief as leduc_belief
from open_spiel.python.scale.policies import particle_belief as particle

C, R = leduc_fixtures.CALL, leduc_fixtures.RAISE
SEAT = 0


def _hand(env, cards, round1, pub, round2):
  """The decision states of one hand and the actions taken between them.

  Args:
    env: The env.
    cards: The private cards dealt.
    round1: The first-round actions.
    pub: The public card.
    round2: The second-round actions.

  Returns:
    (states, actions).
  """
  states, actions = [], []
  s = env.root()
  for c in cards:
    s = env.apply_chance(s, jnp.int32(c))
  for a in round1:
    states.append(s)
    actions.append(a)
    s = env.apply(s, jnp.int32(a))
  s = env.apply_chance(s, jnp.int32(pub))
  for a in round2:
    states.append(s)
    actions.append(a)
    s = env.apply(s, jnp.int32(a))
  states.append(s)
  return states, actions


def _seat_steps(env, states, actions, seat):
  """The SeatSteps `seat` observes over the hand, each [1].

  Args:
    env: The env.
    states: The hand's decision states.
    actions: The actions between them.
    seat: The observing seat.

  Returns:
    The SeatSteps.
  """
  out = []
  for t, s in enumerate(states):
    s1 = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], s)
    if t == 0:
      la, lr = types.NO_ACTION, types.NO_ACTOR
    else:
      prev = states[t - 1]
      la = env.visible_action(prev, jnp.int32(actions[t - 1]), jnp.int32(seat))
      lr = env.actor(prev)
    out.append(
        plies.seat_step(env, s1, jnp.int32(seat), t == 0, t == 0, la, lr)
    )
  return out


def _run(sampler, params, steps, key):
  """Observes `steps` from the initial carry.

  Args:
    sampler: The sampler.
    params: The model's params.
    steps: The SeatSteps, in order.
    key: The init key.

  Returns:
    The final carry.
  """
  carry = sampler.init_carry(params, 1, jnp.full((1,), SEAT, jnp.int32), key)
  observe = jax.jit(sampler.observe)
  for step in steps:
    carry = observe(params, carry, step)
  return carry


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Leduc env.

  Returns:
    The env.
  """
  return leduc.LeducEnv()


def test_belief_converges_to_the_exact_posterior(env):
  """Belief converges to the exact posterior."""
  pol, pparams = leduc_fixtures.table_policy()
  sampler = particle.ParticleSampler(env, pol, num_particles=4096)
  states, actions = _hand(env, (2, 4), (C, R, C), 0, (R,))
  steps = _seat_steps(env, states, actions, SEAT)
  carry = _run(sampler, pparams, steps, jax.random.key(0))
  assert not bool(sampler.degenerate(carry)[0])
  info = steps[-1].info
  exact = np.asarray(
      leduc_belief.LeducBeliefSampler(env, pol).posterior(pparams, info)
  )[0]
  k = 4096
  worlds, logw = sampler.sample_worlds(
      pparams, carry, info, k, jax.random.key(1)
  )
  np.testing.assert_array_equal(np.asarray(logw), np.zeros((1, k)))
  flat = jax.tree_util.tree_map(lambda x: x[0], worlds)
  got_info = jax.vmap(lambda s: env.info_state(s, jnp.int32(SEAT)))(flat)
  for g, w in zip(
      jax.tree_util.tree_leaves(got_info), jax.tree_util.tree_leaves(info)
  ):
    np.testing.assert_array_equal(
        np.asarray(g), np.broadcast_to(np.asarray(w), np.shape(g))
    )
  # The belief itself: the particles' opponent cards under their weights.
  particles = jax.tree_util.tree_map(lambda x: x[0], carry.states)
  opp = np.asarray(
      jax.vmap(lambda s: env.labels(s, jnp.int32(SEAT))["opp_card"])(particles)
  )
  freq = np.bincount(
      opp, weights=np.asarray(sampler.belief(carry))[0], minlength=6
  )
  tv = 0.5 * np.abs(freq - exact).sum()
  print(f"particle belief vs exact posterior: total variation {tv:.4f}")
  assert tv < 0.02, (freq, exact)


def test_an_impossible_history_is_degenerate(env):
  """An impossible history is degenerate."""
  pol, pparams = leduc_fixtures.table_policy()
  logits = np.asarray(pparams["logits"]).copy()
  logits[:, R] = -np.inf  # this model never raises
  pparams = pol.params_from_logits(logits)
  sampler = particle.ParticleSampler(env, pol, num_particles=64, max_restarts=2)
  # but the opponent raised
  states, actions = _hand(env, (2, 4), (C, R, C), 0, ())
  carry = _run(
      sampler,
      pparams,
      _seat_steps(env, states, actions, SEAT),
      jax.random.key(0),
  )
  assert bool(sampler.degenerate(carry)[0])
  _, logw = sampler.sample_worlds(pparams, carry, None, 2, jax.random.key(0))
  assert np.all(np.isneginf(np.asarray(logw)))


def test_registered_and_rebuilt_from_kwargs(env):
  """Registered and rebuilt from kwargs."""
  pol, _ = leduc_fixtures.table_policy()
  s = registry.build("particle", env, opponent=pol, num_particles=8)
  assert registry.name_of(s) == "particle"
  assert s.params_source == "opponent" and s.stateful
  assert not s.is_perfect_information
  again = registry.build("particle", env, **s.kwargs)
  assert again.kwargs == s.kwargs


def test_invalid_sizes_are_refused(env):
  """Invalid sizes are refused."""
  pol, _ = leduc_fixtures.table_policy()
  with pytest.raises(ValueError, match="num_particles"):
    particle.ParticleSampler(env, pol, num_particles=0)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
