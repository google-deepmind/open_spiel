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

"""Tests for OpeningPolicy.

In an Arena of connect_four games the first N plies of every
hand are drawn uniformly over the legal columns (frequencies), every later
ply is the base's (the prefer-center teacher's move), the counter restarts
at every hand, and the base's aux (a DAgger teacher's target) labels the
random plies too; determinism under the key.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs.connect_four import teacher as c4_teacher
from open_spiel.python.scale.policies import opening
from open_spiel.python.scale.policies import registry
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.sources.tests import helpers
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import steps as steps_lib


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Connect Four env.

  Returns:
    The env.
  """
  return connect_four.ConnectFourEnv()


def _roll(env, n_plies, batch=256, steps=20, seed=0):
  """Self-play of the opening policy over a prefer-centre base.

  Args:
    env: The env.
    n_plies: The opening plies.
    batch: The rows.
    steps: The plies.
    seed: The seed.

  Returns:
    (policy, batch, log).
  """
  pol = opening.OpeningPolicy(c4_teacher.PreferCenterPolicy(), n_plies)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.ARGMAX)])
  table = arena_lib.self_play_table(batch, 2)
  params = [jax.tree.map(lambda x: x[None], pol.init_params())]
  dk, pk = arena_lib.independent_keys(jax.random.key(seed), batch, 2)
  _, b, log = ar.collect(params, table, dk, pk, steps)
  return pol, b, log


def test_contract(env):
  # num_plies 1: after (step1, step2) the base plays, after step2 alone the
  # opening does, so check 6 (the carry matters) can see the counter.
  """Contract."""
  pol = opening.OpeningPolicy(uniform.UniformPolicy("connect_four", 7, 2), 1)
  s1, s2 = steps_lib.two_ply_steps(env, batch=4, seat=0)
  p = pol.init_params()
  check_policy.check_policy(pol, p, p, s1, s2)


def test_registry_rebuilds_it_and_validates(env):
  """Registry rebuilds it and validates."""
  del env  # Unused.
  pol = opening.OpeningPolicy(c4_teacher.PreferCenterPolicy(), 3)
  again = registry.build(pol.spec.arch_name, pol.spec.arch_kwargs)
  assert again.spec == pol.spec and again.num_plies == 3
  assert pol.spec.stateful
  for bad in (0, -1, 1.5, True):
    with pytest.raises(ValueError):
      opening.OpeningPolicy(c4_teacher.PreferCenterPolicy(), bad)


def _ply_in_hand(log_terminal):
  """i32[B, T] ply index within the hand (terminal plies included).

  Args:
    log_terminal: bool[B, T], the terminal plies.

  Returns:
    int64[B, T].
  """
  term = np.asarray(log_terminal)
  out = np.zeros(term.shape, np.int64)
  for r in range(term.shape[0]):
    k = 0
    for t in range(term.shape[1]):
      out[r, t] = k
      k = 0 if term[r, t] else k + 1
  return out


def test_first_n_plies_uniform_then_base(env):
  """First n plies uniform then base."""
  n = 3
  _, b, log = _roll(env, n, batch=512, steps=50)
  idx = _ply_in_hand(log.terminal_ply)
  act = np.asarray(log.action)
  live = ~np.asarray(log.terminal_ply)
  # After the opening every move is the teacher's (prefer center).
  # [B, T], perspective seat
  teacher = np.asarray(jax.vmap(c4_teacher.actions)(b.steps.info))
  persp_acts = np.asarray(b.valid)
  late = persp_acts & (idx[:, :] >= n)
  assert late.sum() > 1000
  np.testing.assert_array_equal(np.asarray(b.action)[late], teacher[late])
  # Ply 0 of a hand: 7 legal columns, uniform (chi-square-ish bound on counts).
  first = live & (idx == 0)
  counts = np.bincount(act[first], minlength=7)
  expect = first.sum() / 7
  assert first.sum() > 500
  assert np.all(np.abs(counts - expect) < 5 * np.sqrt(expect)), counts
  # Plies 1..n-1 are random too: column 3 (the teacher's favourite) is not
  # dominant.
  mid = live & (idx >= 1) & (idx < n)
  frac3 = np.mean(act[mid] == 3)
  assert 0.08 < frac3 < 0.22, frac3


def test_aux_marks_opening_and_base_aux_passes_through(env):
  """Aux marks opening and base aux passes through."""
  del env  # Unused.
  kenv = kuhn.KuhnEnv()
  teacher = helpers.kuhn_tabular(kenv)
  mix = dagger.DaggerPolicy(teacher, teacher)
  pol = opening.OpeningPolicy(mix, 1)
  tp = teacher.params_from_logits(helpers.constant_logits(1))
  params = pol.make_params(mix.make_params(tp, tp, 1.0))
  s1, s2 = steps_lib.two_ply_steps(kenv, batch=8, seat=0)
  c = pol.init_carry(params, 8, jnp.zeros((8,), jnp.int32), jax.random.key(0))
  c = pol.observe(params, c, s1)
  out = pol.evaluate(params, c, s1, jax.random.key(1))
  assert bool(jnp.all(out.aux[opening.AUX_OPENING]))
  # The base's target (the teacher plays action 1 with prob ~1) labels the
  # random ply.
  np.testing.assert_allclose(
      np.asarray(out.aux[types.AUX_TARGET_POLICY])[:, 1], 1.0, atol=1e-6
  )
  c = pol.observe(params, c, s2)
  out2 = pol.evaluate(params, c, s2, jax.random.key(1))
  assert not bool(jnp.any(out2.aux[opening.AUX_OPENING]))
  base_out = mix.evaluate(
      params["base"], c["base"], s2, jax.random.fold_in(jax.random.key(1), 1)
  )
  np.testing.assert_array_equal(
      np.asarray(out2.logits), np.asarray(base_out.logits)
  )


def test_counter_restarts_every_hand_and_is_int64(env):
  """Counter restarts every hand and is int64."""
  pol = opening.OpeningPolicy(uniform.UniformPolicy("connect_four", 7, 2), 2)
  s1, s2 = steps_lib.two_ply_steps(env, batch=3, seat=0)
  p = pol.init_params()
  c = pol.init_carry(p, 3, jnp.zeros((3,), jnp.int32), jax.random.key(0))
  assert c["ply"].dtype == jnp.int64
  for _ in range(5):
    c = pol.observe(p, c, s2)
  assert int(c["ply"][0]) == 5 and not bool(jnp.any(pol.in_opening(c)))
  # new hand
  c = pol.observe(p, c, s1._replace(new_match=jnp.zeros_like(s1.new_match)))
  assert int(c["ply"][0]) == 1 and bool(jnp.all(pol.in_opening(c)))


def test_deterministic_under_key(env):
  """Deterministic under key."""
  _, b1, _ = _roll(env, 4, batch=64, steps=12, seed=3)
  _, b2, _ = _roll(env, 4, batch=64, steps=12, seed=3)
  _, b3, _ = _roll(env, 4, batch=64, steps=12, seed=4)
  np.testing.assert_array_equal(np.asarray(b1.action), np.asarray(b2.action))
  assert np.any(np.asarray(b1.action) != np.asarray(b3.action))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
