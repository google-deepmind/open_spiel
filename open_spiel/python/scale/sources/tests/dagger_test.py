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

"""Contract + known-answer tests for the DAgger ArenaSource (sources.dagger).

Components first, alone: beta_schedule (linear, then flat), DaggerPolicy
(passes check_policy; beta 1 plays the teacher, 0 the student; target_policy
is the teacher's distribution), league.sampler.entries_table (exact quotas, the
one sampler). Then the source: known answers on kuhn with deterministic tabular
policies.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.league import sampler as league_sampler
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.sources import targets
from open_spiel.python.scale.sources.tests import helpers
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import fixtures
from open_spiel.python.scale.testing import steps as steps_lib

PASS, BET = 0, 1


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """A Kuhn env.

  Returns:
    The env.
  """
  return kuhn.KuhnEnv()


# --- beta (the one Schedule) --------------------------------------------------
def test_beta_schedule_known_values():
  """Beta schedule known values."""
  s = dagger.beta_schedule(start=1.0, end=0.2, decay_iters=2000)

  def v(it):
    """The schedule's value at iteration `it`.

    Args:
      it: The iteration.

    Returns:
      The value.
    """
    return float(s.value(np.int64(it)))

  assert v(0) == 1.0
  assert v(1000) == pytest.approx(0.6, abs=1e-12)
  assert v(2000) == pytest.approx(0.2, abs=1e-12)
  assert v(10**12) == pytest.approx(
      0.2, abs=1e-12
  )  # an int64 iteration past 2**31
  assert float(dagger.beta_schedule(0.3, 0.3).value(np.int64(5))) == 0.3
  with pytest.raises(AssertionError):
    dagger.beta_schedule(start=1.5, end=0.2)


# --- DaggerPolicy -------------------------------------------------------------
def _mix(env, teacher_action=BET, student_action=PASS):
  """A dagger mix of constant tabular student and teacher.

  Args:
    env: The Kuhn env.
    teacher_action: The teacher's action.
    student_action: The student's action.

  Returns:
    (mix, student params, teacher params).
  """
  pol_t, pol_s = helpers.kuhn_tabular(env), helpers.kuhn_tabular(env)
  mix = dagger.DaggerPolicy(pol_s, pol_t)
  tp = pol_t.params_from_logits(helpers.constant_logits(teacher_action))
  sp = pol_s.params_from_logits(helpers.constant_logits(student_action))
  return mix, sp, tp


def test_dagger_policy_passes_check_policy(env):
  """DAgger policy passes check policy."""
  mix, sp, tp = _mix(env)
  steps = steps_lib.two_ply_steps(env, batch=8, seat=1)
  check_policy.check_policy(
      mix, mix.make_params(sp, tp, 0.5), mix.make_params(tp, sp, 0.25), *steps
  )


def test_dagger_policy_stateful_parts_pass_check_policy(env):
  """DAgger policy stateful parts pass check policy."""
  st = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  mix = dagger.DaggerPolicy(st, helpers.kuhn_tabular(env))
  assert mix.spec.stateful
  tp = helpers.kuhn_tabular(env).params_from_logits(
      helpers.constant_logits(BET)
  )
  pa = mix.make_params(st.make_params(), tp, 0.5)
  pb = mix.make_params(st.make_params(c=jnp.ones((2,), jnp.float32)), tp, 0.5)
  check_policy.check_policy(
      mix, pa, pb, *steps_lib.two_ply_steps(env, batch=8, seat=0)
  )


def test_dagger_policy_beta_extremes_and_target(env):
  """DAgger policy beta extremes and target."""
  mix, sp, tp = _mix(env)
  step = steps_lib.two_ply_steps(env, batch=32, seat=0)[0]  # seat 0 opens
  key = jax.random.key(3)
  carry = mix.init_carry(mix.make_params(sp, tp, 1.0), 32, step.seat, key)
  for beta, want in ((1.0, BET), (0.0, PASS)):
    out = mix.evaluate(mix.make_params(sp, tp, beta), carry, step, key)
    assert out.logits.dtype == jnp.float32
    assert (np.argmax(np.asarray(out.logits), -1) == want).all()
    tgt = np.asarray(out.aux[types.AUX_TARGET_POLICY])
    assert out.aux[types.AUX_TARGET_POLICY].dtype == jnp.float32
    np.testing.assert_allclose(
        tgt[:, BET], 1.0, atol=1e-6
    )  # the teacher's, whatever beta
    acted = np.asarray(out.aux[dagger.AUX_TEACHER_ACTED])
    assert acted.all() if beta == 1.0 else (~acted).all()


def test_dagger_policy_teacher_target_is_masked_softmax(env):
  """DAgger policy teacher target is masked softmax."""
  pol = helpers.kuhn_tabular(env)
  mix = dagger.DaggerPolicy(pol, pol)
  logits = jnp.asarray(
      np.random.default_rng(0).normal(size=(12, 2)), jnp.float32
  )
  tp = pol.params_from_logits(logits)
  step = steps_lib.two_ply_steps(env, batch=16, seat=1)[1]  # seat 1 to move
  params = mix.make_params(pol.init_params(), tp, 0.5)
  carry = mix.init_carry(params, 16, step.seat, jax.random.key(1))
  out = mix.evaluate(params, carry, step, jax.random.key(0))
  want = jax.nn.softmax(logits[kuhn.info_index(step.info)], axis=-1)
  np.testing.assert_allclose(
      np.asarray(out.aux[types.AUX_TARGET_POLICY]),
      np.asarray(want),
      rtol=1e-6,
      atol=1e-7,
  )


def test_dagger_policy_rejects_mismatched_parts(env):
  """DAgger policy rejects mismatched parts."""
  pol = helpers.kuhn_tabular(env)
  other = fixtures.HistoryPolicy("kuhn_poker", 3, 2)
  with pytest.raises(ValueError):
    dagger.DaggerPolicy(pol, other)


# --- entries_table ------------------------------------------------------------
def test_pool_table_exact_quotas():
  # A distillation pool: the teacher at 0.5 and four others at 0.125 each.
  """Pool table exact quotas."""
  entries = [
      (1, 0, 0.5),
      (1, 1, 0.125),
      (1, 2, 0.125),
      (1, 3, 0.125),
      (1, 4, 0.125),
  ]
  t = league_sampler.entries_table(16, 2, entries, jax.random.key(0))
  slot, member, persp = (np.asarray(x) for x in t)
  assert slot.dtype == np.int32 and member.dtype == np.int32
  assert persp.dtype == np.int32
  np.testing.assert_array_equal(persp, np.arange(16) % 2)
  rows = np.arange(16)
  np.testing.assert_array_equal(slot[rows, persp], 0)
  np.testing.assert_array_equal(member[rows, persp], 0)
  opp = 1 - persp
  np.testing.assert_array_equal(slot[rows, opp], 1)
  counts = np.bincount(member[rows, opp], minlength=5)
  np.testing.assert_array_equal(counts, [8, 2, 2, 2, 2])
  # Each opponent meets the learner in both seats.
  for m in range(5):
    assert set(persp[member[rows, opp] == m].tolist()) == {0, 1}, m


def test_pool_table_rejects_bad_weights():
  """Pool table rejects bad weights."""
  with pytest.raises(ValueError):
    league_sampler.entries_table(
        8, 2, [(1, 0, 0.5), (1, 1, 0.4)], jax.random.key(0)
    )
  with pytest.raises(ValueError):
    league_sampler.entries_table(
        8, 2, [(1, 0, 1.0), (1, 1, 0.0)], jax.random.key(0)
    )


# --- DaggerSource -------------------------------------------------------------
def _source(
    env,
    beta,
    *,
    student_action=PASS,
    teacher_action=BET,
    opp_actions=(BET,),
    weights=None,
    batch=16,
    num_steps=8,
    lookahead=True,
    student=None,
    sp=None
):
  """A Kuhn dagger source against constant tabular opponents.

  Args:
    env: The env.
    beta: The teacher share.
    student_action: The student's action.
    teacher_action: The teacher's action.
    opp_actions: Each opponent member's action.
    weights: The opponent weights (default uniform).
    batch: The rows.
    num_steps: The plies.
    lookahead: Whether to look ahead for returns.
    student: The student policy (default tabular).
    sp: The student params (default constant).

  Returns:
    The DaggerSource.
  """
  pol = helpers.kuhn_tabular(env)
  opp = helpers.kuhn_tabular(env)
  k = len(opp_actions)
  weights = weights or [1.0 / k] * k
  opp_params = {
      "logits": jnp.stack([helpers.constant_logits(a) for a in opp_actions])
  }
  table = league_sampler.entries_table(
      batch, 2, [(1, m, w) for m, w in enumerate(weights)], jax.random.key(0)
  )
  student = student or pol
  sp = (
      sp
      if sp is not None
      else pol.params_from_logits(helpers.constant_logits(student_action))
  )
  return dagger.DaggerSource(
      env,
      student,
      pol,
      student_params=sp,
      teacher_params=pol.params_from_logits(
          helpers.constant_logits(teacher_action)
      ),
      table=table,
      num_steps=num_steps,
      beta=beta,
      opponents=[arena_lib.Slot(opp, k, types.SAMPLE)],
      opponent_params=[opp_params],
      gamma=1.0,
      return_lookahead=lookahead,
  )


def test_source_beta_one_plays_teacher_beta_zero_plays_student(env):
  """Source beta one plays teacher beta zero plays student."""
  for beta, want in ((1.0, BET), (0.0, PASS)):
    src = _source(env, schedules.constant(beta))
    b = src.next(jax.random.key(0))
    v = np.asarray(b.valid)
    assert v.any()
    assert (np.asarray(b.action)[v] == want).all(), beta
    # target_policy is the teacher's (always bet) at every learner decision, 0
    # elsewhere.
    tp = np.asarray(b.target_policy)
    np.testing.assert_allclose(tp[v][:, BET], 1.0, atol=1e-6)
    np.testing.assert_array_equal(tp[~v], 0.0)


def test_source_intermediate_beta_mixes_per_decision(env):
  """Source intermediate beta mixes per decision."""
  src = _source(env, schedules.constant(0.5), batch=64, num_steps=16)
  b = src.next(jax.random.key(1))
  v = np.asarray(b.valid)
  acted = np.asarray(b.aux[dagger.AUX_TEACHER_ACTED])[v]
  np.testing.assert_array_equal(
      np.asarray(b.action)[v] == BET, acted
  )  # coin <-> action
  frac = acted.mean()
  n = acted.size
  assert abs(frac - 0.5) < 4 * np.sqrt(0.25 / n), (frac, n)


def test_source_beta_follows_schedule_per_iteration(env):
  """Source beta follows schedule per iteration."""
  src = _source(
      env,
      dagger.beta_schedule(start=1.0, end=0.0, decay_iters=2),
      lookahead=False,
  )
  b0 = src.next(jax.random.key(0))  # beta(0) = 1
  src.next(jax.random.key(1))  # beta(1) = .5
  b2 = src.next(jax.random.key(2))  # beta(2) = 0
  assert src.iteration == 3 and isinstance(src.iteration, np.int64)
  assert (np.asarray(b0.action)[np.asarray(b0.valid)] == BET).all()
  assert (np.asarray(b2.action)[np.asarray(b2.valid)] == PASS).all()


def test_source_opponent_pool_rows_play_their_member(env):
  # member 0 always passes, member 1 always bets; rows alternate by quota.
  """Source opponent pool rows play their member."""
  src = _source(
      env,
      schedules.constant(1.0),
      opp_actions=(PASS, BET),
      weights=[0.5, 0.5],
      batch=8,
  )
  b = src.next(jax.random.key(0))
  member = np.asarray(src.table.seat_member)
  persp = np.asarray(src.table.perspective_seat)
  opp_member = member[np.arange(8), 1 - persp]
  to_move = np.asarray(b.steps.to_move)
  opp_ply = (to_move >= 0) & ~np.asarray(b.valid)
  act = np.asarray(b.action)
  for r in range(8):
    assert opp_ply[r].any()
    assert (
        act[r][opp_ply[r]] == (BET if opp_member[r] == 1 else PASS)
    ).all(), r


def test_source_value_targets_known_answer_always_bet(env):
  """Source value targets known answer always bet.

  Both sides always bet: every hand is a +-2 showdown, so every complete target
  is +2 iff the learner's card beats the opponent's (labels).

  Args:
    env: the kuhn env (fixture).
  """
  # T=7 with 3-ply hands: the hand opened at ply 6 ends in the next segment.
  src = _source(env, schedules.constant(1.0), batch=16, num_steps=7)
  b = src.next(jax.random.key(4))
  v = np.asarray(b.valid)
  ok = np.asarray(b.labels[targets.TARGET_VALUE_VALID])
  assert ok.dtype == np.bool_ and (ok <= v).all()
  assert v[:, 6].any()
  assert ok[
      v
  ].all()  # with the lookahead segment every decision's hand finishes
  seat = np.asarray(b.steps.seat)
  tv = np.take_along_axis(np.asarray(b.target_value), seat[..., None], -1)[
      ..., 0
  ]
  own = np.asarray(b.steps.info.card)
  opp = np.asarray(b.labels["opp_card"])
  want = np.where(own > opp, 2.0, -2.0)
  np.testing.assert_array_equal(tv[ok], want[ok])


def test_source_without_lookahead_masks_unfinished_hands(env):
  """Source without lookahead masks unfinished hands."""
  src = _source(
      env, schedules.constant(1.0), batch=16, num_steps=4, lookahead=False
  )
  b = src.next(jax.random.key(4))
  v = np.asarray(b.valid)
  ok = np.asarray(b.labels[targets.TARGET_VALUE_VALID])
  assert (ok <= v).all() and not ok[v].all()


def test_source_init_carry_is_the_students(env):
  """Source init carry is the students."""
  st = fixtures.HistoryPolicy("kuhn_poker", env.num_actions, env.num_players)
  src = _source(env, schedules.constant(0.5), student=st, sp=st.make_params())
  b1 = src.next(jax.random.key(0))
  b2 = src.next(jax.random.key(1))
  want = st.init_carry(
      st.make_params(), 16, jnp.zeros((16,), jnp.int32), jax.random.key(0)
  )
  for b in (b1, b2):
    assert jax.tree.structure(b.init_carry) == jax.tree.structure(want)
    for x, y in zip(jax.tree.leaves(b.init_carry), jax.tree.leaves(want)):
      assert x.shape == y.shape and x.dtype == y.dtype


def test_source_deterministic_and_counters_int64(env):
  """Source deterministic and counters int64."""
  a = _source(env, schedules.constant(0.5))
  c = _source(env, schedules.constant(0.5))
  for i in range(2):
    helpers.assert_tree_equal(
        a.next(jax.random.key(i)), c.next(jax.random.key(i))
    )
  assert isinstance(a.frames, np.int64) and isinstance(a.iteration, np.int64)
  assert a.frames == 3 * 16 * 8  # lookahead: 3 segments rolled for 2 batches


def test_set_student_params_takes_effect(env):
  """Set student params takes effect."""
  pol = helpers.kuhn_tabular(env)
  src = _source(
      env, schedules.constant(0.0), student_action=PASS, lookahead=False
  )
  b = src.next(jax.random.key(0))
  assert (np.asarray(b.action)[np.asarray(b.valid)] == PASS).all()
  src.set_student_params(pol.params_from_logits(helpers.constant_logits(BET)))
  b = src.next(jax.random.key(1))
  assert (np.asarray(b.action)[np.asarray(b.valid)] == BET).all()


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
