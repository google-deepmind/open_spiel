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

"""Tests of held-out teacher agreement.

Held-out teacher agreement (eval/heldout.py): known answers, exact state
count, determinism, independence from the student, stateful students.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs.connect_four import teacher as c4_teacher
from open_spiel.python.scale.eval import heldout
from open_spiel.python.scale.policies import uniform as uniform_lib
from open_spiel.python.scale.sources import dagger
from open_spiel.python.scale.sources.tests import helpers
from open_spiel.python.scale.testing import fixtures

F32 = jnp.float32


def test_agreement_known_answers():
  """The metrics of `agreement` have their known values."""
  legal = jnp.asarray(
      [[True, True, False], [True, True, True], [True, False, True]]
  )
  target = jnp.asarray(
      [[0.25, 0.75, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.5, 0.0, 0.5]], F32
  )
  mask = jnp.asarray([True, True, False])
  uniform = jnp.zeros((3, 3), F32)
  m = heldout.agreement(uniform, legal, target, mask)
  # Row 0: KL(t || uniform over 2) = log 2 - H(t); row 1: 0. Row 2 masked.
  h0 = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
  np.testing.assert_allclose(float(m["kl"]), (np.log(2) - h0) / 2, rtol=1e-6)
  np.testing.assert_allclose(
      float(m["teacher_entropy"]), (h0 + np.log(3)) / 2, rtol=1e-5
  )
  np.testing.assert_allclose(
      float(m["ce"]), (np.log(2) + np.log(3)) / 2, rtol=1e-6
  )
  # top1: uniform argmax is index 0; teacher argmax 1 (row 0), 0 (row 1, first
  # index).
  assert float(m["top1"]) == 0.5
  # A student equal to the teacher: top1 1, KL 0.
  same = heldout.agreement(
      jnp.log(jnp.where(legal, target, 1.0)), legal, target, mask
  )
  assert float(same["top1"]) == 1.0
  assert abs(float(same["kl"])) < 1e-6
  # Illegal logits never matter.
  big = uniform.at[0, 2].set(100.0)
  m_big = heldout.agreement(big, legal, target, mask)
  assert {k: float(v) for k, v in m_big.items()} == {
      k: float(v) for k, v in m.items()
  }


def _c4_source(rows=16):
  """A Connect Four dagger source with a prefer-centre teacher.

  Args:
    rows: The rows.

  Returns:
    (env, source).
  """
  env = connect_four.ConnectFourEnv()
  t = c4_teacher.PreferCenterPolicy()
  return env, dagger.DaggerSource(
      env,
      t,
      t,
      student_params={},
      teacher_params={},
      table=arena_lib.self_play_table(rows, 2, all_seats=True),
      num_steps=42,
      beta=schedules.constant(1.0),
      rule=types.ARGMAX,
      value_target="teacher",
      opening_plies=6,
      first_hand_only=True,
  )


def test_collect_exact_count_and_deterministic():
  """`collect` returns exactly num_states states, the same per key."""
  _, src = _c4_source()
  hs = heldout.collect(src, jax.random.key(3), 1000)
  assert hs.num_states == 1000 and int(jnp.sum(hs.mask)) == 1000
  assert bool(jnp.all(hs.batch.valid | ~hs.mask))
  _, src2 = _c4_source()
  hs2 = heldout.collect(src2, jax.random.key(3), 1000)
  helpers.assert_tree_equal(hs.batch, hs2.batch)
  np.testing.assert_array_equal(np.asarray(hs.mask), np.asarray(hs2.mask))
  # The mask keeps the FIRST 1000 valid states in row-major order.
  v = np.asarray(hs.batch.valid).reshape(-1)
  m = np.asarray(hs.mask).reshape(-1)
  last = np.nonzero(m)[0][-1]
  assert v[: last + 1].sum() == 1000 and not m[last + 1 :].any()


def test_score_teacher_copy_is_perfect_and_uniform_is_not():
  """A copy of the teacher agrees perfectly; a uniform student does not."""
  _, src = _c4_source()
  hs = heldout.collect(src, jax.random.key(0), 500)
  t = c4_teacher.PreferCenterPolicy()
  m = heldout.score(t, {}, hs)
  assert m["top1"] == 1.0 and m["num_states"] == 500
  assert abs(m["kl"]) < 1e-6 and abs(m["teacher_entropy"]) < 1e-6
  u = heldout.score(uniform_lib.UniformPolicy("connect_four", 7, 2), {}, hs)
  assert u["top1"] < 0.5
  # uniform student vs a one-hot teacher: KL = mean log(#legal)
  legal = np.asarray(hs.batch.steps.legal_mask)[np.asarray(hs.mask)]
  np.testing.assert_allclose(u["kl"], np.mean(np.log(legal.sum(-1))), rtol=1e-5)


def test_heldout_set_does_not_depend_on_the_student():
  """The held-out set does not depend on the student."""
  env = kuhn.KuhnEnv()
  teacher = helpers.kuhn_tabular(env)
  tp = teacher.params_from_logits(
      jnp.asarray(np.random.default_rng(0).normal(size=(12, 2)), F32)
  )
  sets = []
  for seed in (1, 2):
    sp = teacher.params_from_logits(
        jnp.asarray(np.random.default_rng(seed).normal(size=(12, 2)), F32)
    )
    src = dagger.DaggerSource(
        env,
        teacher,
        teacher,
        student_params=sp,
        teacher_params=tp,
        table=arena_lib.self_play_table(8, 2, all_seats=True),
        num_steps=6,
        beta=schedules.constant(1.0),
        value_target="teacher",
    )
    sets.append(heldout.collect(src, jax.random.key(9), 50))
  helpers.assert_tree_equal(sets[0].batch.steps, sets[1].batch.steps)
  helpers.assert_tree_equal(
      sets[0].batch.target_policy, sets[1].batch.target_policy
  )


def test_stateful_student_uses_stored_init_carry():
  """A stateful student starts from the stored init carry."""
  env = kuhn.KuhnEnv()
  teacher = helpers.kuhn_tabular(env)
  student = fixtures.HistoryPolicy(
      "kuhn_poker", env.num_actions, env.num_players
  )
  src = dagger.DaggerSource(
      env,
      student,
      teacher,
      student_params=student.make_params(),
      teacher_params=teacher.init_params(),
      table=arena_lib.self_play_table(8, 2, all_seats=True),
      num_steps=5,
      beta=schedules.constant(1.0),
      value_target="teacher",
  )
  src.next(jax.random.key(0))  # continue hands: later segments start mid-hand
  hs = heldout.collect(src, jax.random.key(1), 40)
  assert jax.tree.leaves(hs.batch.init_carry)
  m = heldout.score(student, student.make_params(), hs)
  assert 0.0 <= m["top1"] <= 1.0 and np.isfinite(m["kl"])


def test_collect_raises_when_the_source_runs_dry():
  """`collect` raises when the source yields no valid states."""

  class Empty:
    """A source whose batches hold no valid states."""

    def next(self, key):
      """A batch with no valid states.

      Args:
        key: unused.

      Returns:
        the batch.
      """
      del key
      return helpers.fake_batch(
          jnp.zeros((2, 3)), jnp.zeros((2, 3), bool), jnp.zeros((2, 3), bool), 0
      )

  with pytest.raises(ValueError, match="valid states"):
    heldout.collect(Empty(), jax.random.key(0), 5, max_batches=3)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
