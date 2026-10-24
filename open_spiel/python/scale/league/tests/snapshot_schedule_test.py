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

"""controller.SnapshotSchedule, the schedule of ScheduledPromotion.

The snapshot schedule ScheduledPromotion runs on (controller.SnapshotSchedule):
round starts, harvested pool steps and the seated draw, with known answers.
The mix rule and the controller in a League are in snapshot_pool_test. Runs
without jax (jax_free_control_test): keys are core.rng keys.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.core import rng
from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import sampler

MIX = dict(self_weight=0.9, snapshot_every=100, harvest_every=500, max_pool=4)


def test_every_schedule_value_is_the_recipes():
  """No default: a recipe names its own schedule."""
  with pytest.raises(TypeError):
    # The missing argument is what the test checks.
    controller.SnapshotSchedule()  # pylint: disable=no-value-for-parameter
  mix = controller.SnapshotSchedule(**MIX)
  assert (
      mix.self_weight,
      mix.snapshot_every,
      mix.harvest_every,
      mix.max_pool,
      mix.pool_slots,
  ) == (0.9, 100, 500, 4, 1)


def test_snapshot_step_known_answers():
  """Snapshot step known answers."""
  mix = controller.SnapshotSchedule(**MIX)
  assert [mix.snapshot_step(u) for u in (0, 1, 99, 100, 101, 599, 2999)] == [
      0,
      0,
      0,
      100,
      100,
      500,
      2900,
  ]
  assert mix.round_index(0) == 0 and mix.round_index(250) == 2
  assert isinstance(mix.snapshot_step(np.int64(2**40)), int)


def test_pool_steps_known_answers():
  """Pool steps known answers."""
  mix = controller.SnapshotSchedule(**MIX)
  assert not mix.pool_steps(0)
  assert not mix.pool_steps(499)
  assert mix.pool_steps(500) == (500,)
  assert mix.pool_steps(1234) == (500, 1000)
  small = controller.SnapshotSchedule(
      **{**MIX, "snapshot_every": 10, "harvest_every": 20, "max_pool": 2}
  )
  assert small.pool_steps(79) == (40, 60)  # oldest dropped past max_pool
  assert small.pool_steps(80) == (60, 80)


def test_allocate_largest_remainder_known_answers():
  """The sampler's exact allocation, which splits a round's rows."""
  # Reaches private _largest_remainder of the code under test.
  lr = sampler._largest_remainder  # pylint: disable=protected-access
  np.testing.assert_array_equal(lr(np.array([0.95, 0.05]), 1024), [973, 51])
  np.testing.assert_array_equal(
      lr(np.array([0.95, 0.025, 0.025]), 1024), [973, 26, 25]
  )
  # one seat class
  np.testing.assert_array_equal(lr(np.array([0.95, 0.05]), 512), [486, 26])
  np.testing.assert_array_equal(lr(np.array([1.0]), 8), [8])
  # tie: low index
  np.testing.assert_array_equal(lr(np.array([0.5, 0.5]), 3), [2, 1])
  assert lr(np.array([0.95, 0.05]), 1024).dtype == np.int64
  with pytest.raises(ValueError, match="sum"):
    sampler.Mix(0.0, {"a": 0.5, "b": 0.6})


def test_active_pool_is_a_uniform_draw_over_the_pool():
  """The active pool is a uniform draw over the pool.

  pool_slots members of the pool are seated per round, uniformly: over
  4000 rounds each of 4 members is seated 1000 +- 5 sigma times.
  """
  mix = controller.SnapshotSchedule(
      **{**MIX, "snapshot_every": 1, "harvest_every": 1, "pool_slots": 1}
  )
  pool = (1, 2, 3, 4)
  draws = [mix.active_pool(pool, rng.key(i)) for i in range(4000)]
  assert all(len(d) == 1 and d[0] in pool for d in draws)
  counts = np.bincount([d[0] for d in draws], minlength=5)[1:]
  sigma = np.sqrt(4000 * 0.25 * 0.75)
  assert np.all(np.abs(counts - 1000) < 5 * sigma), counts
  two = controller.SnapshotSchedule(**{**MIX, "pool_slots": 2})
  d = two.active_pool(pool, rng.key(0))
  assert (
      len(d) == 2
      and len(set(d)) == 2
      and set(d) <= set(pool)
      and list(d) == sorted(d)
  )
  assert two.active_pool((7,), rng.key(0)) == (7,)  # fewer than pool_slots: all
  assert not two.active_pool((), rng.key(0))


def test_validation_fails_loud():
  """Validation fails loud."""
  with pytest.raises(ValueError):
    controller.SnapshotSchedule(**{**MIX, "self_weight": 0.0})
  with pytest.raises(ValueError):
    controller.SnapshotSchedule(**{**MIX, "self_weight": 1.5})
  with pytest.raises(ValueError):
    # harvest on a round start
    controller.SnapshotSchedule(
        **{**MIX, "snapshot_every": 100, "harvest_every": 150}
    )
  with pytest.raises(ValueError):
    controller.SnapshotSchedule(**{**MIX, "pool_slots": 0})
  with pytest.raises(ValueError):
    controller.SnapshotSchedule(**{**MIX, "max_pool": 0})


def _view(it, updates, members):
  """A LeagueView with a main learner after `updates`.

  Args:
    it: The iteration.
    updates: The learner's updates, or None for no learner.
    members: The members by name.

  Returns:
    The LeagueView.
  """
  learners = (
      ()
      if updates is None
      else (controller.LearnerInfo("main", "main", updates),)
  )
  return controller.LeagueView(it, tuple(members.values()), learners, None)


def _run_schedule(seed, rounds=30):
  """Runs ScheduledPromotion over `rounds` rounds.

  ScheduledPromotion over `rounds` rounds of 10 updates, its actions applied
  to a member table as the League would: the seated pool per round.

  Args:
    seed: the controller's seed.
    rounds: rounds to run.

  Returns:
    The (members, seated, mixes) per round.
  """
  ctl = controller.ScheduledPromotion(
      main="main",
      snapshot_every=10,
      harvest_every=20,
      max_pool=3,
      pool_slots=2,
      self_weight=0.9,
      seed=seed,
  )
  members, seated, mixes = {}, [], []
  for r in range(rounds):
    acts = ctl.step(_view(r, None if r == 0 else 10 * r, members))
    for a in acts:
      if isinstance(a, controller.Admit):
        members[a.member] = controller.MemberInfo(
            a.member, a.role, a.lifecycle, a.learner, r
        )
      elif isinstance(a, controller.SetLifecycle):
        members[a.member] = controller.MemberInfo(
            a.member, "snapshot", a.lifecycle, "main", r
        )
      elif isinstance(a, controller.SetMix):
        mixes.append(a.mix)
    # Reaches private _seated of the code under test.
    seated.append(ctl._seated)  # pylint: disable=protected-access
  return members, seated, mixes


def test_scheduled_promotion_runs_on_the_jax_free_rng():
  """ScheduledPromotion runs on the jax-free rng.

  The controller admits main@<10r> every round, evicts snapshots past
  max_pool, seats min(pool_slots, pool) harvested members, and is
  deterministic in the seed (core.rng draws).
  """
  members, seated, mixes = _run_schedule(seed=3)
  assert sorted(n for n, m in members.items() if m.lifecycle != "evicted") == [
      "main@240",
      "main@260",
      "main@280",
      "main@290",
  ]  # pool (240, 260, 280) + snapshot
  assert not seated[0] and seated[2] == ("main@20",)
  for r, s in enumerate(seated):
    sch = controller.SnapshotSchedule(
        self_weight=0.9,
        snapshot_every=10,
        harvest_every=20,
        max_pool=3,
        pool_slots=2,
    )
    pool = [f"main@{x}" for x in sch.pool_steps(10 * r)]
    assert len(s) == min(2, len(pool)) and set(s) <= set(pool)
    # by step
    assert [pool.index(n) for n in s] == sorted(pool.index(n) for n in s)
  assert mixes[-1].self_play == 0.0 and mixes[-1].weights[
      "main@290"
  ] == pytest.approx(0.9)
  again = _run_schedule(seed=3)[1]
  assert again == seated
  assert _run_schedule(seed=4)[1] != seated


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
