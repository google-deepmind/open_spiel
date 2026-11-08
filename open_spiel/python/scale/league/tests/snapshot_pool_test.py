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

"""The snapshot-pool recipe (runner.snapshot_pool).

The snapshot-pool recipe (runner.snapshot_pool): the sampler rule
sampler.snapshot_pool under controller.ScheduledPromotion, in a League with
pool_layout "slots", bench and redraw "on_change". Known answers on synthetic
views first, then a small league on kuhn.
"""

import copy
import dataclasses
import gc
import glob
import pickle
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.ckpt import artifact
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.core import types
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league import sampler


# -- the sampler rule ----------------------------------------------------------
def test_snapshot_pool_known_answers():
  """Snapshot pool known answers."""
  # no pool: all snapshot
  assert sampler.snapshot_pool("s@0", [], 0.95) == sampler.Mix(
      0.0, {"s@0": 1.0}
  )
  m = sampler.snapshot_pool("s@2", ["p@1", "p@2"], 0.95)
  assert m.self_play == 0.0 and m.weights == pytest.approx(
      {"s@2": 0.95, "p@1": 0.025, "p@2": 0.025}
  )
  # the seated member IS the snapshot
  merged = sampler.snapshot_pool("s@2", ["s@2"], 0.95)
  assert merged.weights == pytest.approx({"s@2": 1.0})
  with pytest.raises(ValueError):
    sampler.snapshot_pool("s@0", ["a", "a"], 0.95)
  with pytest.raises(ValueError):
    sampler.snapshot_pool("s@0", ["a"], 0.0)


def test_snapshot_pool_table_counts_are_exact_per_seat():
  """Snapshot-pool table counts are exact per seat.

  The one sampler draws the rule's table: per learner-seat class the rows
  split by largest remainder (1024 rows, 2 seats: 486 / 26 each).
  """
  pools = tuple(
      sampler.Pool(policy=None, rule=types.ARGMAX, names=(n,), params=None)
      for n in ("snap@5", "pool@5")
  )
  mix = sampler.snapshot_pool("snap@5", ["pool@5"], 0.95)
  d = sampler.sample(
      sampler.LeagueSlots(num_players=2, batch=1024, pools=pools, mix=mix),
      jax.random.key(3),
  )
  slot = np.asarray(d.table.seat_slot)
  persp = np.asarray(d.table.perspective_seat)
  np.testing.assert_array_equal(slot[np.arange(1024), persp], 0)  # the learner
  opp = slot[np.arange(1024), 1 - persp]
  for s in range(2):
    np.testing.assert_array_equal(
        np.bincount(opp[persp == s], minlength=3)[1:], [486, 26]
    )
  assert d.counts == {"snap@5": 972, "pool@5": 52}


# -- the controller ------------------------------------------------------------
def _view(it, updates, members):
  """A LeagueView with a main learner after `updates`.

  Args:
    it: The iteration.
    updates: The learner's updates, or None for no learner.
    members: The members.

  Returns:
    The LeagueView.
  """
  learners = (
      ()
      if updates is None
      else (controller.LearnerInfo("main", "main", updates),)
  )
  return controller.LeagueView(
      iteration=it, members=tuple(members), learners=learners, payoff=None
  )


def _play(ctl, ticks):
  """Applies the controller's actions to a toy book-keeping of members.

  Args:
    ctl: The controller.
    ticks: (iteration, updates) per step.

  Returns:
    (the final members, each step's actions).
  """
  members, log = {}, []
  for it, u in ticks:
    acts = ctl.step(_view(it, u if it else None, members.values()))
    for a in acts:
      if isinstance(a, controller.Admit):
        members[a.member] = controller.MemberInfo(
            a.member, a.role, a.lifecycle, a.learner, it
        )
      elif isinstance(a, controller.SetLifecycle):
        members[a.member] = controller.MemberInfo(
            a.member, "snapshot", a.lifecycle, "main", it
        )
    log.append(acts)
  return members, log


def test_scheduled_promotion_known_answers():
  """ScheduledPromotion known answers.

  snapshot_every 2, harvest_every 4, max_pool 2, pool_slots 1, ticks at
  updates 0..9: promotions at 0, 2, 4, 6, 8; the pool is the harvested
  4 and 8 (and their predecessors evicted); one seated per round.
  """
  ctl = controller.ScheduledPromotion(
      snapshot_every=2,
      harvest_every=4,
      max_pool=2,
      pool_slots=1,
      self_weight=0.9,
      seed=0,
  )
  members, log = _play(ctl, [(i, i) for i in range(10)])
  assert isinstance(log[0][0], controller.SpawnLearner)
  assert log[0][1] == controller.Admit("main", "main@0", "snapshot", "fixed")
  acts_flat = []
  for acts in log:
    acts_flat.extend(acts)
  admitted = [a.member for a in acts_flat if isinstance(a, controller.Admit)]
  assert admitted == ["main@0", "main@2", "main@4", "main@6", "main@8"]
  evicted = [
      a.member for a in acts_flat if isinstance(a, controller.SetLifecycle)
  ]
  assert evicted == [
      "main@0",
      "main@2",
      "main@6",
  ]  # 4 and 8 are the pool (max_pool 2)
  live = sorted(n for n, m in members.items() if m.lifecycle != "evicted")
  assert live == ["main@4", "main@8"]
  mixes = [a.mix for a in acts_flat if isinstance(a, controller.SetMix)]
  assert len(mixes) == 10  # every tick restates the mix
  assert mixes[0] == sampler.Mix(0.0, {"main@0": 1.0})
  assert mixes[3] == sampler.Mix(0.0, {"main@2": 1.0})
  # seated main@4 == snapshot
  assert mixes[4].weights == pytest.approx({"main@4": 1.0})
  assert set(mixes[6].weights) == {"main@6", "main@4"}
  assert mixes[6].weights["main@6"] == pytest.approx(0.9)
  assert ctl.last_info == dict(snapshot_step=8, pool_size=2, pool_seated=1)
  assert ctl.schedule == controller.SnapshotSchedule(0.9, 2, 4, 2, 1)
  # a tick that lands inside a round it has not seen (a missed round start)
  # fails loud
  late = controller.ScheduledPromotion(
      snapshot_every=4, harvest_every=4, max_pool=2, self_weight=0.9
  )
  late.step(_view(0, None, ()))
  with pytest.raises(ValueError, match="multiple of updates_per_iteration"):
    late.step(_view(1, 6, ()))


# -- gated pool: what max_pool counts -----------------------------------------
class _ScriptedGate:
  """A PromotionGate whose verdict per generation is scripted."""

  name = "scripted"

  def __init__(self, passes):
    self._passes = dict(passes)

  def seed(self, generation):
    """The gate's seed for a generation.

    Args:
      generation: the generation.

    Returns:
      The generation itself.
    """
    return int(generation)

  def passes(self, candidate, incumbent, result):
    """The scripted verdict of the result's generation.

    Args:
      candidate: unused.
      incumbent: unused.
      result: the gate result (its generation picks the verdict).

    Returns:
      Whether that generation passes.
    """
    del candidate, incumbent
    return bool(self._passes[int(result.generation)])


def _gated_play(ctl, verdicts, every):
  """Ticks ctl at every round start, applying its actions to a member book.

  Round r starts at u = r * every, r = 0..len(verdicts), each with generation
  r's gate cell.

  Args:
    ctl: a gated ScheduledPromotion (its gate a _ScriptedGate).
    verdicts: the gate verdict of generation 1, 2, ... (unused here: the gate
      holds them; only the count matters).
    every: snapshot_every.

  Returns:
    (live members, [(update, pool names, evicted names)] per round).
  """
  members, rounds = {}, []
  for r in range(len(verdicts) + 1):
    u = r * every
    gate = None
    if r:
      req = ctl.gate_request(_view(u, u, members.values()))
      assert req is not None and req.generation == r
      gate = controller.GateCell(
          r, "main", req.opponent, req.seed, 4, 0.0, 0.1, -0.2, 0.2
      )
    learners = () if not r else (controller.LearnerInfo("main", "main", u),)
    view = controller.LeagueView(
        iteration=u,
        members=tuple(members.values()),
        learners=learners,
        payoff=None,
        gate=gate,
    )
    acts = ctl.step(view)
    gone = []
    for a in acts:
      if isinstance(a, controller.Admit):
        members[a.member] = controller.MemberInfo(
            a.member, a.role, a.lifecycle, a.learner, u
        )
      elif isinstance(a, controller.SetLifecycle):
        gone.append(a.member)
        del members[a.member]
    rounds.append((u, ctl._pool, tuple(gone)))  # pylint: disable=protected-access
  return members, rounds


def _gated(verdicts, pool_counts=None, max_pool=2):
  """A ScheduledPromotion gated by a _ScriptedGate of `verdicts`.

  Args:
    verdicts: generation g + 1 passes iff verdicts[g].
    pool_counts: the pool_counts field (None: the default).
    max_pool: the pool size.

  Returns:
    The controller.
  """
  kw = {} if pool_counts is None else {"pool_counts": pool_counts}
  gate = _ScriptedGate({g + 1: v for g, v in enumerate(verdicts)})
  return controller.ScheduledPromotion(
      snapshot_every=2,
      harvest_every=2,
      max_pool=max_pool,
      pool_slots=1,
      self_weight=0.5,
      seed=0,
      gate=gate,
      gate_deals=4,
      **kw,
  )


def test_failed_gate_never_pushes_a_passed_member_out_with_pool_counts_passed():
  """With pool_counts "passed" a failed gate never evicts a passed member.

  Generations 1 and 2 pass, 3..8 fail. "passed": main@2 and main@4 stay the
  pool through every failed gate; "attempts" (the default) drops them once
  the newest max_pool harvest STEPS move past them (the attempts bug).
  """
  verdicts = [True, True] + [False] * 6
  live, rounds = _gated_play(_gated(verdicts, "passed"), verdicts, 2)
  assert all(pool == ("main@2", "main@4") for u, pool, _ in rounds if u >= 4)
  assert not any({"main@2", "main@4"} & set(gone) for _, _, gone in rounds)
  assert sorted(live) == [
      "main@16",
      "main@2",
      "main@4",
  ]  # + the lagged snapshot
  _, attempts = _gated_play(_gated(verdicts), verdicts, 2)
  assert [(u, pool) for u, pool, _ in attempts] == [
      (0, ()),
      (2, ("main@2",)),
      (4, ("main@2", "main@4")),
      (6, ("main@4",)),
      (8, ()),
      (10, ()),
      (12, ()),
      (14, ()),
      (16, ()),
  ]
  # A third pass pushes the oldest passed member out ("past max_pool").
  verdicts = [True, True, False, False, True]
  _, rounds = _gated_play(_gated(verdicts, "passed"), verdicts, 2)
  assert rounds[-1] == (10, ("main@4", "main@10"), ("main@2", "main@8"))


def _old_scheduled_pool(passed, u, every, max_pool):
  """The pool rule before pool_counts existed, verbatim.

  Args:
    passed: the steps whose gate passed.
    u: the current update.
    every: the snapshot period, in updates.
    max_pool: the window, in snapshot steps.

  Returns:
    The pool: the last max_pool snapshot steps (pool_steps), then the passed
    among them, as member names.
  """
  steps = tuple(range(every, u + 1, every))[-max_pool:]
  return tuple(f"main@{s}" for s in steps if s in passed)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pool_counts_attempts_reproduces_the_old_window_exactly(seed):
  """The default pool_counts keeps the old rule's pool and evictions.

  The default (and "attempts", and an instance pickled before the field
  existed) gives the old rule's pool and evictions at every round of a random
  pass / fail sequence.

  Args:
    seed: the seed of the pass / fail sequence.
  """
  rng_ = np.random.default_rng(seed)
  verdicts = [bool(v) for v in rng_.random(40) < 0.3]
  passed = {2 * (g + 1) for g, v in enumerate(verdicts) if v}
  old = _gated(verdicts, max_pool=3)
  pickled = copy.deepcopy(old)
  del pickled.__dict__["pool_counts"]  # as unpickled from before the field
  assert pickled.pool_counts == controller.POOL_COUNTS_ATTEMPTS
  runs = [
      _gated_play(c, verdicts, 2)[1]
      for c in (old, _gated(verdicts, "attempts", max_pool=3), pickled)
  ]
  assert runs[0] == runs[1] == runs[2]
  prev = ()
  for u, pool, gone in runs[0]:
    want = _old_scheduled_pool(passed, u, 2, 3)
    assert pool == want, (u, pool, want)
    prev_snap = () if u == 0 else (f"main@{u - 2}",)
    assert set(gone) == (set(prev) | set(prev_snap)) - set(want) - {f"main@{u}"}
    prev = pool


def test_every_eviction_logs_its_member_reason_and_update(caplog):
  """One ScheduledPromotion log line per eviction.

  Each names the member, the update and the reason.

  Args:
    caplog: pytest's log capture fixture.
  """
  verdicts = [True, False, False, False]
  with caplog.at_level("INFO"):
    _, rounds = _gated_play(_gated(verdicts), verdicts, 2)
  lines = [
      r.getMessage()
      for r in caplog.records
      if "ScheduledPromotion: evict" in r.getMessage()
  ]
  gone = []
  for u, _, g in rounds:
    gone += [(u, n) for n in g]
  assert len(lines) == len(gone) and gone
  assert (
      "ScheduledPromotion: evict main@2 at update 6 (round 3, pool_counts "
      "attempts): outside the newest max_pool 2 harvest steps (pool_counts "
      "attempts: failed gates count)"
  ) in lines
  assert (
      "ScheduledPromotion: evict main@4 at update 6 (round 3, pool_counts "
      "attempts): superseded snapshot (its gate failed)"
  ) in lines


def test_league_logs_and_emits_every_eviction(caplog, monkeypatch):
  """League._log_evictions: a log line per evicted member, a record per call.

  The record is one series record (league/evictions,
  league/evicted/<origin>).

  Args:
    caplog: pytest's log capture fixture.
    monkeypatch: pytest's monkeypatch fixture.
  """
  from open_spiel.python.scale import metrics as metrics_lib  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415
  from open_spiel.python.scale.league import loop  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  got = []
  monkeypatch.setattr(metrics_lib, "_SERIES_EMITTERS", [got.append])
  fake = type("L", (), {"updates_total": np.int64(1046)})()
  evicted = [
      (
          "main@50",
          controller.MemberInfo("main@50", "snapshot", "fixed", "main", 50),
          "superseded snapshot / past max_pool",
      ),
      (
          "exploiter-x3@111",
          controller.MemberInfo(
              "exploiter-x3@111", "exploiter", "candidate", "exploiter", 111
          ),
          "superseded by exploiter-x4@189",
      ),
  ]
  with caplog.at_level("INFO"):
    loop.League._log_evictions(fake, evicted, 523)  # pylint: disable=protected-access
  lines = [
      r.getMessage()
      for r in caplog.records
      if r.getMessage().startswith("league: evicted member")
  ]
  assert lines == [
      (
          "league: evicted member main@50 (role snapshot, from main, admitted"
          " at iteration 50) at iteration 523, update 1046: superseded snapshot"
          " / past max_pool"
      ),
      (
          "league: evicted member exploiter-x3@111 (role exploiter, from"
          " exploiter, admitted at iteration 111) at iteration 523, update"
          " 1046: superseded by exploiter-x4@189"
      ),
  ]
  assert len(got) == 1 and got[0]["step"] == 1046
  assert got[0]["league/evictions"] == 2
  assert got[0]["league/evicted/main"] == 50
  assert got[0]["league/evicted/exploiter"] == 111


def test_pool_counts_kwargs_leave_the_legacy_mode_out_of_the_config():
  kwargs = controller.pool_counts_kwargs("attempts")
  assert isinstance(kwargs, dict) and not kwargs, kwargs
  assert controller.pool_counts_kwargs("passed") == {"pool_counts": "passed"}
  with pytest.raises(ValueError):
    controller.pool_counts_kwargs("newest")
  with pytest.raises(ValueError):
    controller.ScheduledPromotion(
        snapshot_every=2,
        harvest_every=2,
        max_pool=2,
        self_weight=0.5,
        pool_counts="newest",
    )


# -- a snapshot-pool league on kuhn --------------------------------------------
def _kuhn_snapshot_pool(updates):
  """A Kuhn snapshot-pool run config.

  Args:
    updates: The learner's updates.

  Returns:
    The run config.
  """
  # Lazy: optional dependency: league_fixtures needs the games package.
  from open_spiel.python.scale.league.tests import league_fixtures  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  pol = league_fixtures.kuhn_policy(16)
  learner = league_fixtures.br_learner(pol).cfg

  learner = dataclasses.replace(learner, num_updates=updates)
  return runner.snapshot_pool(
      "snapshot_pool_kuhn_test",
      "kuhn_poker",
      kinds.PG(learner),
      dict(pol.spec.arch_kwargs),
      num_envs=32,
      num_steps=8,
      num_updates=updates,
      snapshot_every=2,
      harvest_every=2,
      max_pool=2,
      pool_slots=1,
      self_weight=0.75,
      keep_every=2,
  )


def test_snapshot_pool_league_on_kuhn_promotes_seats_and_compiles_once(
    tmp_path,
):
  """Snapshot pool league on kuhn promotes seats and compiles once."""
  cfg = _kuhn_snapshot_pool(8)
  assert cfg.league.pool_capacity == 2 and cfg.league.bench
  res = runner.run(cfg, str(tmp_path / "ck"))
  lg = res.league
  assert int(lg.learners["main"].updates) == 8
  live = sorted(n for n, m in lg.members.items() if m.lifecycle != "evicted")
  assert live == ["main@4", "main@6"]  # the pool: newest 2 harvested
  assert sorted(lg.store.ids()) == ["main@0", "main@2", "main@4", "main@6"]
  # positions hold the snapshot and the seated member only; the rest is benched
  assert set(n for n in lg.placed() if n) <= {"main@6", "main@4"}
  counts = lg.learners["main"].draw_counts
  # one table per round
  assert counts[0] == {"main@0": 32} and counts[1] == counts[0]
  assert sum(counts[-1].values()) == 32 and set(counts[-1]) <= {
      "main@6",
      "main@4",
  }
  # values change, shapes never: one rollout and one update compile
  assert lg.arena._rollout_jit._cache_size() == 1  # pylint: disable=protected-access
  # Reaches private _cache_size, _update_jit of the code under test.
  assert lg.learner._update_jit._cache_size() == 1  # pylint: disable=protected-access
  # a resumed snapshot-pool league picks up the controller and the benched
  # members
  more = runner.run(_kuhn_snapshot_pool_extend(), str(tmp_path / "ck"))
  assert int(more.league.learners["main"].updates) == 10
  assert sorted(
      n for n, m in more.league.members.items() if m.lifecycle != "evicted"
  ) == ["main@6", "main@8"]


def _kuhn_snapshot_pool_extend():
  """The snapshot-pool config extended to 10 updates.

  Returns:
    The run config.
  """
  cfg = _kuhn_snapshot_pool(10)
  pgc = dataclasses.replace(cfg.learner.learner, num_updates=10)
  return dataclasses.replace(
      cfg, learner=dataclasses.replace(cfg.learner, learner=pgc)
  )


# -- commit protocol and residency ----------------------------
class _KilledError(Exception):
  pass


def test_kill_before_the_commit_resumes_the_same_trajectory_stream(
    tmp_path, monkeypatch
):
  """A kill before the commit resumes the same trajectory stream.

  A process killed after League.save and the round's member artifacts of
  iteration 6 but before train_state.json names them: the resume (at 4)
  continues the checkpointed hands (redraw on_change: the ArenaSource state
  is in the commit) and recomputes 4..5 without comparing anything with the
  killed attempt's outputs, so every update's metrics and the final params
  equal the uninterrupted run's bitwise.

  Args:
    tmp_path: pytest's tmp_path.
    monkeypatch: pytest's monkeypatch.
  """
  # Lazy: optional dependency: league_fixtures needs the games package.
  from open_spiel.python.scale.league.tests import league_fixtures  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  full = runner.run(_kuhn_snapshot_pool(8), str(tmp_path / "full"))
  d = str(tmp_path / "cut")
  real = run_state.write_train_state

  def killed(ckpt, record):
    if int(record["step"]) == 6:
      raise _KilledError("killed between the attempt's outputs and the commit")
    real(ckpt, record)

  monkeypatch.setattr(run_state, "write_train_state", killed)
  with pytest.raises(_KilledError):
    runner.run(_kuhn_snapshot_pool(8), d)
  monkeypatch.setattr(run_state, "write_train_state", real)
  ts = run_state.read_train_state(d)
  assert ts["step"] == 4 and "6" not in ts["league_commits"]
  assert (
      len(glob.glob(f"{d}/{commit.COMMITS_DIR}/6-*/{commit.STATE_FILE}")) == 1
  )
  # published after commits only
  assert not artifact.exists(str(tmp_path / "cut"), 6)
  snap = commit.state_path(d, 4, ts["league_ref"])

  # The state file the league under test just wrote.
  sources = pickle.loads(snap.read_bytes())["sources"]  # noqa: S301
  # live hands
  assert set(sources) == {"main"} and sources["main"]["state"] is not None
  res = runner.run(_kuhn_snapshot_pool(8), d)
  assert len(res.metrics) == 4 and len(full.metrics) == 8  # updates 5..8
  for got, want in zip(res.metrics, full.metrics[4:]):
    assert got == want, (got, want)
  assert league_fixtures.leaves_equal(
      res.league.learner_params("main"), full.league.learner_params("main")
  )
  ts = run_state.read_train_state(d)
  assert (
      ts["step"] == 8 and len(glob.glob(f"{d}/{commit.COMMITS_DIR}/6-*")) == 2
  )
  assert ts["league_commits"]["6"].endswith(res.league.attempt)
  assert sorted(res.league.store.ids()) == sorted(full.league.store.ids())
  # the replayed admission
  assert res.league.attempt in res.league.store.get("main@4").path
  # restore loaded no evicted member
  for (
      name,
      where,
  ) in res.league.resident_members().items():
    if res.league.members[name].lifecycle == "evicted":
      assert where == "artifact", name


def _kuhn_snapshot_pool_mid_round(updates):
  """A kuhn snapshot-pool config that commits inside rounds.

  A snapshot-pool league committing every update (keep_every 1) with rounds of 4
  updates: commits 5, 6, 7 fall inside a round, where the pool and the mix
  do not change, so redraw="on_change" deals no fresh hands there.

  Args:
    updates: the run's updates.

  Returns:
    The run config.
  """
  # Lazy: optional dependency: league_fixtures needs the games package.
  from open_spiel.python.scale.league.tests import league_fixtures  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  pol = league_fixtures.kuhn_policy(16)
  learner = dataclasses.replace(
      league_fixtures.br_learner(pol).cfg, num_updates=updates
  )
  return runner.snapshot_pool(
      "snapshot_pool_kuhn_mid_round",
      "kuhn_poker",
      kinds.PG(learner),
      dict(pol.spec.arch_kwargs),
      num_envs=32,
      num_steps=8,
      num_updates=updates,
      snapshot_every=4,
      harvest_every=4,
      max_pool=2,
      pool_slots=1,
      self_weight=0.75,
      keep_every=1,
  )


def test_resume_inside_a_round_continues_the_checkpointed_hands(
    tmp_path, monkeypatch
):
  """A resume inside a round continues the checkpointed hands.

  Killed before iteration 7's commit, resumed at 6: 6 is inside a round
  (no admission, no pool or mix change), so the resumed learner's first
  update plays on from the hands in the commit's ArenaSource snapshot. Its
  metrics and final params equal the uninterrupted run's only if that
  snapshot is restored (fresh hands at 6 give different trajectories: a
  load_sources_state that drops the snapshot fails here).

  Args:
    tmp_path: pytest's tmp_path.
    monkeypatch: pytest's monkeypatch.
  """
  # Lazy: optional dependency: league_fixtures needs the games package.
  from open_spiel.python.scale.league.tests import league_fixtures  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  full = runner.run(_kuhn_snapshot_pool_mid_round(8), str(tmp_path / "full"))
  # 4 and 8 are the round starts; between them the draw table never changes
  counts = full.league.learners["main"].draw_counts
  assert counts[4] == counts[5] == counts[6] == counts[7], counts
  d = str(tmp_path / "cut")
  real = run_state.write_train_state

  def killed(ckpt, record):
    if int(record["step"]) == 7:
      raise _KilledError("killed between the attempt's outputs and the commit")
    real(ckpt, record)

  monkeypatch.setattr(run_state, "write_train_state", killed)
  with pytest.raises(_KilledError):
    runner.run(_kuhn_snapshot_pool_mid_round(8), d)
  monkeypatch.setattr(run_state, "write_train_state", real)
  assert run_state.read_train_state(d)["step"] == 6
  res = runner.run(_kuhn_snapshot_pool_mid_round(8), d)
  assert len(res.metrics) == 2 and len(full.metrics) == 8  # updates 7, 8
  for got, want in zip(res.metrics, full.metrics[6:]):
    assert got == want, (got, want)
  assert league_fixtures.leaves_equal(
      res.league.learner_params("main"), full.league.learner_params("main")
  )


def test_commit_without_publish_is_published_at_resume(tmp_path, monkeypatch):
  """A commit without its publish is published at resume.

  Killed after train_state.json commits iteration 4 but before the run's
  artifact is published: the resume publishes the committed state's
  artifact first (the run's artifact only ever holds committed params).

  Args:
    tmp_path: pytest's tmp_path.
    monkeypatch: pytest's monkeypatch.
  """
  # Lazy: optional dependency: league_fixtures needs the games package.
  from open_spiel.python.scale.league.tests import league_fixtures  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  d = str(tmp_path / "pub")
  # Reaches private _publish_run_artifact of the code under test.
  real = runner._publish_run_artifact  # pylint: disable=protected-access

  def killed(ckpt, cfg, lg, run_id, previous_step):
    """Kills the process before publishing iteration 4.

    Args:
      ckpt: the checkpoint directory.
      cfg: the run config.
      lg: the league.
      run_id: the run id.
      previous_step: the previous artifact step.

    Returns:
      The real publish's result.

    Raises:
      _KilledError: at iteration 4.
    """
    if int(lg.iteration) == 4:
      raise _KilledError("killed between the commit and the publish")
    return real(ckpt, cfg, lg, run_id, previous_step)

  monkeypatch.setattr(runner, "_publish_run_artifact", killed)
  with pytest.raises(_KilledError):
    runner.run(_kuhn_snapshot_pool(8), d)
  monkeypatch.setattr(runner, "_publish_run_artifact", real)
  ts = run_state.read_train_state(d)
  assert ts["step"] == 4 and ts["artifact_step"] == 4
  assert not artifact.exists(str(tmp_path / "pub"), 4)
  res = runner.run(_kuhn_snapshot_pool(8), d)
  want = run_state.restore(
      commit.learner_dir(d, "main", 4), 4, res.learner
  ).params
  assert league_fixtures.leaves_equal(artifact.load_artifact(d, 4).params, want)
  assert artifact.load_artifact(d, 8).manifest.step == 8


def test_device_resident_bytes_stay_bounded_across_admissions_and_evictions(
    tmp_path,
):
  """Device-resident bytes stay bounded across admissions.

  A snapshot every update, a pool of 2: 40 admissions and 37 evictions.
  Only the Arena's positions hold member weights on the device (benched:
  host numpy; evicted: none, loaded lazily from the artifact), so the live
  device bytes after 40 iterations exceed those after 10 by less than one
  member's weights (no growth by member
  count; 0-d metric scalars excluded).

  Args:
    tmp_path: pytest's tmp_path.
  """
  # Lazy: optional dependency: league_fixtures needs the games package.
  from open_spiel.python.scale.league.tests import league_fixtures  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  pol = league_fixtures.kuhn_policy(128)
  learner = dataclasses.replace(
      league_fixtures.br_learner(pol).cfg, num_updates=40
  )
  cfg = runner.snapshot_pool(
      "snapshot_pool_kuhn_residency",
      "kuhn_poker",
      kinds.PG(learner),
      dict(pol.spec.arch_kwargs),
      num_envs=32,
      num_steps=8,
      num_updates=40,
      snapshot_every=1,
      harvest_every=1,
      max_pool=2,
      pool_slots=1,
      self_weight=0.75,
      keep_every=1000,
  )
  lg = runner.build(cfg, str(tmp_path / "r"))
  member_bytes = sum(
      np.asarray(x).nbytes
      for x in jax.tree_util.tree_leaves(
          lg.driver.wrap(pol.init_params(jax.random.key(0)))
      )
  )

  def device_bytes():
    """Bytes of the live non-scalar device arrays.

    Returns:
      The total nbytes, after a garbage collection.
    """
    gc.collect()
    return sum(x.nbytes for x in jax.live_arrays() if x.ndim >= 1)

  seen = {}
  for it in range(40):
    lg.run(1)
    where = lg.resident_members()
    assert sum(w == "arena" for w in where.values()) <= cfg.league.pool_capacity
    for name, w in where.items():
      if lg.members[name].lifecycle == "evicted":
        assert w == "artifact", (it, name, w)
      if w == "host":
        assert all(
            isinstance(x, np.ndarray)
            # Reaches private _off_arena of the code under test.
            for x in jax.tree_util.tree_leaves(lg._off_arena[name])  # pylint: disable=protected-access
        ), name
    if it in (9, 39):
      seen[it] = device_bytes()
  assert len(lg.members) == 40
  assert sum(m.lifecycle == "evicted" for m in lg.members.values()) >= 37
  assert seen[39] - seen[9] < member_bytes, (seen, member_bytes)
  # an evicted member's weights come back from its artifact on demand
  evicted = sorted(
      n for n, m in lg.members.items() if m.lifecycle == "evicted"
  )[0]
  got = lg.member_params(evicted)
  assert all(isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(got))
  assert lg.resident_members()[evicted] == "artifact"  # not kept


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
