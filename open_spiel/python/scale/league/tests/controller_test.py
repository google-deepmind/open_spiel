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

"""League controllers against synthetic payoff matrices (no training).

Pins each controller's actions for known views: PSRO and pipeline PSRO
(spawns, promotions, meta-strategy mixes), GatedPool (gate, active-set cap,
adaptive lambda, retirement, pool cap, anchors), ContinualExploiter and
BROracle (snapshot refreshes, lower-bound admission), PBT (exploit and
explore), LaggedAudit and Composite.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import sampler


def _m(name, role="psro", lifecycle="fixed", origin=None, at=0):
  """A MemberInfo.

  Args:
    name: The name.
    role: The role.
    lifecycle: The lifecycle.
    origin: The origin learner.
    at: The iteration admitted.

  Returns:
    The MemberInfo.
  """
  return controller.MemberInfo(
      name=name, role=role, lifecycle=lifecycle, origin=origin, admitted_at=at
  )


def _l(name, role, updates=0, scores=(), mix=None, at=0):
  """A LearnerInfo.

  Args:
    name: The name.
    role: The role.
    updates: Its updates.
    scores: Its scores.
    mix: Its mix (default pure self-play).
    at: The iteration it started.

  Returns:
    The LearnerInfo.
  """
  return controller.LearnerInfo(
      name=name,
      role=role,
      updates=updates,
      started_at=at,
      mix=mix if mix is not None else sampler.Mix(1.0, {}),
      scores=tuple(scores),
  )


def _view(it, members, learners, mean, se=None):
  """A LeagueView with the given payoffs.

  Args:
    it: The iteration.
    members: The members.
    learners: The learners.
    mean: The mean payoffs.
    se: Their standard errors.

  Returns:
    The LeagueView.
  """
  names = [m.name for m in members] + [l.name for l in learners]
  mean = np.asarray(mean, np.float64)
  assert mean.shape == (len(names), len(names)), (mean.shape, names)
  return controller.LeagueView(
      iteration=it,
      members=tuple(members),
      learners=tuple(learners),
      payoff=controller.payoff_matrix(names, mean, se),
  )


def _of(actions, kind):
  """The actions of type `kind`.

  Args:
    actions: The actions.
    kind: The action type.

  Returns:
    The actions.
  """
  return [a for a in actions if isinstance(a, kind)]


def _mix_of(actions, learner):
  """The last mix set for `learner`.

  Args:
    actions: The actions.
    learner: The learner.

  Returns:
    The Mix.
  """
  got = [a.mix for a in _of(actions, controller.SetMix) if a.learner == learner]
  assert got, f"no SetMix for {learner} in {actions}"
  return got[-1]


# -- views and actions ---------------------------------------------------------
def test_payoff_matrix_helper_and_view_lookups():
  """Payoff matrix helper and view lookups."""
  pm = controller.payoff_matrix(
      ["a@1", "b@1"], [[0.0, 1.0], [-1.0, 0.0]], [[0.0, 0.1], [0.1, 0.0]]
  )
  assert pm.names == ("a@1", "b@1")
  np.testing.assert_allclose(pm.lo, pm.mean - 1.959963984540054 * pm.se)
  v = _view(3, [_m("a@1"), _m("b@1")], [], [[0.0, 1.0], [-1.0, 0.0]])
  assert v.cell("a@1", "b@1") == (1.0, 0.0)
  assert v.member("b@1").name == "b@1"
  with pytest.raises(KeyError):
    v.member("zzz@1")
  with pytest.raises(ValueError):  # a view whose payoff misses a member
    controller.LeagueView(
        iteration=0,
        members=(_m("a@1"),),
        learners=(),
        payoff=controller.payoff_matrix(["b@1"], [[0.0]]),
    )
  with pytest.raises(ValueError):  # member names are artifact ids
    _m("a")


def test_actions_validate():
  """Actions validate."""
  controller.Admit(
      learner="br0", member="psro1@10", role="psro", lifecycle="fixed"
  )
  with pytest.raises(ValueError):
    # not an id
    controller.Admit(
        learner="br0", member="psro1", role="psro", lifecycle="fixed"
    )
  with pytest.raises(ValueError):
    controller.SetLifecycle(member="a@1", lifecycle="dead")
  with pytest.raises(ValueError):
    controller.CopyMutate(src="a", dst="b", shrink=-1.0)


# -- PSRO ----------------------------------------------------------------------
def test_psro_spawns_trains_admits_and_resolves_the_meta_game():
  """Psro spawns trains admits and resolves the meta game."""
  ctl = controller.PSRO(meta_solver="nash", iterations=2, br_updates=10)
  init = _m("init@0", role="init")
  acts = ctl.step(_view(0, [init], [], [[0.0]]))
  (sp,) = _of(acts, controller.SpawnLearner)
  assert (sp.name, sp.role, sp.init_from) == ("br0", "psro-br", None)
  assert _mix_of(acts, "br0") == sampler.Mix(0.0, {"init@0": 1.0})
  assert not _of(acts, controller.Admit)

  # Not ready: no admission, the mix stays the meta-strategy.
  acts = ctl.step(_view(1, [init], [_l("br0", "psro-br", 5)], np.zeros((2, 2))))
  assert not _of(acts, controller.Admit) and _mix_of(
      acts, "br0"
  ) == sampler.Mix(0.0, {"init@0": 1.0})

  # Ready: admit the BR as psro1@<updates>, retire it, spawn the next.
  acts = ctl.step(
      _view(2, [init], [_l("br0", "psro-br", 10)], np.zeros((2, 2)))
  )
  (ad,) = _of(acts, controller.Admit)
  assert (ad.learner, ad.member, ad.role, ad.lifecycle) == (
      "br0",
      "psro1@10",
      "psro",
      "fixed",
  )
  assert [a.name for a in _of(acts, controller.RetireLearner)] == ["br0"]
  assert [a.name for a in _of(acts, controller.SpawnLearner)] == ["br1"]

  # Re-evaluated with the new member: psro1 beats init, Nash is all on psro1.
  pop = [init, _m("psro1@10", origin="br0")]
  mean = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  acts = ctl.step(_view(2, pop, [_l("br1", "psro-br", 0)], mean))
  mix = _mix_of(acts, "br1")
  assert mix.self_play == 0.0
  assert mix.weights["psro1@10"] == pytest.approx(1.0, abs=1e-9)
  assert not _of(acts, controller.Admit) and not _of(acts, controller.Stop)

  # Second admission reaches iterations=2: admit, retire, Stop, no new BR.
  acts = ctl.step(_view(3, pop, [_l("br1", "psro-br", 10)], mean))
  assert [a.member for a in _of(acts, controller.Admit)] == ["psro2@10"]
  assert _of(acts, controller.Stop) and not _of(acts, controller.SpawnLearner)


def test_psro_meta_solver_is_pluggable():
  """Psro meta solver is pluggable."""
  rps = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
  pop = [_m("init@0", role="init"), _m("psro1@5"), _m("psro2@5")]
  mean = np.zeros((4, 4))
  mean[:3, :3] = rps
  for solver in (
      "nash",
      "alpha_rank",
      "fictitious_play",
      "regret_matching",
      "prd",
      "uniform",
  ):
    ctl = controller.PSRO(meta_solver=solver, iterations=5, br_updates=10)
    ctl.step(_view(0, pop, [], mean[:3, :3]))  # spawns br0
    acts = ctl.step(_view(1, pop, [_l("br0", "psro-br", 1)], mean))
    w = _mix_of(acts, "br0").weights
    np.testing.assert_allclose(
        [w["init@0"], w["psro1@5"], w["psro2@5"]],
        [1 / 3] * 3,
        atol=2e-2,
        err_msg=solver,
    )


def test_psro_plateau_promotes_early():
  """Psro plateau promotes early."""
  ctl = controller.PSRO(
      meta_solver="uniform",
      iterations=3,
      br_updates=100,
      min_br_updates=2,
      plateau_patience=2,
      plateau_threshold=0.01,
  )
  init = _m("init@0", role="init")
  ctl.step(_view(0, [init], [], [[0.0]]))
  acts = ctl.step(
      _view(
          1,
          [init],
          [_l("br0", "psro-br", 3, scores=(0.1, 0.5, 0.6))],
          np.zeros((2, 2)),
      )
  )
  assert not _of(acts, controller.Admit)  # still improving
  acts = ctl.step(
      _view(
          2,
          [init],
          [_l("br0", "psro-br", 4, scores=(0.1, 0.5, 0.6, 0.605, 0.606))],
          np.zeros((2, 2)),
      )
  )
  assert [a.member for a in _of(acts, controller.Admit)] == ["psro1@4"]


def test_pipeline_psro_levels_mixes_and_promotion():
  """Pipeline psro levels mixes and promotion."""
  ctl = controller.PSRO(
      meta_solver="uniform",
      iterations=4,
      br_updates=10,
      pipeline_depth=2,
      active_weight_share=0.5,
      pipeline_warm_start=True,
  )
  init = _m("init@0", role="init")
  acts = ctl.step(_view(0, [init], [], [[0.0]]))
  assert [
      (a.name, a.init_from) for a in _of(acts, controller.SpawnLearner)
  ] == [
      ("br0", None),
      ("br1", None),
  ]
  assert _mix_of(acts, "br0") == sampler.Mix(0.0, {"init@0": 1.0})
  m1 = _mix_of(acts, "br1")
  assert m1.weights == {"init@0": 0.5, "br0": 0.5}  # lower worker, live
  assert ctl.levels() == {"br0": 0, "br1": 1}
  # br1 being ready does not promote it: only the lowest worker is promoted.
  acts = ctl.step(
      _view(
          1,
          [init],
          [_l("br0", "psro-br", 5), _l("br1", "psro-br", 10)],
          np.zeros((3, 3)),
      )
  )
  assert not _of(acts, controller.Admit)
  acts = ctl.step(
      _view(
          2,
          [init],
          [_l("br0", "psro-br", 10), _l("br1", "psro-br", 15)],
          np.zeros((3, 3)),
      )
  )
  assert [a.member for a in _of(acts, controller.Admit)] == ["psro1@10"]
  assert [a.name for a in _of(acts, controller.RetireLearner)] == ["br0"]
  # The new top worker is warm-started from the shifted top worker
  # (pipeline_warm_start).
  assert [
      (a.name, a.init_from) for a in _of(acts, controller.SpawnLearner)
  ] == [("br2", "br1")]
  assert ctl.levels() == {"br1": 0, "br2": 1}
  assert _mix_of(acts, "br2").weights == {"init@0": 0.5, "br1": 0.5}


def test_psro_time_averaged_meta_strategy():
  """Psro time averaged meta strategy."""
  ctl = controller.PSRO(
      meta_solver="nash", iterations=5, br_updates=10, time_averaged=True
  )
  a, b = _m("init@0", role="init"), _m("psro1@10")
  ctl.step(_view(0, [a], [], [[0.0]]))
  # One member: nash = [1]; two members, b dominates: nash = [0, 1];
  # time average over the two solves = [(1 + 0)/2, (0 + 1)/2].
  mean = np.zeros((3, 3))
  mean[0, 1], mean[1, 0] = -1.0, 1.0
  acts = ctl.step(_view(1, [a, b], [_l("br0", "psro-br", 1)], mean))
  w = _mix_of(acts, "br0").weights
  assert w["init@0"] == pytest.approx(0.5) and w["psro1@10"] == pytest.approx(
      0.5
  )


# -- GatedPool -----------------------------------------------------------------
def _fly_view(it, members, r, se=0.1):
  """Members vs the live learner 'main': member i scores r[i] (EV vs main).

  Args:
    it: The iteration.
    members: The members.
    r: Each member's EV against main.
    se: The standard error of every cell.

  Returns:
    The LeagueView.
  """
  names = [m.name for m in members]
  n = len(names) + 1
  mean = np.zeros((n, n))
  s = np.zeros((n, n))
  for i, v in enumerate(r):
    mean[i, -1], mean[-1, i] = v, -v
    s[i, -1] = s[-1, i] = se
  return _view(it, members, [_l("main", "main", 100)], mean, s)


def test_gated_pool_gate_lambda_weights_and_mix_known_answer():
  """Gated pool gate lambda weights and mix known answer."""
  ctl = controller.GatedPool(main="main", gate=controller.GateConfig())
  members = [
      _m("m1@1", "exploiter", "candidate"),
      _m("m2@1", "exploiter", "candidate"),
      _m("m3@1", "exploiter", "active"),
  ]
  acts = ctl.step(_fly_view(5, members, [2.0, 0.3, -1.0]))
  life = {a.member: a.lifecycle for a in _of(acts, controller.SetLifecycle)}
  assert life == {
      "m1@1": "active",
      "m3@1": "inactive",
  }  # m2 in the dead band: candidate
  mix = _mix_of(acts, "main")
  # lambda = clip(0.05 + 0.1 * (2.0 - 0.5), 0.05, 0.28) = 0.2; eps 0.05 on the
  # inactive m3.
  assert mix.weights["m1@1"] == pytest.approx(0.2)
  assert mix.weights["m3@1"] == pytest.approx(0.05)
  assert mix.self_play == pytest.approx(0.75)
  assert "m2@1" not in mix.support()


def test_gated_pool_spawns_main_and_folds_mass_without_members():
  """Gated pool spawns main and folds mass without members."""
  ctl = controller.GatedPool(main="main", gate=controller.GateConfig())
  acts = ctl.step(_view(0, [], [], np.zeros((0, 0))))
  assert [(a.name, a.role) for a in _of(acts, controller.SpawnLearner)] == [
      ("main", "main")
  ]
  assert _mix_of(acts, "main") == sampler.Mix(1.0, {})


def test_gated_pool_active_set_cap_keeps_the_strongest():
  """Gated pool active set cap keeps the strongest."""
  ctl = controller.GatedPool(
      main="main", gate=controller.GateConfig(active_set_max=1)
  )
  members = [
      _m("a@1", "exploiter", "active"),
      _m("b@1", "exploiter", "candidate"),
  ]
  acts = ctl.step(_fly_view(1, members, [1.0, 3.0]))
  life = {a.member: a.lifecycle for a in _of(acts, controller.SetLifecycle)}
  assert life == {"a@1": "inactive", "b@1": "active"}
  mix = _mix_of(acts, "main")
  assert mix.weights["b@1"] == pytest.approx(min(0.05 + 0.1 * 2.5, 0.28))


def test_gated_pool_active_member_at_or_below_tau_off_is_demoted():
  # Dead band holds it active (ucb > tau_off) but its fresh score <= tau_off: no
  # weight, so it is demoted to inactive.
  """Gated pool active member at or below tau off is demoted."""
  ctl = controller.GatedPool(main="main", gate=controller.GateConfig())
  acts = ctl.step(
      _fly_view(1, [_m("a@1", "exploiter", "active")], [0.15], se=0.1)
  )
  assert {
      a.member: a.lifecycle for a in _of(acts, controller.SetLifecycle)
  } == {"a@1": "inactive"}


def test_gated_pool_retirement_k_of_n_and_reinstatement():
  """Gated pool retirement k of n and reinstatement."""
  ret = controller.RetirementConfig(
      retire_threshold=0.5,
      retire_after=3,
      retire_window=4,
      archive_epsilon=0.02,
  )
  ctl = controller.GatedPool(
      main="main", gate=controller.GateConfig(), retirement=ret
  )
  m = [_m("x@1", "exploiter", "inactive")]
  seq = [
      -2.0,
      0.0,
      -2.0,
      -2.0,
  ]  # beaten (ucb < -0.5) on 3 of the 4 scored ticks
  acts = []
  for t, r in enumerate(seq):
    acts = ctl.step(_fly_view(t, m, [r]))
    if t < 3:
      assert not _of(acts, controller.SetLifecycle), (t, acts)
  assert {
      a.member: a.lifecycle for a in _of(acts, controller.SetLifecycle)
  } == {"x@1": "archived"}
  m = [_m("x@1", "exploiter", "archived")]
  acts = ctl.step(_fly_view(4, m, [-2.0]))
  assert _mix_of(acts, "main").weights["x@1"] == pytest.approx(0.02)
  acts = ctl.step(_fly_view(5, m, [1.5]))  # LCB > tau_on: reinstated
  assert {
      a.member: a.lifecycle for a in _of(acts, controller.SetLifecycle)
  } == {"x@1": "active"}


def test_gated_pool_pool_cap_evicts_archived_then_weakest_inactive():
  """Gated pool pool cap evicts archived then weakest inactive."""
  ret = controller.RetirementConfig(pool_cap=2)
  ctl = controller.GatedPool(
      main="main", gate=controller.GateConfig(), retirement=ret
  )
  members = [
      _m("act@1", "exploiter", "active"),
      _m("in1@1", "exploiter", "inactive"),
      _m("in2@1", "exploiter", "inactive"),
      _m("arc@1", "exploiter", "archived"),
  ]
  acts = ctl.step(_fly_view(1, members, [2.0, -0.1, -0.3, -0.2]))
  life = {a.member: a.lifecycle for a in _of(acts, controller.SetLifecycle)}
  assert life == {"arc@1": "evicted", "in2@1": "evicted"}


def test_gated_pool_anchors_are_never_gated_and_ride_the_audit_tail():
  """Gated pool anchors are never gated and ride the audit tail."""
  ctl = controller.GatedPool(
      main="main", gate=controller.GateConfig(), anchors=("anc@1",)
  )
  members = [
      _m("anc@1", "anchor", "inactive"),
      _m("m@1", "exploiter", "active"),
  ]
  acts = ctl.step(_fly_view(1, members, [5.0, 2.0]))
  assert "anc@1" not in {a.member for a in _of(acts, controller.SetLifecycle)}
  mix = _mix_of(acts, "main")
  assert mix.weights["anc@1"] == pytest.approx(0.05)
  assert mix.weights["m@1"] == pytest.approx(0.2)


# -- continual exploiter, best-response oracle ---------------------------------
def test_continual_exploiter_refreshes_its_target_and_admits_on_lcb():
  """Continual exploiter refreshes its target and admits on lcb."""
  ctl = controller.ContinualExploiter(
      target="main", name="exploiter", refresh_every=3, admit_lcb=0.5
  )
  main = _l("main", "main", 40)
  acts = ctl.step(_view(0, [], [main], [[0.0]]))
  assert [(a.name, a.role) for a in _of(acts, controller.SpawnLearner)] == [
      ("exploiter", "exploiter")
  ]
  (snap,) = _of(acts, controller.Admit)
  assert (snap.learner, snap.member, snap.role, snap.lifecycle) == (
      "main",
      "main-target@40",
      "target",
      "fixed",
  )
  assert _mix_of(acts, "exploiter") == sampler.Mix(0.0, {"main-target@40": 1.0})
  members = [_m("main-target@40", "target", "fixed", origin="main")]
  learners = [main, _l("exploiter", "exploiter", 7)]
  mean = np.zeros((3, 3))
  se = np.full((3, 3), 0.1)
  mean[2, 0], mean[0, 2] = 0.6, -0.6  # LCB 0.6 - 0.196 < 0.5: not yet
  acts = ctl.step(_view(1, members, learners, mean, se))
  assert not _of(acts, controller.Admit)
  mean[2, 0], mean[0, 2] = 1.0, -1.0
  acts = ctl.step(_view(2, members, learners, mean, se))
  (ad,) = _of(acts, controller.Admit)
  assert (ad.learner, ad.member, ad.role, ad.lifecycle) == (
      "exploiter",
      "exploiter-x1@7",
      "exploiter",
      "candidate",
  )
  acts = ctl.step(_view(2, members, learners, mean, se))
  assert not _of(acts, controller.Admit)  # one admission per target
  acts = ctl.step(
      _view(
          3,
          members,
          [_l("main", "main", 70), _l("exploiter", "exploiter", 9)],
          mean,
          se,
      )
  )
  assert [a.member for a in _of(acts, controller.Admit)] == ["main-target@70"]
  assert _mix_of(acts, "exploiter") == sampler.Mix(0.0, {"main-target@70": 1.0})
  # The previous target leaves the pool.
  assert [
      (a.member, a.lifecycle) for a in _of(acts, controller.SetLifecycle)
  ] == [("main-target@40", "evicted")]


def test_br_oracle_launches_brs_and_gates_admission():
  """Br oracle launches brs and gates admission."""
  ctl = controller.BROracle(
      target="main",
      every=2,
      num_brs=2,
      max_updates=10,
      admission_lcb=0.0,
      max_concurrent=2,
  )
  main = _l("main", "main", 20)
  acts = ctl.step(_view(0, [], [main], [[0.0]]))
  assert [a.name for a in _of(acts, controller.SpawnLearner)] == [
      "oracle0",
      "oracle1",
  ]
  assert [a.member for a in _of(acts, controller.Admit)] == ["main-oracle@20"]
  t = "main-oracle@20"
  members = [_m(t, "target", "fixed", origin="main")]
  learners = [main, _l("oracle0", "oracle", 10), _l("oracle1", "oracle", 10)]
  mean = np.zeros((4, 4))
  se = np.full((4, 4), 0.1)
  mean[2, 0], mean[0, 2] = 1.0, -1.0  # oracle0 beats the target: admitted
  mean[3, 0], mean[0, 3] = 0.1, -0.1  # oracle1's LCB < 0: refused
  acts = ctl.step(_view(1, members, learners, mean, se))
  assert [
      (a.learner, a.member, a.lifecycle) for a in _of(acts, controller.Admit)
  ] == [("oracle0", "oracle0@10", "candidate")]
  assert sorted(a.name for a in _of(acts, controller.RetireLearner)) == [
      "oracle0",
      "oracle1",
  ]
  assert any(
      e["event"] == "refused" and e["learner"] == "oracle1" for e in ctl.events
  )
  # Iteration 2: a new launch (every=2).
  acts = ctl.step(_view(2, members, [main], np.zeros((2, 2))))
  assert [a.name for a in _of(acts, controller.SpawnLearner)] == [
      "oracle2",
      "oracle3",
  ]
  # the target snapshot main-oracle@20 already exists
  assert not _of(acts, controller.Admit)


# -- PBT, lagged audit, composite ----------------------------------------------
def test_pbt_bottom_copies_top_with_explore():
  """Pbt bottom copies top with explore."""
  ctl = controller.PBT(
      learners=("p0", "p1", "p2", "p3"),
      interval=2,
      fraction=0.25,
      shrink=0.9,
      perturb=0.1,
      self_play=0.0,
  )
  acts = ctl.step(_view(0, [], [], np.zeros((0, 0))))
  assert [a.name for a in _of(acts, controller.SpawnLearner)] == [
      "p0",
      "p1",
      "p2",
      "p3",
  ]
  learners = [_l(n, "pbt", 10) for n in ("p0", "p1", "p2", "p3")]
  # Fitness = mean payoff vs the other learners: p2 > p0 > p3 > p1.
  strength = np.array([0.5, -1.0, 1.0, 0.0])
  mean = strength[:, None] - strength[None, :]
  acts = ctl.step(_view(1, [], learners, mean))
  assert not _of(acts, controller.CopyMutate)  # not an interval tick
  m = _mix_of(acts, "p0")
  assert m.weights == pytest.approx({"p1": 1 / 3, "p2": 1 / 3, "p3": 1 / 3})
  acts = ctl.step(_view(2, [], learners, mean))
  (cm,) = _of(acts, controller.CopyMutate)
  assert (cm.src, cm.dst, cm.shrink, cm.perturb) == ("p2", "p1", 0.9, 0.1)
  # The exploit step banks the source as a member, then explores.
  assert [(a.learner, a.member) for a in _of(acts, controller.Admit)] == [
      ("p2", "p2-pbt@10")
  ]


def test_lagged_audit_refreshes_the_lagged_snapshot():
  """Lagged audit refreshes the lagged snapshot."""
  ctl = controller.LaggedAudit(
      main="main",
      audit=("p1@1", "p2@1"),
      refresh_every=2,
      self_play=0.5,
      lagged_mass=0.3,
  )
  members = [_m("p1@1", "audit", "fixed"), _m("p2@1", "audit", "fixed")]
  acts = ctl.step(_view(0, members, [], np.zeros((2, 2))))
  assert [a.name for a in _of(acts, controller.SpawnLearner)] == ["main"]
  mix = _mix_of(acts, "main")
  assert mix.self_play == pytest.approx(0.8) and mix.weights[
      "p1@1"
  ] == pytest.approx(0.1)
  acts = ctl.step(_view(1, members, [_l("main", "main", 50)], np.zeros((3, 3))))
  assert not _of(acts, controller.Admit)
  acts = ctl.step(
      _view(2, members, [_l("main", "main", 100)], np.zeros((3, 3)))
  )
  assert [a.member for a in _of(acts, controller.Admit)] == ["main-lag@100"]
  mix = _mix_of(acts, "main")
  assert mix.weights["main-lag@100"] == pytest.approx(0.3)
  assert mix.weights["p2@1"] == pytest.approx(
      0.1
  ) and mix.self_play == pytest.approx(0.5)
  members.append(_m("main-lag@100", "lagged", "fixed", origin="main"))
  acts = ctl.step(
      _view(4, members, [_l("main", "main", 200)], np.zeros((4, 4)))
  )
  assert [a.member for a in _of(acts, controller.Admit)] == ["main-lag@200"]
  assert [
      (a.member, a.lifecycle) for a in _of(acts, controller.SetLifecycle)
  ] == [("main-lag@100", "evicted")]


def test_composite_concatenates_and_keeps_order():
  """Composite concatenates and keeps order."""
  ctl = controller.Composite([
      controller.FixedMix("a", sampler.Mix(1.0, {}), role="main"),
      controller.FixedMix("b", sampler.Mix(1.0, {}), role="main"),
  ])
  acts = ctl.step(_view(0, [], [], np.zeros((0, 0))))
  assert [a.name for a in _of(acts, controller.SpawnLearner)] == ["a", "b"]


def test_composite_forwards_meta_info_and_refuses_two_meta_children():
  """Composite forwards meta info and refuses two meta children."""
  psro = controller.PSRO(meta_solver="uniform", iterations=2, br_updates=10)
  ctl = controller.Composite(
      [controller.FixedMix("a", sampler.Mix(1.0, {}), role="main"), psro]
  )
  assert ctl.last_meta is None and not ctl.last_info and not ctl.draw_guards()
  psro.last_meta = (("x",), np.ones(1))
  assert ctl.last_meta is psro.last_meta
  assert ctl.gate_request(_view(0, [], [], np.zeros((0, 0)))) is None
  with pytest.raises(ValueError, match="last_meta"):
    controller.Composite([
        psro,
        controller.PSRO(meta_solver="uniform", iterations=2, br_updates=10),
    ])


def test_controllers_do_not_mutate_the_view():
  """Controllers do not mutate the view."""
  ctl = controller.GatedPool(main="main", gate=controller.GateConfig())
  members = [_m("m1@1", "exploiter", "candidate")]
  v = _fly_view(0, members, [2.0])
  before = v.payoff.mean.copy()
  ctl.step(v)
  np.testing.assert_array_equal(v.payoff.mean, before)
  assert v.members[0].lifecycle == "candidate"


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
