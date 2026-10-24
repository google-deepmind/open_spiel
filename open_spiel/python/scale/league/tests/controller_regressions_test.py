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

"""Controller edge cases (pipeline PSRO, best-response oracle).

Controller edge cases: pipeline PSRO promotes at most once per iteration,
and the best-response oracle evicts a target snapshot no job attacks.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import sampler


def _m(name, role="psro", lifecycle="fixed"):
  """A MemberInfo.

  Args:
    name: The name.
    role: The role.
    lifecycle: The lifecycle.

  Returns:
    The MemberInfo.
  """
  return controller.MemberInfo(name=name, role=role, lifecycle=lifecycle)


def _l(name, role, updates=0, scores=()):
  """A LearnerInfo in pure self-play.

  Args:
    name: The name.
    role: The role.
    updates: Its updates.
    scores: Its scores.

  Returns:
    The LearnerInfo.
  """
  return controller.LearnerInfo(
      name=name,
      role=role,
      updates=updates,
      mix=sampler.Mix(1.0, {}),
      scores=tuple(scores),
  )


def _view(it, members, learners):
  """A LeagueView with zero payoffs.

  Args:
    it: The iteration.
    members: The members.
    learners: The learners.

  Returns:
    The LeagueView.
  """
  names = [m.name for m in members] + [l.name for l in learners]
  n = len(names)
  return controller.LeagueView(
      iteration=it,
      members=tuple(members),
      learners=tuple(learners),
      payoff=controller.payoff_matrix(names, np.zeros((n, n))),
  )


def test_pipeline_psro_promotes_at_most_once_per_iteration():
  """Pipeline PSRO promotes at most once per iteration.

  League.tick calls step() again in the same iteration after every
  structural action. With pipeline_depth=2 both initial workers reach
  br_updates together; after br0's promotion, br1 (then lowest, never trained
  as the lowest) must not be promoted in the same iteration.
  """
  ctl = controller.PSRO(
      meta_solver="uniform", iterations=4, br_updates=10, pipeline_depth=2
  )
  init = _m("init@0", role="init")
  ctl.step(_view(0, [init], []))  # spawns br0, br1
  # Iteration 5, round 0: both workers at br_updates; br0 is promoted.
  acts0 = ctl.step(
      _view(5, [init], [_l("br0", "psro-br", 10), _l("br1", "psro-br", 10)])
  )
  assert [a.member for a in acts0 if isinstance(a, controller.Admit)] == [
      "psro1@10"
  ]
  # Iteration 5, round 1 (League.tick re-evaluates after the structural
  # actions and calls step again): the view has psro1, br1 and br2.
  members = [init, _m("psro1@10")]
  acts1 = ctl.step(
      _view(5, members, [_l("br1", "psro-br", 10), _l("br2", "psro-br", 0)])
  )
  promoted = [a.member for a in acts1 if isinstance(a, controller.Admit)]
  assert not promoted, (
      f"a second promotion {promoted} in the same iteration 5 "
      "(tick round 1): the pipeline collapses"
  )


def test_br_oracle_evicts_a_target_snapshot_once_no_job_attacks_it():
  """BROracle evicts a target snapshot once no job attacks it.

  BROracle admits a `<main>-oracle@<step>` member per launch; once its BRs
  retire, a snapshot no running job targets is evicted (otherwise every stale
  snapshot would cost a payoff row and column every iteration).
  """
  ctl = controller.BROracle(
      target="main", every=1, num_brs=1, max_updates=5, max_concurrent=1
  )
  main0 = _l("main", "main", 0)
  acts = ctl.step(_view(0, [], [main0]))
  snap0 = [a.member for a in acts if isinstance(a, controller.Admit)]
  assert snap0 == ["main-oracle@0"], acts
  members = [_m("main-oracle@0", role="target")]
  acts = ctl.step(
      _view(1, members, [_l("main", "main", 10), _l("oracle0", "oracle", 5)])
  )
  # oracle0 finished (retired); a new launch targets main-oracle@10.
  assert any(
      isinstance(a, controller.RetireLearner) and a.name == "oracle0"
      for a in acts
  ), acts
  evicted = [
      a.member
      for a in acts
      if isinstance(a, controller.SetLifecycle) and a.lifecycle == "evicted"
  ]
  assert "main-oracle@0" in evicted, (
      "main-oracle@0 backs no job after oracle0 retired, yet it stays in the "
      f"pool: {acts}"
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
