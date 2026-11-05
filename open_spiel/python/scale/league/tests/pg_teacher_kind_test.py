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

"""kinds.PG(teacher=...): a frozen distillation teacher for a PG run.

* None (the default) is left out of the config hash, so every config built
  before the field hashes as before; a teacher needs learner.teacher_coef and
  teacher_coef needs a teacher; the teacher is never the warm start;
* the driver builds the teacher (the base of a routed artifact) and hands it
  to the learner; a teacher of another game is refused;
* a kuhn snapshot-pool run with a teacher trains (teacher_kl logged), names
  the teacher in its checkpoint driver state (member artifacts unchanged),
  never seats it (the pool holds the learner's own snapshots), and resumes
  bit for bit; a checkpoint with another teacher is refused.
"""

import dataclasses
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league.kinds import pg as pg_kind
from open_spiel.python.scale.league.tests import league_fixtures
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import catalog

COEF = schedules.Schedule("linear", 1.0, 0.0, decay_steps=4)


def _teacher(width=8):
  """The teacher: a kuhn network of `width`.

  Args:
    width: the network width.

  Returns:
    Its PolicyRef.
  """
  return refs.PolicyRef(
      arch_name="network", arch_kwargs=league_fixtures.kuhn_policy_kwargs(width)
  )


def _learner(updates=8):
  """The kickstarted PG learner config.

  Args:
    updates: num_updates.

  Returns:
    A br_learner config with the teacher schedule COEF.
  """
  pol = league_fixtures.kuhn_policy(16)
  return dataclasses.replace(
      league_fixtures.br_learner(pol).cfg,
      num_updates=updates,
      teacher_coef=COEF,
  )


def test_default_teacher_is_not_hashed_and_is_validated():
  plain = kinds.PG(pg.PGConfig(num_updates=10))
  assert "teacher" not in common.config_dict(plain)
  assert common.plain(plain)["teacher"] is None
  lc = _learner()
  spec = kinds.PG(lc, teacher=_teacher())
  assert common.config_dict(spec)["teacher"]["arch_name"] == "network"
  assert common.config_hash(spec) != common.config_hash(
      dataclasses.replace(spec, teacher=_teacher(4))
  )
  with pytest.raises(ValueError, match="set teacher"):
    kinds.PG(lc)
  with pytest.raises(ValueError, match="teacher_coef"):
    kinds.PG(pg.PGConfig(num_updates=10), teacher=_teacher())
  with pytest.raises(ValueError, match="warm start"):
    kinds.PG(lc, teacher=_teacher(), init=_teacher())


def test_driver_refuses_a_teacher_of_another_game():
  pol = league_fixtures.kuhn_policy(16)
  ref = refs.PolicyRef(
      arch_name="network",
      arch_kwargs=common.network_policy_kwargs(
          "leduc_poker",
          "leduc_poker_info_state",
          [30],
          leduc.LeducEnv().num_actions,
          2,
          catalog.mlp([8], "relu", "separate"),
      ),
  )
  d = pg_kind.PGDriver(
      pol, pg.PGLearner(pol, _learner()), teacher=ref, env=league_fixtures.env()
  )
  with pytest.raises(ValueError, match="leduc_poker"):
    d.init_state(jax.random.key(0), league_fixtures.env())


def test_driver_state_names_the_teacher():
  pol = league_fixtures.kuhn_policy(16)
  ref = _teacher()
  d = pg_kind.PGDriver(pol, pg.PGLearner(pol, _learner()), teacher=ref)
  assert d.host_state() == {"teacher": ref.id}
  d.load_host_state({"teacher": ref.id})
  with pytest.raises(ValueError, match="teacher cannot change"):
    d.load_host_state({"teacher": "another@1"})
  with pytest.raises(ValueError, match="teacher cannot change"):
    d.load_host_state({})
  bare = pg_kind.PGDriver(pol, pg.PGLearner(pol, pg.PGConfig(num_updates=4)))
  host = bare.host_state()
  assert isinstance(host, dict) and not host, host
  assert (
      d.artifact_extra() == bare.artifact_extra()
  )  # member artifacts: format unchanged
  bare.load_host_state({})


def _cfg(updates=8, keep_every=8):
  """A kuhn snapshot-pool run whose PG learner is kickstarted by _teacher().

  Args:
    updates: num_updates.
    keep_every: the snapshot keep interval.

  Returns:
    The league config.
  """
  pol = league_fixtures.kuhn_policy(16)
  return runner.snapshot_pool(
      "snapshot_pool_kuhn_teacher",
      "kuhn_poker",
      kinds.PG(_learner(updates), teacher=_teacher()),
      dict(pol.spec.arch_kwargs),
      num_envs=32,
      num_steps=8,
      num_updates=updates,
      snapshot_every=4,
      harvest_every=4,
      max_pool=2,
      pool_slots=1,
      self_weight=0.75,
      keep_every=keep_every,
  )


def test_a_kickstarted_run_trains_names_its_teacher_and_resumes(
    tmp_path, monkeypatch
):
  full = runner.run(_cfg(), str(tmp_path / "full"))
  assert len(full.metrics) == 8
  kls = [m["teacher_kl"] for m in full.metrics]
  coefs = [m["teacher_coef"] for m in full.metrics]
  assert all(np.isfinite(kls)) and kls[0] > 0, kls
  np.testing.assert_allclose(coefs[:5], [1.0, 0.75, 0.5, 0.25, 0.0])
  lrn = full.league.driver.learner
  assert lrn.teacher is not None and lrn.teacher.name == _teacher().id
  # frozen: the teacher's params are still its architecture's init
  pol, want = _teacher().build()
  assert league_fixtures.leaves_equal(jax.device_get(lrn.teacher.params), want)
  assert pol.spec.arch_kwargs == lrn.teacher.policy.spec.arch_kwargs
  # the league checkpoint's driver state names the teacher (resume checks it)
  assert full.league.driver.host_state() == {"teacher": _teacher().id}
  # never seated: the run's members are the learner's own snapshots
  ids = list(full.league.store.ids())
  assert ids and all(n.startswith("main") for n in ids), ids
  # preempted after 6 updates, resumed: the same updates with the same teacher
  d = str(tmp_path / "cut")
  real = loop.League.run
  state = {"up": False}

  def run(lg, n, *a, **kw):
    """League.run, recording when the league reaches iteration 6.

    Args:
      lg: the league.
      n: League.run's count.
      *a: its other arguments.
      **kw: its keyword arguments.

    Returns:
      League.run's result.
    """
    out = real(lg, n, *a, **kw)
    if lg.iteration >= 6:
      state["up"] = True
    return out

  monkeypatch.setattr(loop.League, "run", run)
  part = runner.run(_cfg(), d, preempted=lambda: state["up"])
  monkeypatch.undo()
  assert part.preempted and len(part.metrics) == 6
  res = runner.run(_cfg(), d)
  assert [m["teacher_kl"] for m in res.metrics] == kls[6:]
  assert league_fixtures.leaves_equal(
      res.league.learner_params("main"), full.league.learner_params("main")
  )


if __name__ == "__main__":
  x64.enable_x64()
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
