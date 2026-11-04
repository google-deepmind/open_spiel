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

"""league.runner and League checkpoint / resume, on kuhn.

* resume == uninterrupted: pipeline-PSRO run 6 iterations straight, or 3,
saved, restored into a freshly built League, then 3 more, gives the same
members, the same learner params and the same payoff matrix bitwise;
replaying an uncommitted iteration (its member artifacts and learner
states already on disk from another attempt) writes the new attempt's
own outputs and compares nothing (league.commit), even when the replay
diverges; retired learners' states leave the device at retirement and
are released at the next save;
* train.run dispatches a LeagueRunConfig ("league" kind): checkpoints commit
train_state.json (kind "league"), a second run with a longer
num_iterations resumes, and the result equals the uninterrupted run;
* LeagueRunConfig refuses a scheduled LR and an unknown controller.
"""

import dataclasses
import os
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale import train
from open_spiel.python.scale.ckpt import run_state
from open_spiel.python.scale.core import types
from open_spiel.python.scale.league import artifact_store
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league.tests import league_fixtures
from open_spiel.python.scale.networks import catalog


def _cfg():
  """A small LeagueConfig.

  Returns:
    The LeagueConfig.
  """
  return loop.LeagueConfig(
      num_envs=32,
      num_steps=8,
      updates_per_iteration=2,
      pool_capacity=8,
      eval_deals=16,
      pool_rule=types.ARGMAX,
      seed=2,
  )


def _league(run_dir):
  """A Kuhn PSRO league in `run_dir`.

  Args:
    run_dir: The run folder.

  Returns:
    The League.
  """
  pol = league_fixtures.kuhn_policy(16)
  # 10 promotions: no Stop within 6 iterations
  ctl = controller.PSRO(
      meta_solver="nash",
      iterations=10,
      br_updates=4,
      pipeline_depth=2,
      active_weight_share=0.5,
  )
  store = artifact_store.ArtifactStore(os.path.join(run_dir, "store"))
  return loop.League(
      league_fixtures.env(),
      pol,
      league_fixtures.br_learner(pol),
      ctl,
      store,
      run_dir,
      _cfg(),
      initial_members={"init@0": league_fixtures.init_params(pol, 0)},
  )


def _snapshot(lg):
  """What a resume must restore of the league.

  Args:
    lg: The league.

  Returns:
    (members, learners, learner params, payoff means).
  """
  return (
      sorted(lg.members),
      sorted(lg.learners),
      {n: lg.learner_params(n) for n in lg.learners},
      None if lg.payoff is None else np.asarray(lg.payoff.mean),
  )


def _same(a, b):
  assert a[0] == b[0] and a[1] == b[1]
  for n in a[2]:
    assert league_fixtures.leaves_equal(a[2][n], b[2][n]), n
  np.testing.assert_array_equal(a[3], b[3])


def test_league_resume_equals_uninterrupted(tmp_path):
  """League resume equals uninterrupted."""
  straight = _league(str(tmp_path / "a"))
  straight.run(6)
  d = str(tmp_path / "b")
  first = _league(d)
  first.run(3)
  r3 = first.save(d, 3)
  first.run(1)
  # an attempt at iteration 4 that is never committed
  r4_lost = first.save(d, 4)
  again = _league(d)
  assert again.attempt != first.attempt
  again.restore(d, 3, r3)
  assert again.iteration == 3 and set(again.members) <= set(first.members)
  again.run(1)
  # its own outputs: the uncommitted attempt's are not read
  r4 = again.save(d, 4)
  assert r4 != r4_lost and commit.ref_step(r4) == 4
  again.run(2)
  assert again.iteration == straight.iteration == 6
  _same(_snapshot(again), _snapshot(straight))
  assert not again.stopped and not straight.stopped


def test_a_divergent_replay_of_uncommitted_work_is_accepted(tmp_path):
  """A divergent replay of uncommitted work is accepted.

  No identical-replay demand: the resumed attempt diverges on purpose
  (its learners' params are halved), writes its own learner states and
  member artifacts, and a restore of its ref sees its work; the store
  registers its admissions, and no entry of the lost attempt survives.

  Args:
    tmp_path: pytest's tmp_path.
  """
  d = str(tmp_path / "div")
  first = _league(d)
  first.run(2)
  r2 = first.save(d, 2)
  before = set(first.members)
  first.run(2)
  first.save(d, 4)  # never committed
  lost = set(first.members) - before
  assert lost, "the lost attempt admitted no member: the test needs one"
  again = _league(d)
  again.restore(d, 2, r2)
  copies = [
      controller.CopyMutate(n, n, shrink=0.5) for n in sorted(again.learners)
  ]
  # Reaches private _apply of the code under test.
  again._apply(copies, 2)  # pylint: disable=protected-access
  again.run(2)
  r4 = again.save(d, 4)
  for n in again.learners:
    if n in first.learners:
      assert not league_fixtures.leaves_equal(
          again.learner_params(n), first.learner_params(n)
      ), n
  third = _league(d)
  third.restore(d, 4, r4)
  for n in again.learners:
    assert league_fixtures.leaves_equal(
        third.learner_params(n), again.learner_params(n)
    ), n
  assert set(third.store.ids()) == set(third.members)
  for m in set(third.members) - before:
    assert again.attempt in third.store.get(m).path, (
        m,
        third.store.get(m).path,
    )
    assert league_fixtures.leaves_equal(
        third.member_params(m), again.member_params(m)
    ), m


def test_retired_learner_state_leaves_the_device_and_is_released_at_save(
    tmp_path,
):
  """Retired learner state leaves the device and is released at save."""
  d = str(tmp_path / "ret")
  lg = _league(d)
  lg.run(6)
  assert lg.retired, "pipeline PSRO retired no worker in 6 iterations"
  for name, ls in lg.retired.items():
    leaves = jax.tree_util.tree_leaves(ls.state)
    assert leaves and all(
        isinstance(x, np.ndarray)
        or jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
        for x in leaves
    ), name
  lg.save(d, 6)
  assert all(ls.state is None for ls in lg.retired.values())
  # the records stay
  assert all(int(ls.updates) > 0 for ls in lg.retired.values())


def test_restore_refuses_another_config(tmp_path):
  """Restore refuses another config."""
  d = str(tmp_path / "r")
  lg = _league(d)
  lg.run(1)
  r1 = lg.save(d, 1)
  pol = league_fixtures.kuhn_policy(16)
  other = loop.League(
      league_fixtures.env(),
      pol,
      league_fixtures.br_learner(pol),
      controller.PSRO(
          meta_solver="nash", iterations=10, br_updates=4, pipeline_depth=2
      ),
      artifact_store.ArtifactStore(os.path.join(d, "store2")),
      d,
      dataclasses.replace(_cfg(), eval_deals=32),
      initial_members={"init@0": league_fixtures.init_params(pol, 0)},
  )
  with pytest.raises(ValueError, match="different config"):
    other.restore(d, 1, r1)


def _run_cfg(num_iterations):
  """A Kuhn PSRO run config.

  Args:
    num_iterations: The iterations.

  Returns:
    The LeagueRunConfig.
  """
  learner = league_fixtures.br_learner(league_fixtures.kuhn_policy(16)).cfg
  return runner.LeagueRunConfig(
      name="league_kuhn_psro_test",
      env="kuhn_poker",
      policy_kwargs=common.network_policy_kwargs(
          "kuhn_poker",
          "kuhn_poker_info_index",
          [12],
          2,
          2,
          catalog.mlp([16], "relu", "separate"),
      ),
      learner=kinds.PG(learner),
      league=_cfg(),
      controller="psro",
      controller_kwargs=dict(
          meta_solver="nash", iterations=10, br_updates=4, pipeline_depth=2
      ),
      num_iterations=num_iterations,
      num_initial=1,
      keep_every=2,
      seed=2,
      artifact_learner="",
  )


def test_train_run_league_kind_checkpoints_and_resumes(tmp_path):
  """Train run league kind checkpoints and resumes."""
  cfg6 = _run_cfg(6)
  assert common.learner_kind(cfg6) == "pg" and common.run_length(cfg6) == 6
  full = train.run(cfg6, str(tmp_path / "full"))
  d = str(tmp_path / "split")
  part = train.run(_run_cfg(3), d)
  ts = run_state.read_train_state(d)
  assert ts["kind"] == "league" and ts["step"] == 3 and len(part.rows) == 3
  rest = train.run(cfg6, d)  # resumes at 3 (constant LR: the run may extend)
  assert len(rest.rows) == 3
  ts = run_state.read_train_state(d)
  assert ts["step"] == 6 and sorted(ts["league_commits"], key=int) == [
      "2",
      "3",
      "4",
      "6",
  ]
  assert (
      ts["league_ref"] == ts["league_commits"]["6"]
      and commit.ref_step(ts["league_ref"]) == 6
  )
  _same(_snapshot(rest.league), _snapshot(full.league))
  assert [m["members"] for m in part.rows + rest.rows] == [
      m["members"] for m in full.rows
  ]
  assert os.path.exists(os.path.join(d, "metrics"))


def test_league_run_config_validation():
  """League run config validation."""
  cfg = _run_cfg(2)
  with pytest.raises(ValueError, match="constant LR"):
    dataclasses.replace(
        cfg,
        learner=kinds.PG(
            dataclasses.replace(
                cfg.learner.learner, lr_end=0.0, anneal_lr=True, num_updates=4
            )
        ),
    )
  with pytest.raises(KeyError, match="unknown controller"):
    dataclasses.replace(cfg, controller="nope")


def test_a_stopped_league_run_ends_and_resumes_as_finished(tmp_path):
  """A stopped league run ends and resumes as finished.

  PSRO's Stop ends the run before num_iterations; the commit records it
  and a resume of the finished run trains nothing.

  Args:
    tmp_path: pytest's tmp_path.
  """
  cfg = dataclasses.replace(
      _run_cfg(20),
      controller_kwargs=dict(
          meta_solver="nash", iterations=2, br_updates=2, pipeline_depth=1
      ),
  )
  d = str(tmp_path / "stop")
  r = train.run(cfg, d)
  ts = run_state.read_train_state(d)
  assert (
      r.league.stopped
      and ts["stopped"]
      and ts["step"] == r.league.iteration < 20
  )
  assert sum(1 for m in ts["members"] if m.startswith("psro")) == 2
  again = train.run(cfg, d)
  assert not again.rows and again.league.iteration == ts["step"]


def test_on_commit_hooks_run_after_each_keep(tmp_path, monkeypatch):
  """On commit hooks run after each keep."""
  # Reaches private _ON_COMMIT of the code under test.
  monkeypatch.setattr(runner, "_ON_COMMIT", dict(runner._ON_COMMIT))  # pylint: disable=protected-access
  # Reaches private _ALWAYS_ON_COMMIT of the code under test.
  always = list(runner._ALWAYS_ON_COMMIT)  # pylint: disable=protected-access
  monkeypatch.setattr(runner, "_ALWAYS_ON_COMMIT", always)
  seen = []
  runner.register_on_commit(
      "record",
      lambda ckpt, step, ref: seen.append(
          (step, ref, run_state.read_train_state(ckpt)["league_ref"])
      ),
  )
  with pytest.raises(KeyError, match="not registered"):
    dataclasses.replace(_run_cfg(4), on_commit=("nope",))
  d = str(tmp_path / "hooked")
  train.run(dataclasses.replace(_run_cfg(4), on_commit=("record",)), d)
  assert [s for s, _, _ in seen] == [2, 4]
  assert all(r == committed for _, r, committed in seen)  # after the commit


# -- a gs:// run dir gets a local scratch dir ----------------------------------
class _StopError(Exception):
  pass


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
