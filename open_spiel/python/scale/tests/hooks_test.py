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

"""Small generic hooks a deployment plugs into, each with a known answer.

* runner.register_run_hook: a hook that skips the update-0 eval, reads on
  its own schedule, adds to the train row, keeps state in LearnerSlot.tallies
  and stops the run; with no hook a run reads at 0 and every `every`;
* metrics.stage / register_stage_record: a run marks build, evals,
  first-update and training, each maker's record goes to every series
  emitter, and with no maker nothing is emitted;
* bc.register_step_padding: a stateless student's steps pass through the
  registered padding before its forward;
* deals: a canonical form's hash_tag makes its hash rules-free;
* controller.ScheduledPromotion.training_row_filter: the gate's own.
"""

import dataclasses
import hashlib
import sys

import numpy as np
import pytest

from open_spiel.python.scale import metrics
from open_spiel.python.scale import train
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import bc
from open_spiel.python.scale.learners import losses
from open_spiel.python.scale.learners.tests import bc_fixtures
from open_spiel.python.scale.sources import deals
from open_spiel.python.scale.sources.tests import helpers


def _cfg(num_updates=6):
  """mmd_kuhn_poker with a vs_uniform in-loop eval every 4 updates.

  Args:
    num_updates: The updates.

  Returns:
    The run config.
  """
  cfg = common.with_overrides(
      common.load_config("mmd_kuhn_poker"), num_updates=num_updates
  )
  return dataclasses.replace(
      cfg,
      eval=(runner.InLoopEval("vs_uniform", every=4, deals=2),),
      keep_every=1,
  )


def _reads(monkeypatch):
  """Spies on runner.eval_row: the updates whose eval rows run.

  Args:
    monkeypatch: The pytest monkeypatch fixture.

  Returns:
    The list the spy appends each update to.
  """
  got = []
  real = runner.eval_row

  def spy(ev, evaluator, lg, ls, u, series_step=None):
    """Records update u and runs the real eval row.

    Args:
      ev: the InLoopEval.
      evaluator: its Evaluator.
      lg: the League.
      ls: the learner slot.
      u: the update.
      series_step: the series step.

    Returns:
      The real eval row.
    """
    got.append(int(u))
    return real(ev, evaluator, lg, ls, u, series_step=series_step)

  monkeypatch.setattr(runner, "eval_row", spy)
  return got


class _EveryOther:
  """A run hook that reads after every second update.

  Counted in tallies, not at update 0, and stops (with a read) after update 3.
  """

  def initial_eval(self):
    """No eval at update 0.

    Returns:
      False.
    """
    return False

  def after_update(self, lg, ls, u, m, row):
    """Counts the update and decides the reads and the stop.

    Args:
      lg: unused.
      ls: the learner slot.
      u: the update.
      m: the update's metrics.
      row: the train row.

    Returns:
      The UpdateVerdict.
    """
    del lg
    assert "update_step" in m
    n = ls.tallies.get("seen", 0) + 1
    ls.tallies["seen"] = n
    row["hook/seen"] = n
    stop = u == 3
    return runner.UpdateVerdict(
        evals_due=n % 2 == 0 or stop, stop="hook stop" if stop else None
    )


def test_no_hook_reads_at_zero_and_every(tmp_path, monkeypatch):
  """No hook reads at zero and every."""
  monkeypatch.setattr(runner, "_RUN_HOOKS", [])
  reads = _reads(monkeypatch)
  r = train.run(_cfg(), str(tmp_path / "r"))
  assert reads == [0, 4, 6] and not r.league.stopped
  assert not r.league.learners["main"].tallies


def test_a_run_hook_schedules_reads_keeps_tallies_and_stops(
    tmp_path, monkeypatch
):
  """A run hook schedules reads keeps tallies and stops."""
  monkeypatch.setattr(runner, "_RUN_HOOKS", [])
  runner.register_run_hook(lambda cfg: None)  # a factory may decline a run
  runner.register_run_hook(lambda cfg: _EveryOther())
  with pytest.raises(TypeError, match="callable"):
    runner.register_run_hook("not callable")
  reads = _reads(monkeypatch)
  d = str(tmp_path / "r")
  r = train.run(_cfg(), d)
  # no update-0 read; the hook's at 2; the stop reads once more
  assert reads == [
      2,
      3,
  ]
  lg = r.league
  assert lg.stopped and lg.stop_reason == "hook stop" and lg.iteration == 3
  assert lg.learners["main"].tallies == {"seen": 3}
  rows = metrics.read_jsonl(f"{d}/metrics/{metrics.JSONL_NAME}")
  assert [
      x["train/main/hook/seen"] for x in rows if "train/main/hook/seen" in x
  ] == [1, 2, 3]


def test_stages_reach_every_emitter_through_the_record_makers(
    tmp_path, monkeypatch
):
  """Stages reach every emitter through the record makers."""
  got = []
  monkeypatch.setattr(metrics, "_SERIES_EMITTERS", [got.append])
  monkeypatch.setattr(metrics, "_STAGE_RECORDS", [])
  metrics.stage("boot")  # no maker: nothing emitted
  assert not got
  metrics.register_stage_record(
      lambda name: None if name == "evals" else {"stage": name}
  )
  monkeypatch.setattr(runner, "eval_row", lambda *a, **k: {})
  train.run(_cfg(2), str(tmp_path / "r"))
  assert [r["stage"] for r in got if "stage" in r] == [
      "build",
      "first-update",
      "training",
  ]
  with pytest.raises(ValueError, match="stage"):
    metrics.stage("unknown")


def test_step_padding_runs_for_a_stateless_student(monkeypatch):
  """Step padding runs for a stateless student."""
  monkeypatch.setattr(bc, "_STEP_PADDING", {})
  env = kuhn.KuhnEnv()
  pol = helpers.kuhn_tabular(env)
  assert not pol.spec.stateful
  cfg = bc_fixtures.cfg(
      loss=losses.DistillLossConfig(policy="ce_target", value="none")
  )
  batch = bc_fixtures.two_ply_batch(env, [0.25, 0.75])
  _, plain = bc.BCLearner(pol, cfg).loss(pol.init_params(), batch)
  seen = []

  def pad(steps, valid):
    """Records the valid mask's shape; the steps pass through.

    Args:
      steps: the steps.
      valid: the valid mask.

    Returns:
      steps.
    """
    seen.append(np.asarray(valid).shape)
    return steps

  bc.register_step_padding(pol.spec.obs_format, pad)
  # the same fn again: a no-op
  bc.register_step_padding(pol.spec.obs_format, pad)
  with pytest.raises(ValueError, match="already has"):
    bc.register_step_padding(pol.spec.obs_format, lambda s, v: s)
  with pytest.raises(ValueError, match="obs_format"):
    bc.register_step_padding("", pad)
  _, padded = bc.BCLearner(pol, cfg).loss(pol.init_params(), batch)
  assert seen == [np.shape(batch.valid)]
  assert float(padded["loss_policy"]) == float(plain["loss_policy"])


def test_a_hash_tag_makes_the_deal_hash_rules_free(monkeypatch):
  """A hash tag makes the deal hash rules free."""
  monkeypatch.setattr(deals, "_CANON", {})

  def plain(info, labels):
    """A canonical form: the labels' deal.

    Args:
      info: unused.
      labels: the labels.

    Returns:
      u8 deal.
    """
    del info
    return np.asarray(labels["d"], np.uint8)

  def tagged(info, labels):
    """The same canonical form, with a hash tag.

    Args:
      info: the info states.
      labels: the labels.

    Returns:
      u8 deal.
    """
    return plain(info, labels)

  tagged.hash_tag = b"toy.deal\0"
  labels = {"d": np.asarray([[1, 2], [3, 4]], np.uint8)}
  deals.register_canon("toy_plain", plain)
  deals.register_canon("toy_tagged", tagged)
  assert deals.hash_tag("toy_plain") is None
  assert (
      deals.hash_prefix("toy_plain", "r1")
      == deals.HASH_TAG + b"toy_plain\0r1\0"
  )
  assert deals.deal_hashes(
      "toy_plain", "r1", None, labels
  ) != deals.deal_hashes("toy_plain", "r2", None, labels)
  got = deals.deal_hashes("toy_tagged", "r1", None, labels)
  assert got == deals.deal_hashes("toy_tagged", "r2", None, labels)
  assert got[0] == hashlib.sha256(b"toy.deal\0" + bytes([1, 2])).hexdigest()

  def bad(info, labels):
    """The same canonical form, with a malformed hash tag.

    Args:
      info: the info states.
      labels: the labels.

    Returns:
      u8 deal.
    """
    return plain(info, labels)

  bad.hash_tag = "no nul"
  # Refused where it is registered, and where it is read (a GameSpec.canon is
  # read without registration).
  with pytest.raises(TypeError, match="NUL-terminated"):
    deals.register_canon("toy_bad", bad)
  with pytest.raises(KeyError):
    deals.canon_of("toy_bad")
  with pytest.raises(TypeError, match="NUL-terminated"):
    deals.CANON_HOOKS.get(bad, "hash_tag")


def test_scheduled_promotion_hands_on_its_gates_row_filter():
  """Scheduled promotion hands on its gates row filter."""

  class Gate(controller.SeededGate):
    """A gate with a row filter."""

    def training_row_filter(self, env):
      """A marker filter.

      Args:
        env: the env.

      Returns:
        ("filter of", env).
      """
      return ("filter of", env)

  kw = dict(snapshot_every=1, harvest_every=1, max_pool=1, self_weight=1.0)
  assert controller.ScheduledPromotion(gate=Gate(1), **kw).training_row_filter(
      "e"
  ) == ("filter of", "e")
  assert (
      controller.ScheduledPromotion(
          gate=controller.SeededGate(1), **kw
      ).training_row_filter("e")
      is None
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
