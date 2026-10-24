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

"""Learner-state rewrites a learner cannot take are refused.

The League rewrites a learner
state only through its state_dict's "params" / "opt_state" entries
(CopyMutate, SpawnLearner(init_from=...), reset_optimizer_each_iteration),
so a learner whose state_dict lacks them is refused at action time, loudly
and before anything changes. A PG learner whose state_dict renames those
entries (and has no init_from_params) stands in for such a learner; the plain
PG learner is the control.
"""

import dataclasses
import os
import re
import sys

import pytest

from open_spiel.python.scale.league import artifact_store
from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league.tests import league_fixtures


class _OpaqueLearner:
  """A PG learner without the params / opt_state entries.

  A PG learner whose state_dict says "weights" / "adam" and that has no
  init_from_params (everything else delegates).

  Attributes:
    inits: init calls so far.
  """

  def __init__(self, inner):
    self._inner = inner
    # init calls (a host learner's init may replace its live trainer)
    self.inits = 0

  def init(self, *args, **kwargs):
    """Counts the call and delegates.

    Args:
      *args: passed to the inner init.
      **kwargs: passed to the inner init.

    Returns:
      The inner learner's state.
    """
    self.inits += 1
    return self._inner.init(*args, **kwargs)

  def __getattr__(self, name):
    if name == "init_from_params":
      raise AttributeError(name)
    return getattr(self._inner, name)

  def state_dict(self, s):
    """The inner state_dict with params and opt_state renamed.

    Args:
      s: the learner state.

    Returns:
      The state dict with "weights" and "adam" entries.
    """
    d = dict(self._inner.state_dict(s))
    return dict(d, weights=d.pop("params"), adam=d.pop("opt_state"))

  def load_state(self, blob):
    """Loads a renamed state dict through the inner learner.

    Args:
      blob: the state dict with "weights" and "adam" entries.

    Returns:
      The learner state.
    """
    b = dict(blob)
    return self._inner.load_state(
        dict(b, params=b.pop("weights"), opt_state=b.pop("adam"))
    )


class _Script:
  """Iteration -> actions, each list emitted once (the first round).

  Attributes:
    script: iteration -> actions.
    done: iterations already emitted.
  """

  def __init__(self, script):
    self.script, self.done = dict(script), set()

  def step(self, view):
    """Returns the scripted actions of an iteration, once.

    Args:
      view: the league view.

    Returns:
      The iteration's actions; none if already returned.
    """
    if view.iteration in self.done:
      return []
    self.done.add(view.iteration)
    return list(self.script.get(view.iteration, []))


def _league(run_dir, script, opaque=True, **cfg):
  """A Kuhn league for the state-rewrite tests.

  Args:
    run_dir: The run folder.
    script: The controller script.
    opaque: Whether the learner state is opaque.
    **cfg: LeagueConfig overrides.

  Returns:
    The League.
  """
  pol = league_fixtures.kuhn_policy(16)
  learner = league_fixtures.br_learner(pol)
  if opaque:
    learner = _OpaqueLearner(learner)
  config = dataclasses.replace(
      loop.LeagueConfig(
          num_envs=32,
          num_steps=8,
          updates_per_iteration=1,
          pool_capacity=4,
          payoff="none",
          seed=1,
      ),
      **cfg,
  )
  store = artifact_store.ArtifactStore(os.path.join(run_dir, "store"))
  return loop.League(
      league_fixtures.env(),
      pol,
      learner,
      _Script(script),
      store,
      run_dir,
      config,
      initial_members={"init@0": league_fixtures.init_params(pol, 0)},
  )


_SPAWN = [
    controller.SpawnLearner("a", "main"),
    controller.SpawnLearner("b", "main"),
]
# _OpaqueLearner's state_dict.
_HAVE = "['adam', 'update_step', 'value_ema', 'weights']"


def _pinned(action, learner, capability, missing):
  """The exact refusal text, as a full-match regex.

  The exact refusal text (it names the learner and the missing
  capability), as a full-match regex.

  Args:
    action: the refused action.
    learner: the learner slot's name.
    capability: the missing capability.
    missing: the missing state_dict entry.

  Returns:
    The regex.
  """
  return (
      "^"
      + re.escape(
          f"{action}: refused: learner '{learner}' (_OpaqueLearner) lacks the '"
          f"{capability}' "
          f"capability: its state_dict has no {missing} (it has {_HAVE}); the "
          "league rewrites "
          "learner states only through those entries"
      )
      + "$"
  )


def test_copy_mutate_is_refused_before_anything_changes(tmp_path):
  """Copy mutate is refused before anything changes."""
  lg = _league(
      str(tmp_path / "cm"),
      {0: _SPAWN, 1: [controller.CopyMutate("a", "b", shrink=0.5)]},
  )
  lg.run(1)
  before = lg.learners["b"].state
  with pytest.raises(
      ValueError,
      match=_pinned("CopyMutate(a -> b)", "b", "perturb", "['params']"),
  ):
    lg.run(1)
  # Reaches private _mutations of the code under test.
  assert lg.learners["b"].state is before and lg._mutations == 0  # pylint: disable=protected-access


def test_spawn_from_a_member_is_refused(tmp_path):
  """Spawn from a member is refused."""
  lg = _league(
      str(tmp_path / "sp"),
      {0: [controller.SpawnLearner("a", "main", init_from="init@0")]},
  )
  with pytest.raises(
      ValueError,
      match=_pinned(
          "SpawnLearner(a, init_from=init@0)",
          "a",
          "replace_params",
          "['params']",
      ),
  ):
    lg.run(1)
  assert "a" not in lg.learners


def test_optimizer_reset_is_refused_at_spawn(tmp_path):
  """Optimizer reset is refused at spawn."""
  lg = _league(
      str(tmp_path / "rs"),
      {0: [controller.SpawnLearner("a", "main")]},
      reset_optimizer_each_iteration=True,
  )
  with pytest.raises(
      ValueError,
      match=_pinned(
          "reset_optimizer_each_iteration",
          "a",
          "reset_opt",
          "['opt_state', 'params']",
      ),
  ):
    lg.run(1)
  assert "a" not in lg.learners


def test_the_plain_pg_learner_takes_every_rewrite(tmp_path):
  """The plain pg learner takes every rewrite."""
  lg = _league(
      str(tmp_path / "ok"),
      {
          0: _SPAWN + [
              controller.SpawnLearner("c", "main", init_from="init@0")
          ],
          1: [controller.CopyMutate("a", "b", shrink=0.5)],
      },
      opaque=False,
      reset_optimizer_each_iteration=True,
  )
  lg.run(2)
  # Reaches private _mutations of the code under test.
  assert sorted(lg.learners) == ["a", "b", "c"] and lg._mutations == 1  # pylint: disable=protected-access


def test_spawn_from_a_member_is_refused_before_the_learner_is_initialised(
    tmp_path,
):
  """A refused SpawnLearner(init_from) calls no learner init.

  With a learner state in the league, a refused SpawnLearner(init_from)
  calls no learner init (a host learner's init may replace its live trainer,
  so a refusal after init_state has already changed something).

  Args:
    tmp_path: pytest's tmp_path.
  """
  lg = _league(
      str(tmp_path / "sp2"),
      {
          0: [controller.SpawnLearner("a", "main")],
          1: [controller.SpawnLearner("b", "main", init_from="init@0")],
      },
  )
  lg.run(1)
  inits = lg.learner.inits
  assert inits >= 1
  with pytest.raises(
      ValueError,
      match=_pinned(
          "SpawnLearner(b, init_from=init@0)",
          "b",
          "replace_params",
          "['params']",
      ),
  ):
    lg.run(1)
  assert lg.learner.inits == inits and "b" not in lg.learners


def test_a_plain_copy_is_refused_as_replace_params(tmp_path):
  """A plain copy is refused as replace params."""
  lg = _league(
      str(tmp_path / "cp"), {0: _SPAWN, 1: [controller.CopyMutate("a", "b")]}
  )
  lg.run(1)
  with pytest.raises(
      ValueError,
      match=_pinned("CopyMutate(a -> b)", "b", "replace_params", "['params']"),
  ):
    lg.run(1)


def test_refusal_message_text_is_pinned():
  """The literal message, character for character."""
  assert loop.refusal_message(
      "CopyMutate(a -> b)",
      "b",
      "TorchLearner",
      "perturb",
      ["params"],
      ["ema", "net", "optim", "update_step"],
  ) == (
      "CopyMutate(a -> b): refused: learner 'b' (TorchLearner) lacks the "
      "'perturb' "
      "capability: its state_dict has no ['params'] (it has ['ema', 'net', "
      "'optim', "
      "'update_step']); the league rewrites learner states only through those "
      "entries"
  )
  with pytest.raises(AssertionError):
    loop.refusal_message("x", "b", "L", "teleport", ["params"], [])


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
