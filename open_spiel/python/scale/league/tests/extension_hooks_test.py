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

"""The league's public extension points.

The league's public extension points: policy formats, population readers,
checkpoint layout readers and controllers.

Each hook is exercised with a toy registration and its refusals: a
PolicyRef of a registered format builds through its loader and records the
format's id and lineage; a population reader is chosen by detect; a layout
reader locates a checkpoint train_state.json does not commit by ref; a
state migration brings a saved state to the current version; a controller
factory receives its ControllerConfig.
"""

import dataclasses
import json
import os
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.league import commit
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import population
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league.tests import league_fixtures


# -- policy formats ------------------------------------------------------------
@pytest.fixture(name="toy_format")
def _toy_format_fixture(monkeypatch):
  """Registers a toy policy format for the test.

  Args:
    monkeypatch: The pytest fixture.

  Returns:
    The toy policy.
  """
  # Reaches private _FORMATS of the code under test.
  monkeypatch.setattr(refs, "_FORMATS", dict(refs._FORMATS))  # pylint: disable=protected-access
  pol = league_fixtures.kuhn_policy(8)

  def load(uri):
    """Builds the toy policy and its params.

    Args:
      uri: the params seed, as a string.

    Returns:
      (policy, params).
    """
    return pol, pol.init_params(jax.random.key(int(uri)))

  refs.register_policy_format("toy", load, ref_id=lambda uri: f"toy-{uri}")
  return pol


def test_a_registered_format_builds_through_its_loader(toy_format):
  """A registered format builds through its loader."""
  ref = refs.PolicyRef(format="toy", uri="3")
  pol, params = ref.build()
  assert pol is toy_format and league_fixtures.leaves_equal(
      params, toy_format.init_params(jax.random.key(3))
  )
  assert ref.id == "toy-3"
  assert ref.lineage() == {
      "parents": ["toy-3"],
      "lineage": {"source": "toy-3", "parent_run_id": None, "step": None},
  }


def test_policy_ref_validation_and_unknown_formats(toy_format):
  """Policy ref validation and unknown formats."""
  del toy_format
  with pytest.raises(ValueError, match="exactly one"):
    refs.PolicyRef(arch_name="uniform", format="toy", uri="1")
  with pytest.raises(ValueError, match="exactly one"):
    refs.PolicyRef()
  with pytest.raises(ValueError, match="go together"):
    refs.PolicyRef(format="toy")
  with pytest.raises(KeyError, match="no policy format 'nope'"):
    refs.PolicyRef(format="nope", uri="x").build()
  with pytest.raises(ValueError, match="already registered"):
    refs.register_policy_format("toy", lambda uri: None)


# -- population readers --------------------------------------------------------
def _artifact(name):
  """A params-free uniform Kuhn artifact named `name`.

  Args:
    name: The path and name.

  Returns:
    The Artifact.
  """
  man = types.Manifest(
      schema_version=2,
      game="kuhn_poker",
      rules_sha="r",
      obs_format="o",
      arch_name="uniform",
      arch_kwargs={},
      framework="jax",
      capability=types.Capability.NATIVE_JAX,
      default_rule=types.SAMPLE,
      params_sha256="s",
      step=0,
      parents=(),
  )
  return types.Artifact(manifest=man, params={}, policy=None, path=name)


class _MarkerReader:
  """A run reader that claims run directories holding a MARKER file."""

  name = "marker"

  def detect(self, run_dir):
    """Whether `run_dir` is a marker run.

    Args:
      run_dir: the run directory.

    Returns:
      True when run_dir holds a MARKER file.
    """
    return os.path.exists(os.path.join(run_dir, "MARKER"))

  def final_step(self, run_dir):
    """The final step of a marker run.

    Args:
      run_dir: the run directory (unused).

    Returns:
      Always 7.
    """
    del run_dir
    return 7

  def load(self, run_dir, step):
    """A fixed two-member population for a marker run.

    Args:
      run_dir: the run directory.
      step: the step read.

    Returns:
      The population.Population.
    """
    return population.Population(
        f"{run_dir}@{step}",
        "kuhn_poker",
        (_artifact("a"), _artifact("b")),
        ("a", "b"),
        np.asarray([0.25, 0.75]),
        self.name,
    )


def test_population_readers_are_chosen_by_detect(tmp_path, monkeypatch):
  """Population readers are chosen by detect."""
  # Reaches private _READERS of the code under test.
  monkeypatch.setattr(population, "_READERS", list(population._READERS))  # pylint: disable=protected-access
  with pytest.raises(ValueError, match="no population reader"):
    population.final_step(str(tmp_path))
  population.register_reader(_MarkerReader())
  with pytest.raises(ValueError, match="no population reader"):
    population.load_population(str(tmp_path), 7)
  (tmp_path / "MARKER").write_text("")
  assert population.final_step(str(tmp_path)) == 7
  pop = population.load_population(str(tmp_path), 7)
  assert pop.source == "marker" and pop.member_names == ("a", "b")
  np.testing.assert_allclose(pop.weights, [0.25, 0.75])


# -- layout readers ------------------------------------------------------------
def test_a_layout_reader_locates_checkpoints_without_commit_refs(
    tmp_path, monkeypatch
):
  """A layout reader locates checkpoints without commit refs."""
  # Reaches private _LAYOUT_READERS of the code under test.
  monkeypatch.setattr(commit, "_LAYOUT_READERS", list(commit._LAYOUT_READERS))  # pylint: disable=protected-access
  d = str(tmp_path)
  (tmp_path / commit.TRAIN_STATE).write_text(json.dumps({"step": 4}))
  with pytest.raises(ValueError, match="no league_commits"):
    commit.locate(d, 4)

  def detect(ckpt_dir):
    """Whether `ckpt_dir` uses the flat layout.

    Args:
      ckpt_dir: the checkpoint directory.

    Returns:
      True when ckpt_dir holds a "flat" entry.
    """
    return os.path.exists(os.path.join(ckpt_dir, "flat"))

  def restore(ckpt_dir):
    """Locates the flat layout's checkpoints.

    Args:
      ckpt_dir: the checkpoint directory.

    Returns:
      The layout record: a label and the checkpoints by step.
    """
    return {
        "label": "flat layout",
        "checkpoints": {
            4: {
                "state_path": os.path.join(ckpt_dir, "flat", "4.pkl"),
                "learners_root": os.path.join(ckpt_dir, "flat_learners"),
            }
        },
    }

  commit.register_layout_reader(detect, restore)
  os.makedirs(os.path.join(d, "flat"))
  loc = commit.locate(d, 4)
  assert str(loc.state_path) == os.path.join(d, "flat", "4.pkl")
  assert (
      loc.learners_root == os.path.join(d, "flat_learners")
      and loc.label == "flat layout"
  )
  assert commit.learner_dir(d, "main") == os.path.join(
      d, "flat_learners", "main"
  )
  with pytest.raises(FileNotFoundError, match="step 5"):
    commit.locate(d, 5)
  r = commit.ref(4, "a1")
  # an explicit ref wins
  assert commit.locate(d, 4, r).state_path == commit.state_path(d, 4, r)


# -- state migrations ----------------------------------------------------------
def test_state_migrations_bring_a_saved_state_to_the_current_version(
    monkeypatch,
):
  """State migrations bring a saved state to the current version."""
  monkeypatch.setattr(loop, "_STATE_TRANSLATIONS", [])
  now = {"state_version": loop.STATE_VERSION, "iteration": 3}
  assert loop.migrate_state(dict(now), "x") == now
  with pytest.raises(ValueError, match="state version None"):
    loop.migrate_state({"iteration": 3}, "x")

  def unversioned(blob):
    """Stamps an unversioned state with the current version.

    Args:
      blob: the saved state.

    Returns:
      The state with state_version set.
    """
    return dict(
        blob, state_version=blob.get("state_version", loop.STATE_VERSION)
    )

  loop.register_state_translation(unversioned)
  assert loop.migrate_state({"iteration": 3}, "x") == now
  with pytest.raises(ValueError, match="state version 99"):
    loop.migrate_state({"iteration": 3, "state_version": 99}, "x")
  with pytest.raises(TypeError):
    loop.register_state_translation("not callable")


# -- controllers ---------------------------------------------------------------
def test_controller_factories_receive_the_run_config(monkeypatch):
  """Controller factories receive the run config."""
  # Reaches private _CONTROLLERS of the code under test.
  monkeypatch.setattr(runner, "_CONTROLLERS", dict(runner._CONTROLLERS))  # pylint: disable=protected-access
  seen = []

  def make(c):
    """Records the run config and builds a fixed-mix controller.

    Args:
      c: the controller config.

    Returns:
      The controller.
    """
    seen.append(c)
    # Reaches private _CONTROLLERS of the code under test.
    return runner._CONTROLLERS["fixed_mix"].factory(  # pylint: disable=protected-access
        dataclasses.replace(c, kwargs={"learner": "main"})
    )

  runner.register_controller("toy_mix", make, single_learner="learner")
  assert runner.single_learner_key("toy_mix") == "learner"
  learner = league_fixtures.br_learner(league_fixtures.kuhn_policy(8)).cfg
  cfg = runner.one_slot(
      "toy",
      "kuhn_poker",
      kinds.PG(learner),
      num_envs=8,
      num_steps=4,
      num_updates=1,
      policy_kwargs=league_fixtures.kuhn_policy_kwargs(8),
  )
  cfg = dataclasses.replace(
      cfg, controller="toy_mix", controller_kwargs={"x": 1}
  )
  ctl = cfg.build_controller()
  assert (
      seen[0].name == "toy_mix"
      and seen[0].kwargs == {"x": 1}
      and seen[0].run is cfg
  )
  assert ctl.learner == "main"
  with pytest.raises(ValueError, match="already registered"):
    runner.register_controller("toy_mix", lambda c: None)
  with pytest.raises(KeyError, match="unknown controller"):
    dataclasses.replace(cfg, controller="nope")


def test_every_built_in_controller_is_registered():
  """Every built in controller is registered."""
  assert set(runner.controller_names()) >= {
      "psro",
      "fixed_mix",
      "scheduled_promotion",
      "gated_pool",
      "continual_exploiter",
      "br_oracle",
      "pbt",
      "lagged_audit",
  }
  assert {
      n for n in runner.controller_names() if runner.single_learner_key(n)
  } >= {"fixed_mix", "scheduled_promotion", "gated_pool", "lagged_audit"}


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
