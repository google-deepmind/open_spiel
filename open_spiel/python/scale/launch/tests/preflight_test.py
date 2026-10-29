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

"""Tests of the game package preflight.

The game package preflight: a run whose game needs a game package the
launch does not ship is refused before any machine exists (the GCP backend's
launch.run, a dry run included, and the cloud launcher); a game that needs
none passes; the built-in GAMES_PACKAGE_PATH names the games package folder.
Nothing touches GCP.
"""

import glob
import os
import sys

import pytest
import yaml

from open_spiel.python.scale import deps
from open_spiel.python.scale.launch import cloud
from open_spiel.python.scale.launch import config as config_lib
from open_spiel.python.scale.launch import preflight
from open_spiel.python.scale.launch import run as run_cli

LAUNCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = sorted(glob.glob(os.path.join(LAUNCH, "examples", "*.yaml")))
ENV = {
    "PROJECT": "my-project-123",
    "BUCKET": "my-bucket",
    "SERVICE_ACCOUNT": "vm@my-project-123.iam.gserviceaccount.com",
}
# The framework name preflight checks against (private to preflight).
FRAMEWORK = preflight._FRAMEWORK  # pylint: disable=protected-access


@pytest.fixture(name="env")
def _env_fixture(monkeypatch):
  """Sets the examples' environment variables.

  Args:
    monkeypatch: pytest's monkeypatch.
  """
  for k, v in ENV.items():
    monkeypatch.setenv(k, v)


def test_shipping_a_parent_folder_ships_the_games_package():
  """Shipping a parent folder ships the games package."""
  games = deps.GAMES_PACKAGE
  assert preflight.unshipped((FRAMEWORK,)) == (games,)
  assert not preflight.unshipped((FRAMEWORK, games))
  parent = games.rpartition(".")[0]
  if parent:  # a nested import name: shipping its parent folder ships it
    assert not preflight.unshipped((parent,))
  assert preflight.unshipped((games + "_other",)) == (games,)


def test_everything_shipped_needs_no_child_process(monkeypatch):
  """When every game package ships, no child process runs."""

  def boom(*a, **k):
    raise AssertionError("no child process when every game package ships")

  monkeypatch.setattr(preflight.subprocess_env, "run_python", boom)
  assert (
      preflight.check_game_packages(
          ["--config=ppo_tic_tac_toe"], (FRAMEWORK, deps.GAMES_PACKAGE)
      )
      is None
  )


def test_tic_tac_toe_without_the_games_package_is_refused():
  """Tic-tac-toe without the games package is refused."""
  with pytest.raises(
      preflight.GamePackageNotShippedError, match=r"games package"
  ) as e:
    preflight.check_game_packages(
        ["--config=ppo_tic_tac_toe", "--seed=1"], (FRAMEWORK,), fix="ADD IT"
    )
  assert "tic_tac_toe" in str(e.value) and str(e.value).endswith("ADD IT")


def test_the_built_in_game_needs_no_game_package():
  """The built-in game needs no game package."""
  assert (
      preflight.check_game_packages(
          ["--config", "mmd_kuhn_poker_jax"], (FRAMEWORK,)
      )
      == "kuhn_poker_jax"
  )


def test_no_config_is_refused():
  """Trainer flags without --config are refused."""
  with pytest.raises(preflight.GamePackageNotShippedError, match="--config"):
    preflight.check_game_packages(["--seed=1"], (FRAMEWORK,))


def test_an_unknown_config_fails_loud():
  """An unknown config fails loud."""
  with pytest.raises(RuntimeError, match="does not resolve"):
    preflight.check_game_packages(["--config=no_such_config_xyz"], (FRAMEWORK,))


def test_games_package_path_is_built_in():
  """GAMES_PACKAGE_PATH is a built-in name."""
  b = config_lib.builtins()
  rel = b["GAMES_PACKAGE_PATH"]
  assert rel == deps.GAMES_PACKAGE.replace(".", "/")
  assert os.path.isdir(os.path.join(b["PACKAGE_ROOT"], rel))


def test_games_package_path_raises_where_used_when_absent(
    monkeypatch, tmp_path
):
  """GAMES_PACKAGE_PATH raises where it is used when the package is absent."""
  monkeypatch.setattr(deps, "GAMES_PACKAGE", "no_such_games_package_xyz")
  assert isinstance(
      config_lib.builtins()["GAMES_PACKAGE_PATH"], config_lib.BackendFileError
  )
  p = tmp_path / "b.yaml"
  p.write_text("backend: local\nroot: ${GAMES_PACKAGE_PATH}\npython: python3\n")
  with pytest.raises(
      config_lib.BackendFileError,
      match=r"root: \$\{GAMES_PACKAGE_PATH\}: .*not installed",
  ):
    config_lib.load(str(p))


def _without_games(path, tmp_path):
  """The example config at path with only the framework in code.paths.

  Args:
    path: The example config.
    tmp_path: The folder to write to.

  Returns:
    The new config's path.
  """
  with open(path, encoding="utf-8") as f:
    raw = yaml.safe_load(f)
  raw["code"]["paths"] = ["${PACKAGE_PATH}"]
  out = tmp_path / os.path.basename(path)
  out.write_text(yaml.safe_dump(raw))
  return str(out)


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
@pytest.mark.parametrize("dry", [True, False], ids=["dry_run", "launch"])
def test_run_refuses_a_config_that_does_not_ship_the_games_package(
    path, dry, env, tmp_path, monkeypatch
):
  """launch.run refuses a config that does not ship the games package."""
  del env

  def no_supervisor(*a, **k):
    raise AssertionError("the supervisor must not start")

  monkeypatch.setattr(run_cli.supervisor_lib, "Supervisor", no_supervisor)
  cfg = _without_games(path, tmp_path)
  argv = [
      "prog",
      f"--backend_config={cfg}",
      "--run_id=ttt-a",
      *(["--dry_run"] if dry else []),
      "--",
      "--config=ppo_tic_tac_toe",
  ]
  with pytest.raises(
      SystemExit,
      match=r"preflight: .*games package.*GAMES_PACKAGE_PATH.*"
      r"nothing created",
  ):
    run_cli.main(argv)


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_the_examples_ship_the_games_package_and_pass(path, env, capsys):
  """The examples ship the games package and pass."""
  del env
  rc = run_cli.main([
      "prog",
      f"--backend_config={path}",
      "--run_id=ttt-a",
      "--dry_run",
      "--",
      "--config=ppo_tic_tac_toe",
  ])
  out = capsys.readouterr().out
  assert rc == 0 and "nothing created" in out
  shipped = [
      l for l in out.splitlines() if l.startswith("# code shipped from ")
  ]
  assert len(shipped) == 1, out[:2000]
  assert (
      deps.GAMES_PACKAGE.replace(".", "/")
      in shipped[0].split(": ", 1)[1].split()
  )


def test_the_cloud_launcher_refuses_when_it_cannot_ship_the_games_package(
    monkeypatch,
):
  """The cloud launcher refuses when it cannot ship the games package."""
  # not importable here
  monkeypatch.setattr(cloud, "GAMES", "no_such_games_package_xyz")
  assert cloud.shipped_packages() == (cloud.FRAMEWORK,)
  with pytest.raises(
      cloud.LaunchError, match="games package.*nothing was created"
  ):
    cloud.check_shipped(["--config=ppo_tic_tac_toe"], ())
  cloud.check_shipped(["--config=mmd_kuhn_poker_jax"], ())


def test_the_cloud_launcher_ships_the_games_package_when_importable(
    monkeypatch,
):
  """The cloud launcher ships the games package when it is importable."""

  def boom(*a, **k):
    raise AssertionError("no child process when the games package ships")

  monkeypatch.setattr(preflight.subprocess_env, "run_python", boom)
  assert deps.GAMES_PACKAGE in cloud.shipped_packages()
  cloud.check_shipped(["--config=ppo_tic_tac_toe"], ())


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
