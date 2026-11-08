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

"""The package without its optional game packages.

a game (or config) whose adapter needs one that is not installed raises
ModuleNotFoundError naming the package and the game that runs without any, and
the test suite reports such tests SKIPPED (conftest.py), never as a collection
error. The absent package is simulated (a name that is never installed), so this
runs anywhere.
"""

import os
import sys
import textwrap

import pytest

from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.league import config_common

FAKE = "never_installed_game_pkg_for_tests"


@pytest.fixture(name="fake_package")
def _fake_package_fixture(monkeypatch):
  """Registers a fake game that needs a missing package.

  Args:
    monkeypatch: The pytest fixture.

  Returns:
    The fake package name.
  """
  # Reaches private _GAMES of the code under test.
  # pylint: disable=protected-access
  monkeypatch.setitem(registry.GAME_PACKAGES, FAKE, f"{FAKE} (a test stand-in)")
  monkeypatch.setitem(
      registry._GAMES,
      # Reaches private members of the code under test.
      "fake_needs_pkg",  # pylint: disable=protected-access
      registry.GameSpec("fake_needs_pkg", f"{FAKE}.env:Env"),
  )
  return FAKE


def test_a_game_that_needs_an_absent_package_says_so(fake_package):
  """A game that needs an absent package says so."""
  with pytest.raises(
      ModuleNotFoundError, match="needs .* not installed.*kuhn_poker_jax"
  ) as e:
    registry.env_class("fake_needs_pkg")
  assert e.value.name == fake_package


def test_a_config_that_needs_an_absent_package_says_so(fake_package, tmp_path):
  """A config that needs an absent package says so."""
  cfg = tmp_path / "needs_pkg.py"
  cfg.write_text(
      f"import {fake_package}.configs\n\ndef get_config():\n  return None\n"
  )
  with pytest.raises(
      ModuleNotFoundError, match="config .* needs .*mmd_kuhn_poker_jax"
  ):
    config_common.load_config(str(cfg))


def test_an_unrelated_missing_module_is_not_disguised(tmp_path):
  """An unrelated missing module is not reported as an absent package."""
  cfg = tmp_path / "typo.py"
  cfg.write_text(
      "import no_such_module_for_tests\n\ndef get_config():\n  return None\n"
  )
  with pytest.raises(
      ModuleNotFoundError, match="No module named 'no_such_module_for_tests'"
  ):
    config_common.load_config(str(cfg))


def test_the_suite_skips_what_needs_an_absent_optional_package(tmp_path):
  """conftest.py's hooks, in a child pytest.

  a module that imports an absent OPTIONAL package and a test that does are
  SKIPPED; a missing module that is not optional still errors.

  Args:
    tmp_path: pytest's temporary directory.
  """
  conftest = os.path.join(subprocess_env.PACKAGE_DIR, "conftest.py")
  assert os.path.isfile(conftest), conftest
  # The child's conftest loads the package's by path and uses its hooks with
  # OPTIONAL set to the stand-in (the hooks read OPTIONAL when they run). Its
  # session fixture is not taken: the child has no architectures to register.
  (tmp_path / "conftest.py").write_text(textwrap.dedent(f"""
      import importlib.util

      _spec = importlib.util.spec_from_file_location(
          "package_conftest", {conftest!r})
      _pkg = importlib.util.module_from_spec(_spec)
      _spec.loader.exec_module(_pkg)
      assert _pkg.OPTIONAL, _pkg.OPTIONAL
      _pkg.OPTIONAL = ({FAKE!r},)
      pytest_configure = _pkg.pytest_configure
      pytest_make_collect_report = _pkg.pytest_make_collect_report
      pytest_runtest_makereport = _pkg.pytest_runtest_makereport
  """))
  (tmp_path / "test_mod.py").write_text(
      f"import {FAKE}\n\ndef test_x():\n  pass\n"
  )
  (tmp_path / "test_fn.py").write_text(textwrap.dedent(f"""
      def test_lazy():
        import {FAKE}.sub

      def test_ok():
        pass
  """))
  (tmp_path / "test_bad.py").write_text("import no_such_module_for_tests\n")
  r = subprocess_env.run_python(
      [
          "-m",
          "pytest",
          "-q",
          "-rs",
          "-p",
          "no:cacheprovider",
          "--rootdir",
          str(tmp_path),
          "--continue-on-collection-errors",
          str(tmp_path),
      ],
      timeout=300,
  )
  out = r.stdout + r.stderr
  assert "1 passed, 2 skipped, 1 error" in out, out
  assert out.count(f"needs {FAKE}") == 2, out


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
