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

"""The one interpreter setting of child Python processes (subprocess_env)."""

import sys

import pytest

from open_spiel.python.scale.launch import backend as backend_lib
from open_spiel.python.scale.launch import local
from open_spiel.python.scale.launch import subprocess_env

_RUN = backend_lib.RunSpec("kuhn-a", train_args=("--config=c",))


@pytest.fixture(name="no_setting")
def _no_setting_fixture(monkeypatch):
  monkeypatch.setattr(subprocess_env, "PYTHON", None)
  monkeypatch.delenv(subprocess_env.PYTHON_ENV, raising=False)


@pytest.mark.usefixtures("no_setting")
def test_the_default_is_this_interpreter():
  """With no setting, the interpreter is sys.executable."""
  assert subprocess_env.python_command() == [sys.executable]
  assert subprocess_env.python_available()


@pytest.mark.usefixtures("no_setting")
def test_the_environment_variable_is_a_shell_quoted_command(monkeypatch):
  """PYTHON_ENV is split as a shell command."""
  monkeypatch.setenv(subprocess_env.PYTHON_ENV, "/opt/run 'a b' --py")
  assert subprocess_env.python_command() == ["/opt/run", "a b", "--py"]


@pytest.mark.usefixtures("no_setting")
def test_the_module_setting_comes_first(monkeypatch):
  """An explicit interpreter, then PYTHON, then PYTHON_ENV."""
  monkeypatch.setenv(subprocess_env.PYTHON_ENV, "/from/env")
  monkeypatch.setattr(subprocess_env, "PYTHON", ["/from/module", "-X"])
  assert subprocess_env.python_command() == ["/from/module", "-X"]
  assert subprocess_env.python_command("/explicit") == ["/explicit"]


@pytest.mark.usefixtures("no_setting")
def test_no_interpreter_skips_inside_a_test_and_raises_outside(monkeypatch):
  """No interpreter: a skip inside a test, NoInterpreterError outside."""
  monkeypatch.setattr(sys, "executable", None)
  assert not subprocess_env.python_available()
  with pytest.raises(pytest.skip.Exception, match=subprocess_env.PYTHON_ENV):
    subprocess_env.python_command()
  with pytest.raises(pytest.skip.Exception):
    subprocess_env.run_python(["-c", "pass"])
  monkeypatch.delenv("PYTEST_CURRENT_TEST")
  with pytest.raises(subprocess_env.NoInterpreterError, match="sys.executable"):
    subprocess_env.python_command()


@pytest.mark.usefixtures("no_setting")
def test_the_launchers_take_the_same_setting(monkeypatch, tmp_path):
  """The trainer argv and the local backend use the same setting."""
  monkeypatch.setenv(subprocess_env.PYTHON_ENV, "/opt/py -E")
  assert _RUN.trainer_argv("/r")[:3] == ["/opt/py", "-E", "-u"]
  be = local.LocalBackend(str(tmp_path))
  assert be.plan(_RUN, 0).startswith(
      f"/opt/py -E -u -m {backend_lib.TRAINER_MODULE} "
  )
  explicit = local.LocalBackend(str(tmp_path), python="/usr/bin/python3")
  assert explicit.plan(_RUN, 0).startswith("/usr/bin/python3 -u -m ")


def test_a_child_finds_the_package():
  """run_python's child imports the package."""
  r = subprocess_env.run_python(
      ["-c", f"import {subprocess_env.PACKAGE}; print('OK')"], timeout=120
  )
  assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
