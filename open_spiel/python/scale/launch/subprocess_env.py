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

"""The package's import name and root, and a child Python process that finds it.

A test that starts a Python subprocess must not assume where the package sits
or what it is called: the same code runs as a top-level package and nested
inside another one (`<root>/a/b/<name>/`). Everything here is derived from
this module's own `__name__` and `__file__`, never from a test module's,
because pytest may import a test file under a name derived from its path.

  PACKAGE       the package's dotted import name
  IMPORT_ROOT   the sys.path entry it is imported from
  module(rel)   PACKAGE + "." + rel, a module name to pass to `python -m`
  child_env()   os.environ with IMPORT_ROOT first on PYTHONPATH
  child_cwd()   an empty directory: a child's sys.path[0] that shadows nothing
  python_command()  the interpreter command every child Python process runs
  python_available()  whether there is one
  run_python()  subprocess.run([*python_command(), ...]) with both

The interpreter comes from one place, python_command(): the module setting
PYTHON (a command as a list, for a runner that substitutes its own launcher),
else the environment variable PYTHON_ENV (a shell-quoted command), else
sys.executable. When there is none (sys.executable is None or empty, as in a
self-contained test binary), python_command() skips the test with a reason
when called inside a pytest test (PYTEST_CURRENT_TEST is set) and raises
NoInterpreterError otherwise. A child inherits PYTHON_ENV, so a grandchild
uses the same interpreter.

The working directory matters: `python -m` and `python -c` put it first on
sys.path, so a working directory holding a folder named like an installed
package (a `jax/` beside the package, say) hides that package from the child.
"""

import functools
import os
import shlex
import subprocess
import sys
import tempfile
from typing import Mapping, Optional, Sequence, Union
import unittest

# launch/subprocess_env.py -> the package is two components up.
PACKAGE = __name__.rsplit(".", 2)[0]
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_root() -> str:
  """The folder `PACKAGE` imports from.

  Returns:
    The import root.

  Raises:
    RuntimeError: The package does not sit at its import path.
  """
  root = PACKAGE_DIR
  for _ in PACKAGE.split("."):
    root = os.path.dirname(root)
  expected = os.path.join(root, *PACKAGE.split("."))
  if os.path.realpath(expected) != os.path.realpath(PACKAGE_DIR):
    raise RuntimeError(
        f"package {PACKAGE!r} is at {PACKAGE_DIR}, which is not "
        f"{expected}: the import root cannot be derived"
    )
  return root


IMPORT_ROOT = _import_root()

# The interpreter of every child Python process (module doc): the module
# setting first, then the environment variable, then sys.executable.
PYTHON: Optional[Sequence[str]] = None
PYTHON_ENV = "OPEN_SPIEL_SCALE_PYTHON"


class NoInterpreterError(RuntimeError):
  """No Python interpreter to start a child process with."""


def _configured(python: Union[str, Sequence[str], None] = None) -> list[str]:
  """The interpreter command (python_command), or [] when there is none.

  Args:
    python: An interpreter command, overriding the configuration.

  Returns:
    The command, or [] when there is none.
  """
  if python is not None:
    cmd = [python] if isinstance(python, str) else list(python)
  elif PYTHON is not None:
    cmd = list(PYTHON)
  elif os.environ.get(PYTHON_ENV, "").strip():
    cmd = shlex.split(os.environ[PYTHON_ENV])
  else:
    cmd = [sys.executable] if sys.executable else []
  return cmd if cmd and all(cmd) else []


def python_available() -> bool:
  """Whether python_command() has an interpreter (it never skips or raises).

  Returns:
    True when an interpreter is configured or sys.executable is set.
  """
  return bool(_configured())


def python_command(python: Union[str, Sequence[str], None] = None) -> list[str]:
  """The command that runs a child Python process (module doc).

  Args:
    python: an explicit interpreter (a path, or a command as a list); None
      takes the configured one.

  Returns:
    The command as a list, arguments not included.

  Raises:
    NoInterpreterError: there is no interpreter, outside a pytest test (inside
      one the test is skipped instead).
    unittest.SkipTest: there is no interpreter, inside a test of a process
      that has not loaded pytest (pytest's skip exception when it has).
  """
  cmd = _configured(python)
  if cmd:
    return cmd
  reason = (
      "no Python interpreter to start a child process with "
      f"(sys.executable is {sys.executable!r} and {PYTHON_ENV} is "
      f"unset): set {PYTHON_ENV} or {__name__}.PYTHON to the command "
      "that runs Python"
  )
  if os.environ.get("PYTEST_CURRENT_TEST"):
    pytest = sys.modules.get("pytest")  # loaded: report the test's own line
    if pytest is not None:
      pytest.skip(reason)
    raise unittest.SkipTest(reason)
  raise NoInterpreterError(reason)


def module(rel: str) -> str:
  """Returns the full name of the package's module `rel`.

  Args:
    rel: The module relative to the package, e.g. "core.tests.probe".

  Returns:
    The full module name.

  Raises:
    ValueError: `rel` is empty or starts with a dot.
  """
  if not rel or rel.startswith("."):
    raise ValueError(f"a relative module name like 'core.arena', got {rel!r}")
  return f"{PACKAGE}.{rel}"


def child_env(
    extra: Mapping[str, str] | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
  """Returns `base` plus `extra`, with IMPORT_ROOT first on PYTHONPATH.

  The PYTHONPATH entries already present (e.g. where pyspiel is built) stay,
  after IMPORT_ROOT. A PYTHONPATH in `extra` is taken as the existing value.

  Args:
    extra: Variables to add.
    base: The environment; defaults to os.environ.

  Returns:
    The child's environment.
  """
  env = dict(os.environ if base is None else base)
  env.update(extra or {})
  rest = [
      p
      for p in env.get("PYTHONPATH", "").split(os.pathsep)
      if p and os.path.realpath(p) != os.path.realpath(IMPORT_ROOT)
  ]
  env["PYTHONPATH"] = os.pathsep.join([IMPORT_ROOT] + rest)
  return env


@functools.cache
def child_cwd() -> str:
  """Returns an empty directory for a child's working directory.

  It is made once per process (cached).

  Returns:
    The directory.
  """
  return tempfile.mkdtemp(prefix="child_cwd_")


def run_python(
    args: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    base_env: Mapping[str, str] | None = None,
    timeout: Optional[float] = 600,
    python: Union[str, Sequence[str], None] = None,
    cwd: Optional[str] = None,
    **kwargs,
) -> subprocess.CompletedProcess[str]:
  """Runs `python *args` in child_cwd() under child_env(env, base_env).

  The exit status is not checked: assert on the returned process.

  Args:
    args: the interpreter's arguments.
    env: extra environment (child_env).
    base_env: the environment it extends (default os.environ).
    timeout: seconds, or None for no limit.
    python: the interpreter (python_command); None takes the configured one.
    cwd: the working directory; None is child_cwd().
    **kwargs: passed to subprocess.run.

  Returns:
    The finished process.
  """
  return subprocess.run(
      [*python_command(python), *args],
      cwd=child_cwd() if cwd is None else cwd,
      env=child_env(env, base_env),
      capture_output=True,
      text=True,
      timeout=timeout,
      check=False,
      **kwargs,
  )
