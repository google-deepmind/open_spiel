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

"""pytest setup of the package's tests.

Markers are registered here so they hold on any machine (a copy of the
package folder alone has no repo-root pyproject.toml) and under
--strict-markers.

Optional packages (OPTIONAL: deps.GAMES_PACKAGE, the import name of the pure-JAX
OpenSpiel games package most game envs are built on; pyspiel; the tensorboard
and wandb sinks): a test module that fails to import because one of them is not
installed, or a test that raises that ModuleNotFoundError (also as the cause of
another error, or in the output of a subprocess it ran), is reported SKIPPED
with the package named, so `python -m pytest <package folder>` on a minimal
install runs everything that can run.
Only an absent package is skipped: when the package is installed, any import
error (a missing submodule included) fails as usual. A test that needs a
pyspiel game only an optional OpenSpiel dependency builds (universal_poker:
ACPC) loads it through testing/optional_pyspiel_games, which skips it on a
build without that game.

The test architectures of testing/ (fixtures, the host operator) are
registered for the whole session, and jax_enable_x64 is turned on (the
package's enable_x64) before any test module loads.
"""

import importlib.util
import os
import re
import sys

import pytest

from open_spiel.python.scale import registry
from open_spiel.python.scale.core import frameworks

# The package's tests run as its entry points do: x64 on before any test
# module loads jax (importing the package alone leaves it unchanged).
frameworks.enable_x64()


def _load_deps():
  """deps.py beside this file, loaded by path.

  the hooks below also run from a conftest loaded outside the package.

  Returns:
    the deps module.
  """
  spec = importlib.util.spec_from_file_location(
      "_package_deps",
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps.py"),
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


_DEPS = _load_deps()
OPTIONAL = _DEPS.OPTIONAL
_NO_MODULE = re.compile(r"No module named '([\w.]+)'")


def pytest_configure(config):
  """Registers the slow marker.

  Args:
    config: the pytest config.
  """
  config.addinivalue_line(
      "markers", "slow: a long test (end to end, subprocesses)"
  )


@pytest.fixture(scope="session", autouse=True)
def _test_architectures():
  """Registers the testing/ architectures for the session.

  A session that blocks jax (the jax-free control-plane tests) has none to
  register.

  Yields:
    None, with the architectures registered.
  """
  if "jax" in sys.modules and sys.modules["jax"] is None:
    yield
    return
  # Lazy: testing/ imports jax, which a jax-free session blocks (optional).
  from open_spiel.python.scale.testing import fixtures  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415
  from open_spiel.python.scale.testing import host_operator_fixtures  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  with fixtures.registered(), host_operator_fixtures.registered():
    yield


def absent_optional(name):
  """The OPTIONAL package module `name` is in (or holds) when not installed.

  An entry is a package's full import name (it may be nested).

  Args:
    name: a module name.

  Returns:
    the absent OPTIONAL entry, or None.
  """
  return registry.absent_package(name, OPTIONAL)


def _missing_from(exc):
  """The optional package whose absence caused `exc`, if any.

  Args:
    exc: The exception, followed through its causes.

  Returns:
    The package name, or None.
  """
  seen = set()
  while exc is not None and id(exc) not in seen:
    seen.add(id(exc))
    if isinstance(exc, ModuleNotFoundError):
      root = absent_optional(exc.name)
      if root:
        return root
    exc = exc.__cause__ or exc.__context__
  return None


def _skip(
    rep, path, line, root, why="which is not installed (an optional dependency)"
):
  rep.outcome = "skipped"
  rep.longrepr = (str(path), int(line or 0), f"Skipped: needs {root}, {why}")


# A pytest hook wrapper both yields (to the next hook) and returns (the
# report), so its docstring documents both.
# pylint: disable=g-doc-return-or-yield
@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
  """Reports an import failure for want of an OPTIONAL package as skipped.

  Args:
    collector: the collector.

  Returns:
    the report.

  Yields:
    to the next hook, which sends back the report.
  """
  rep = yield
  if rep.failed and isinstance(collector, pytest.Module):
    m = _NO_MODULE.search(str(rep.longrepr))
    root = absent_optional(m.group(1)) if m else None
    if root:
      _skip(rep, collector.path, 0, root)
  return rep


#: A child process's report of an absent package, in a failure's text (a test
#: that runs the trainer in a subprocess asserts on its exit code and output).
_CHILD_MISSING = re.compile(
    r"ModuleNotFoundError: (?:No module named '([\w.]+)'|game '[^']*' "
    r"needs ([\w.]+))"
)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
  """Reports a test that fails for want of an OPTIONAL package as skipped.

  Args:
    item: the test item.
    call: the call phase.

  Returns:
    the report.

  Yields:
    to the next hook, which sends back the report.
  """
  rep = yield
  if rep.failed and call.excinfo is not None:
    root = _missing_from(call.excinfo.value)
    if not root:
      for m in _CHILD_MISSING.finditer(str(rep.longrepr)):
        root = absent_optional(m.group(1) or m.group(2))
        if root:
          break
    if root:
      _skip(rep, item.path, item.location[1], root)
  return rep


# pylint: enable=g-doc-return-or-yield
