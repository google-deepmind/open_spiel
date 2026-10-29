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

"""Template of the generated main of OpenSpiel `py_test`s.

Bazel passes a test whose process exits with status 0, so a test file that
defines tests but never starts a test runner (no `absltest.main()`, e.g. a
pytest-style file of bare `test_*` functions) would pass without running
anything. `py_test.bzl` therefore runs every test file through a copy of this
file, with `%test_files%` and `%main_file%` filled in. The test file is run
exactly as Python runs a script; if it returns without having started a
unittest runner although it defines tests, the test fails.

Files that start a runner keep their behavior: absltest and unittest exit by
themselves (and, on Python 3.12+, fail when no test ran). Files that define no
tests, such as examples run as smoke tests, pass as before.
"""

import os
import runpy
import sys
import types
from typing import Any, Mapping
import unittest

# Paths of the test file and of this generated file, relative to the runfiles
# directory of the main repository.
_TEST_FILE = "%test_files%"
_MAIN_FILE = "%main_file%"

# Exit status of pytest and of unittest (Python 3.12+) when no test ran.
_NO_TESTS_RAN = 5
_MAX_LISTED_TESTS = 5


class _RunnerWatch:
  """Records whether a unittest runner started a test run."""

  def __init__(self) -> None:
    self.started = False

  def install(self) -> None:
    """Wraps `unittest.TestResult.startTestRun` to record test runs."""
    original = unittest.TestResult.startTestRun

    def start_test_run(result: unittest.TestResult) -> None:
      self.started = True
      original(result)

    unittest.TestResult.startTestRun = start_test_run


def _defined_tests(test_globals: Mapping[str, Any]) -> list[str]:
  """Returns the names of the tests defined by a test file.

  Args:
    test_globals: the globals of the test file, run as `__main__`.

  Returns:
    The `unittest.TestCase` test methods and module-level `test*` functions
    (pytest style) of the classes and functions the file defines itself.
  """
  loader = unittest.TestLoader()
  names = []
  for name, value in test_globals.items():
    if getattr(value, "__module__", None) != "__main__":
      continue
    if isinstance(value, type) and issubclass(value, unittest.TestCase):
      names.extend(
          f"{name}.{method}" for method in loader.getTestCaseNames(value))
    elif name.startswith("test") and isinstance(value, types.FunctionType):
      names.append(name)
  return names


def main() -> None:
  """Runs the test file; fails if it defines tests but ran none."""
  own_file = os.path.abspath(__file__)
  repo_runfiles_dir = own_file[:-len(_MAIN_FILE)]
  test_file = os.path.join(repo_runfiles_dir, _TEST_FILE)
  # As for `python <test_file>`: argv[0] and sys.path[0] name the test file.
  sys.argv[0] = test_file
  if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(own_file):
    sys.path[0] = os.path.dirname(test_file)

  watch = _RunnerWatch()
  watch.install()
  test_globals = runpy.run_path(test_file, run_name="__main__")
  if watch.started:
    return
  tests = _defined_tests(test_globals)
  if tests:
    listed = ", ".join(tests[:_MAX_LISTED_TESTS])
    if len(tests) > _MAX_LISTED_TESTS:
      listed += ", ..."
    sys.stderr.write(
        f"{_TEST_FILE} defines {len(tests)} test(s) ({listed}) but ran none. "
        "Call absltest.main() under `if __name__ == \"__main__\":`, or use "
        "pytest_test (bazel/python/pytest.bzl) for pytest-style tests.\n")
    sys.exit(_NO_TESTS_RAN)


if __name__ == "__main__":
  main()
