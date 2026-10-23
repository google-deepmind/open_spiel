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

"""Template of the generated main of OpenSpiel `pytest_test`s.

`pytest.bzl` writes a copy of this file for every `pytest_test`, with
`%test_files%` and `%main_file%` filled in, and runs it with the `args` of the
target. It runs pytest on the test files, from the runfiles directory of the
main repository (the working directory of a test), so that test ids (and
`--deselect` prefixes) are paths like `open_spiel/python/foo_test.py::test_x`.
Collecting no test fails the test, like any other pytest error.
"""

import sys

import pytest

# Test files (or directories), relative to the runfiles directory of the main
# repository, separated by spaces.
_TEST_FILES = "%test_files%"

# pytest options given before the target's `args`, which can override them:
# * import test files by path, without changing sys.path (their imports
#   resolve like those of any other test);
# * do not write a .pytest_cache directory into the runfiles.
_DEFAULT_ARGS = (
    "--rootdir=.",
    "--import-mode=importlib",
    "-p",
    "no:cacheprovider",
)


def main() -> None:
  """Runs pytest on the test files; exits with its status."""
  # Do not write rewritten test modules next to the test files.
  sys.dont_write_bytecode = True
  status = pytest.main(
      [*_DEFAULT_ARGS, *sys.argv[1:], *_TEST_FILES.split()])
  if status == pytest.ExitCode.NO_TESTS_COLLECTED:
    sys.stderr.write(f"No tests were collected from {_TEST_FILES}.\n")
  sys.exit(int(status))


if __name__ == "__main__":
  main()
