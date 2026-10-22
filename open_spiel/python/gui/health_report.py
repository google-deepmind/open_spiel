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


"""The server's health report: the code it runs and what it can do."""

import functools
import importlib.metadata
import pathlib
import subprocess

from open_spiel.python.gui import contract
from open_spiel.python.gui import controllers
from open_spiel.python.gui import gil_safety
from open_spiel.python.gui import health

# The version reported when OpenSpiel runs from a source tree, uninstalled.
UNKNOWN_VERSION = "unknown"

_SOURCE_DIR = pathlib.Path(__file__).resolve().parent

# Where this package sits in an OpenSpiel source tree.
_PACKAGE_PATH = pathlib.Path("open_spiel", "python", "gui")


def build() -> health.Health:
  """Returns the server's health report.

  The first call probes whether search can play games written in Python,
  which takes a few seconds; later calls reuse the answer.
  """
  return health.Health(
      version=_version(),
      revision=_own_revision(),
      capabilities=health.Capabilities(
          controllers=controllers.registered(),
          chance_policies=tuple(contract.ChancePolicy),
          python_game_search=gil_safety.python_games_are_safe(),
      ),
  )


@functools.cache
def _version() -> str:
  """Returns the installed OpenSpiel package's version."""
  try:
    return importlib.metadata.version("open_spiel")
  except importlib.metadata.PackageNotFoundError:
    return UNKNOWN_VERSION


@functools.cache
def _own_revision() -> str | None:
  """Returns the commit of the source tree this package runs from, if any."""
  return revision(_SOURCE_DIR)


def revision(source_dir: pathlib.Path) -> str | None:
  """Returns the commit of the OpenSpiel source tree the server runs from.

  Args:
    source_dir: This package's directory.

  Returns:
    The commit, or None unless `source_dir` is `open_spiel/python/gui` at
    the top of a git repository: an installed package inside some other
    repository reports nothing rather than that repository's commit.
  """
  try:
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "--show-toplevel", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
  except (OSError, subprocess.SubprocessError):
    return None
  lines = result.stdout.splitlines()
  if len(lines) != 2:
    return None
  toplevel, commit = lines
  if pathlib.Path(toplevel).resolve() / _PACKAGE_PATH != source_dir:
    return None
  return commit
