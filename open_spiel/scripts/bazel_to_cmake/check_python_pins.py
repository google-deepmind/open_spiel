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

"""Checks that the Python version pins of the CMake/pip and Bazel builds agree.

Pins live in three places:
  * scripts/python_extra_deps.sh: the CMake and pip CI pins, per Python
    version;
  * bazel/python/requirements.in: the Bazel build's requirements;
  * bazel/python/requirements_lock.txt: generated from requirements.in by
    bazel/python/update_requirements_lock.sh.

This fails if a package that python_extra_deps.sh pins for the Python version
of the Bazel build (python.MODULE.bazel) is not pinned to the same version in
requirements.in, or if requirements.in names a package the lock lacks or pins
it to another version than the lock. Needs only the standard library (and
bash, to evaluate python_extra_deps.sh):

  python3 open_spiel/scripts/bazel_to_cmake/check_python_pins.py
"""

import argparse
import dataclasses
import os
import re
import stat
import subprocess
import sys
import tempfile
from typing import Optional, Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The OpenSpiel directory (the one holding spiel.h).
_OPEN_SPIEL_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?(.*)$")
_MODULE_VERSION_RE = re.compile(r'^_PYTHON_VERSION = "([0-9.]+)"$', re.M)
_UPDATE_SCRIPT_VERSION_RE = re.compile(
    r'^readonly PYTHON_VERSION="([0-9.]+)"$', re.M)
_EXTRA_DEPS_PREFIX = "OPEN_SPIEL_PYTHON_"
_DEPS_VARIABLE_RE = re.compile(r"^OPEN_SPIEL_PYTHON_\w+_DEPS$")

# Prints every exported OPEN_SPIEL_PYTHON_* variable after sourcing
# python_extra_deps.sh ($0) for the Python interpreter $1.
_DUMP_EXTRA_DEPS = (
    'source "$0" "$1" > /dev/null; '
    'for v in ${!OPEN_SPIEL_PYTHON_@}; do printf "%s=%s\\n" "$v" "${!v}"; done')


@dataclasses.dataclass(frozen=True)
class PinFiles:
  """Paths of the files holding Python pins."""

  extra_deps: str
  requirements_in: str
  lock: str
  module: str
  update_script: str

  @classmethod
  def find(cls, open_spiel_dir: str) -> "PinFiles":
    """Returns the pin files of the tree holding `open_spiel_dir`.

    Args:
      open_spiel_dir: The OpenSpiel directory. bazel/python is next to it in
        the exported repository, and in opensource/ below it in the source
        repository.
    """
    bazel_python = os.path.join(open_spiel_dir, "opensource", "bazel", "python")
    if not os.path.isdir(bazel_python):
      bazel_python = os.path.join(
          os.path.dirname(open_spiel_dir), "bazel", "python")
    return cls(
        extra_deps=os.path.join(open_spiel_dir, "scripts",
                                "python_extra_deps.sh"),
        requirements_in=os.path.join(bazel_python, "requirements.in"),
        lock=os.path.join(bazel_python, "requirements_lock.txt"),
        module=os.path.join(bazel_python, "python.MODULE.bazel"),
        update_script=os.path.join(bazel_python,
                                   "update_requirements_lock.sh"),
    )


def normalize(name: str) -> str:
  """Returns the normalized form of a pip distribution name, with `_`."""
  return re.sub(r"[-_.]+", "_", name).lower()


def _read(path: str) -> str:
  with open(path, encoding="utf-8") as f:
    return f.read()


def parse_requirements(text: str) -> dict[str, str]:
  """Returns {normalized name: version specifier} of a requirements file.

  Comments, options (`-r`, `--hash`) and environment markers are dropped, and
  whitespace is removed from the specifier: `numpy >= 1.2` gives `>=1.2`.

  Args:
    text: Contents of a requirements file or lock file.
  """
  requirements = {}
  for line in text.splitlines():
    line = line.split("#")[0].strip().rstrip("\\").strip()
    if not line or line.startswith("-"):
      continue
    match = _REQUIREMENT_RE.match(line)
    if match:
      specifier = match.group(3).split(";")[0]
      requirements[normalize(match.group(1))] = "".join(specifier.split())
  return requirements


def bazel_python_version(files: PinFiles) -> str:
  """Returns the Python version of the Bazel build.

  Args:
    files: The pin files.

  Raises:
    ValueError: If python.MODULE.bazel and update_requirements_lock.sh do
      not name the same version.
  """
  versions = []
  for path, regex in ((files.module, _MODULE_VERSION_RE),
                      (files.update_script, _UPDATE_SCRIPT_VERSION_RE)):
    match = regex.search(_read(path))
    if not match:
      raise ValueError(f"{path}: cannot find the Python version")
    versions.append(match.group(1))
  if versions[0] != versions[1]:
    raise ValueError(
        f"{files.module} uses Python {versions[0]} but {files.update_script} "
        f"locks for Python {versions[1]}")
  return versions[0]


def extra_deps(extra_deps_script: str, python_version: str) -> dict[str, str]:
  """Evaluates python_extra_deps.sh for a Python version.

  Args:
    extra_deps_script: Path of python_extra_deps.sh.
    python_version: e.g. "3.12".

  Returns:
    {OPEN_SPIEL_PYTHON_<GROUP>_DEPS: value}, as the script exports them.
  """
  with tempfile.TemporaryDirectory() as tmp:
    fake_python = os.path.join(tmp, "python")
    with open(fake_python, "w") as f:
      f.write(f"#!/bin/sh\necho 'Python {python_version}.0'\n")
    os.chmod(fake_python, os.stat(fake_python).st_mode | stat.S_IXUSR)
    # Only the script may define OPEN_SPIEL_PYTHON_*: CI has often sourced
    # python_extra_deps.sh already, and a variable the script leaves unset
    # would otherwise keep its inherited value.
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(_EXTRA_DEPS_PREFIX)
    }
    output = subprocess.run(
        ["bash", "-c", _DUMP_EXTRA_DEPS, extra_deps_script, fake_python],
        check=True, capture_output=True, text=True, env=env).stdout
  values = {}
  for line in output.splitlines():
    name, _, value = line.partition("=")
    if _DEPS_VARIABLE_RE.match(name):
      values[name] = value
  return values


def extra_deps_pins(values: dict[str, str]) -> dict[str, tuple[str, str]]:
  """Returns {normalized name: (`==<version>`, variable)} of `values`.

  Args:
    values: What `extra_deps` returns.

  Raises:
    ValueError: If an entry is not of the form `<name>==<version>`.
  """
  pins = {}
  for variable, value in sorted(values.items()):
    for pin in value.split():
      name, sep, version = pin.partition("==")
      if not sep or not name or not version:
        raise ValueError(f"{variable}: {pin!r} is not pinned with ==")
      pins[normalize(name)] = ("==" + version, variable)
  return pins


def problems(files: PinFiles) -> list[str]:
  """Returns one line per pin disagreement among `files` (empty if none)."""
  try:
    python_version = bazel_python_version(files)
    pins = extra_deps_pins(extra_deps(files.extra_deps, python_version))
  except (ValueError, subprocess.CalledProcessError) as e:
    return [str(e)]
  requirements_in = parse_requirements(_read(files.requirements_in))
  lock = parse_requirements(_read(files.lock))
  found = []
  for name, (pin, variable) in sorted(pins.items()):
    if requirements_in.get(name) != pin:
      found.append(
          f"{name}: {variable} in python_extra_deps.sh pins {pin} for Python "
          f"{python_version}, requirements.in has "
          f"{requirements_in.get(name) or 'no pin'}")
  for name, specifier in sorted(requirements_in.items()):
    if name not in lock:
      found.append(f"{name}: in requirements.in but not in "
                   "requirements_lock.txt")
    elif specifier.startswith("==") and lock[name] != specifier:
      found.append(f"{name}: requirements.in pins {specifier}, "
                   f"requirements_lock.txt has {lock[name]}")
  return found


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Prints the pin disagreements; returns 1 if there are any."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--open_spiel_dir", default=_OPEN_SPIEL_DIR,
      help="The OpenSpiel directory (default: the one holding this script).")
  options = parser.parse_args(argv)
  found = problems(PinFiles.find(os.path.abspath(options.open_spiel_dir)))
  for problem in found:
    print(f"error: {problem}", file=sys.stderr)
  if found:
    print("Python pins disagree: keep scripts/python_extra_deps.sh and "
          "bazel/python/requirements.in in sync, then run "
          "bazel/python/update_requirements_lock.sh.", file=sys.stderr)
    return 1
  print("Python pins agree.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
