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

"""Regenerates OpenSpiel's CMake files from its BUILD files.

Pure Python with no dependencies beyond the standard library: Bazel is not
needed. Run it through scripts/generate_cmake.sh:

  scripts/generate_cmake.sh          # rewrite the generated files
  scripts/generate_cmake.sh --check  # fail if any is stale
"""

import argparse
import os
import sys
from typing import Optional, Sequence

from open_spiel.scripts.bazel_to_cmake import generator

_REGENERATE_HINT = "Run scripts/generate_cmake.sh to regenerate them."


def run(root: str, check: bool, freshness_inputs: bool = False) -> int:
  """Generates (or checks) every CMake file under `root`.

  Args:
    root: The OpenSpiel source directory.
    check: Only report stale files instead of writing them.
    freshness_inputs: Also generate the input list of the source repository's
      freshness test (only available there).

  Returns:
    The process exit code.
  """
  try:
    workspace = generator.Workspace(root)
    outputs = workspace.generate()
  except generator.GenerationError as e:
    print(f"error: {e}", file=sys.stderr)
    return 2
  for skipped in workspace.skipped:
    print(f"note: not built by CMake: {skipped}", file=sys.stderr)
  existing = generator.find_generated_files(root)
  if check:
    problems = generator.diff_outputs(root, outputs, existing)
    for problem in problems:
      print(problem, file=sys.stderr)
    if problems:
      print(f"Generated CMake files are out of date. {_REGENERATE_HINT}",
            file=sys.stderr)
      return 1
    return 0
  changed = {}
  for rel, contents in sorted(outputs.items()):
    path = os.path.join(root, rel)
    if not os.path.exists(path):
      changed[rel] = contents
      continue
    with open(path) as f:
      current = f.read()
    if current == contents:
      continue
    if not current.startswith(generator.HEADER):
      print(f"error: refusing to overwrite hand-written {rel}; delete it "
            "first if it should be generated", file=sys.stderr)
      return 2
    changed[rel] = contents
  for rel in existing:
    if rel not in outputs:
      os.remove(os.path.join(root, rel))
      print(f"removed {rel}")
  for rel, contents in changed.items():
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      f.write(contents)
    print(f"wrote {rel}")
  return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Parses the command line and runs the generator."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--root", required=True,
      help="The OpenSpiel source directory (the one holding spiel.h).")
  parser.add_argument(
      "--check", action="store_true",
      help="Do not write; exit non-zero if any generated file is stale.")
  options = vars(parser.parse_args(argv))
  return run(os.path.abspath(options.pop("root")), **options)


if __name__ == "__main__":
  sys.exit(main())
