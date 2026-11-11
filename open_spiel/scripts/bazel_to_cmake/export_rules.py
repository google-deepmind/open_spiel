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

"""What the GitHub export drops, mirrored so generation ignores the same.

The generator must produce byte-identical CMake files from the source BUILD
files and from the exported BUILD.bazel files. It therefore skips the packages
and files the export excludes, and applies the export's BUILD-file scrubs
before evaluating a BUILD file. A test checks these rules against the export
configuration.
"""

import dataclasses
import functools
import re
from typing import Optional


@dataclasses.dataclass(frozen=True)
class ExportRules:
  """What the export drops.

  Attributes:
    root: Path prefix of OpenSpiel files in the export configuration's globs.
    excludes: Globs of files the export does not copy (its `open_spiel_files`
      excludes, minus any about BUILD files themselves, which it renames).
    build_scrubs: Regexes whose matches the export deletes from BUILD files,
      in order.
    scrubbed_prefix: Label prefix that must not survive the scrubs: such
      labels outside a line of their own would leak into (and break) the
      export.
  """

  root: str = ""
  excludes: tuple[str, ...] = ()
  build_scrubs: tuple[str, ...] = ()
  scrubbed_prefix: Optional[str] = None


def _rules() -> ExportRules:
  """Returns the export's rules.

  They are empty in the exported repository, which holds no excluded file and
  no unexported BUILD code.
  """
  rules = ExportRules()
  return rules


RULES = _rules()


@functools.cache
def glob_to_regex(pattern: str) -> re.Pattern[str]:
  """Compiles an export/Bazel style glob (`**`, `*`, `?`) to a regex."""
  out = []
  i = 0
  while i < len(pattern):
    if pattern.startswith("**/", i):
      out.append("(?:.*/)?")
      i += 3
    elif pattern.startswith("**", i):
      out.append(".*")
      i += 2
    elif pattern[i] == "*":
      out.append("[^/]*")
      i += 1
    elif pattern[i] == "?":
      out.append("[^/]")
      i += 1
    else:
      out.append(re.escape(pattern[i]))
      i += 1
  return re.compile("".join(out))


def is_excluded(rel_path: str) -> bool:
  """Whether the export drops `rel_path` (relative to the OpenSpiel root)."""
  path = f"{RULES.root}/{rel_path}" if RULES.root else rel_path
  return any(glob_to_regex(p).fullmatch(path) for p in RULES.excludes)
