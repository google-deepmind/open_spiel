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

"""Evaluates a Bazel BUILD file as Python, recording the rules it declares.

BUILD files are Starlark, which for the constructs OpenSpiel uses is a subset of
Python. We execute them with stub functions for every known rule and builtin,
so no Bazel installation is needed. Anything unknown is a hard error.
"""

import dataclasses
import os
import re
from typing import Any, Callable, Iterable, Sequence, Union

from open_spiel.scripts.bazel_to_cmake import export_rules

# Rules that produce C++ targets the generator translates to CMake.
CC_RULES = frozenset({
    "cc_binary",
    "cc_library",
    "cc_test",
    "pybind_extension",
    "pybind_library",
    "pybind_library_test",
})

# Rules that are recorded because C++ rules may refer to them.
SUPPORT_RULES = frozenset({
    "bool_flag",
    "config_setting",
    "filegroup",
})

# Rules with no CMake counterpart: Python ships through setup.py, and the rest
# is Bazel bookkeeping.
IGNORED_RULES = frozenset({
    "colab_binary",
    "exports_files",
    "license",
    "licenses",
    "package",
    "proto_library",
    "py_binary",
    "py_library",
    "py_proto_library",
    "py_test",
    "pytype_strict_binary",
    "pytype_strict_contrib_test",
    "pytype_strict_library",
})

KNOWN_RULES = CC_RULES | SUPPORT_RULES | IGNORED_RULES

# Non-rule symbols that BUILD files load from .bzl files. Their values only
# matter to Bazel; attributes holding them are dropped by the generator.
OPAQUE_SYMBOLS = frozenset({
    "TORCH_COPTS",
    "TORCH_FEATURES",
})

# Python builtins that Starlark also provides and BUILD files may use.
_BUILTINS = {
    "True": True,
    "False": False,
    "None": None,
    "dict": dict,
    "enumerate": enumerate,
    "len": len,
    "list": list,
    "range": range,
    "sorted": sorted,
    "str": str,
    "zip": zip,
}


class Error(Exception):
  """Raised for any BUILD construct the generator does not understand."""


@dataclasses.dataclass(frozen=True)
class Opaque:
  """Value of a symbol loaded from a .bzl file that the generator ignores."""

  symbol: str


@dataclasses.dataclass(frozen=True)
class Select:
  """A `select()` call: (config_setting label, value) pairs in BUILD order."""

  branches: tuple[tuple[str, Any], ...]


class Selectable:
  """An attribute value that is a concatenation of lists and selects."""

  def __init__(self, parts: Sequence[Union[list[Any], Select]]):
    self.parts = tuple(parts)

  def __add__(self, other: Any) -> "Selectable":
    if isinstance(other, Selectable):
      return Selectable(self.parts + other.parts)
    if isinstance(other, list):
      return Selectable(self.parts + (other,))
    return NotImplemented

  def __radd__(self, other: Any) -> "Selectable":
    if isinstance(other, list):
      return Selectable((other,) + self.parts)
    return NotImplemented

  def __repr__(self) -> str:
    return f"Selectable({self.parts!r})"


class GlobResult(list):
  """Result of `glob()`; C++ rules may not use it (runfiles lack sources)."""


@dataclasses.dataclass
class Rule:
  """One rule instantiation in a BUILD file."""

  kind: str
  attrs: dict[str, Any]
  package: str

  @property
  def name(self) -> str:
    return self.attrs["name"]


def preprocess(text: str) -> str:
  """Applies the BUILD-file scrubs the export performs.

  Running the same scrubs here makes the generator see identical input whether
  it reads a source BUILD file or the exported BUILD.bazel.

  Args:
    text: Raw BUILD file contents.

  Returns:
    The contents with the export's scrubs applied.
  """
  for pattern in export_rules.RULES.build_scrubs:
    text = re.sub(pattern, "", text)
  return text


def _glob(package_dir: str) -> Callable[..., GlobResult]:
  """Returns a `glob()` implementation rooted at `package_dir`."""

  def glob(include: Iterable[str],
           exclude: Iterable[str] = (),
           exclude_directories: int = 1,
           allow_empty: bool = True) -> GlobResult:
    del exclude_directories, allow_empty  # Files only; empty is fine.
    matches = set()
    include_res = [export_rules.glob_to_regex(p) for p in include]
    exclude_res = [export_rules.glob_to_regex(p) for p in exclude]
    for dirpath, dirnames, filenames in os.walk(package_dir):
      rel_dir = os.path.relpath(dirpath, package_dir)
      if rel_dir != "." and any(
          f in filenames for f in ("BUILD", "BUILD.bazel")):
        dirnames[:] = []  # Subpackage: glob does not cross it.
        continue
      for filename in filenames:
        rel = os.path.normpath(os.path.join(rel_dir, filename))
        if (any(r.fullmatch(rel) for r in include_res) and
            not any(r.fullmatch(rel) for r in exclude_res)):
          matches.add(rel)
    return GlobResult(sorted(matches))

  return glob


def evaluate(text: str, package: str, package_dir: str,
             filename: str) -> list[Rule]:
  """Evaluates a BUILD file and returns the rules it declares.

  Args:
    text: BUILD file contents (unpreprocessed).
    package: Package path relative to the OpenSpiel root ("" for the root).
    package_dir: Directory of the package on disk, used by `glob()`.
    filename: File name used in error messages.

  Returns:
    All rules, in declaration order.

  Raises:
    Error: On unknown rules, symbols or syntax.
  """
  rules: list[Rule] = []

  def make_rule(kind: str) -> Callable[..., None]:

    def rule(*args: Any, **kwargs: Any) -> None:
      if kind in ("licenses", "exports_files"):
        return
      if args:
        raise Error(
            f"{filename}: {kind}() called with positional arguments")
      if kind == "package":
        return
      if "name" not in kwargs:
        raise Error(f"{filename}: {kind}() without a name")
      rules.append(Rule(kind=kind, attrs=kwargs, package=package))

    return rule

  scope: dict[str, Any] = {"__builtins__": dict(_BUILTINS)}
  for kind in KNOWN_RULES:
    scope[kind] = make_rule(kind)

  def load(bzl: str, *symbols: str, **renamed: str) -> None:
    del bzl  # Only the loaded symbol names matter.
    for local, symbol in [(s, s) for s in symbols] + list(renamed.items()):
      if symbol in KNOWN_RULES:
        scope[local] = make_rule(symbol)
      elif symbol in OPAQUE_SYMBOLS:
        scope[local] = Opaque(symbol)
      else:
        raise Error(
            f"{filename}: load() of unknown symbol {symbol!r}; teach "
            "scripts/bazel_to_cmake/build_file.py about it")

  def select(branches: dict[str, Any], no_match_error: str = "") -> Selectable:
    del no_match_error
    return Selectable([Select(tuple(branches.items()))])

  scope["load"] = load
  scope["select"] = select
  scope["glob"] = _glob(package_dir)

  text = preprocess(text)
  marker = export_rules.RULES.scrubbed_prefix
  if marker and marker in text:
    raise Error(
        f"{filename}: {marker} is referenced "
        "outside a line of its own or a GOOGLE-INTERNAL block, so the "
        "export would leak it")
  try:
    code = compile(text, filename, "exec")
    exec(code, scope)  # pylint: disable=exec-used
  except Error:
    raise
  except NameError as e:
    raise Error(
        f"{filename}: {e}; unknown rule or symbol (teach "
        "scripts/bazel_to_cmake/build_file.py about it)") from e
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise Error(f"{filename}: failed to evaluate: {e!r}") from e
  return rules
