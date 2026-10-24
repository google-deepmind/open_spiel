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

"""Maps labels outside OpenSpiel to CMake targets.

Every entry is keyed by both label forms: the one in the source BUILD files and
the one the export rewrites it to in BUILD.bazel, so either input produces the
same CMake. The CMake targets are defined by the hand-written root
CMakeLists.txt (cmake/dependencies.cmake). A label missing from this table is a
hard error. A test checks the exported forms against the export configuration.
"""

import dataclasses
from typing import Optional

# Prefixes of OpenSpiel's own labels, in source and exported form.
OPEN_SPIEL_PREFIXES = ("//third_party/open_spiel", "//open_spiel")

# Label of the platform constraint that marks a target incompatible.
INCOMPATIBLE_LABELS = frozenset({
    "//third_party/bazel_platforms:incompatible",
    "@platforms//:incompatible",
})

# Select key that matches when no other key does.
DEFAULT_CONDITION = "//conditions:default"

# Platform constraints usable as select() keys, in source and exported form,
# and the CMake variable that is true on that platform.
PLATFORM_CONDITIONS = {
    "//third_party/bazel_platforms/os:windows": "WIN32",
    "@platforms//os:windows": "WIN32",
}

# Prefixes of labels naming Python packages, in source and exported form. The
# top-level package (e.g. `jax` in `//third_party/py/jax/example_libraries`)
# decides which optional dependency group, if any, a dependency needs.
PYTHON_PACKAGE_PREFIXES = ("//third_party/py/", "@py_deps//")

# Top-level Python package -> optional dependency group. A copy of
# OPTIONAL_DEP_GROUPS in the exported bazel/python/py_deps_table.bzl, kept here
# so that generation needs only this directory; a test checks they agree. Like
# the Bazel rules, this keys on the top-level package only, so every label
# under it (`jax`, `jax/experimental`, `flax:core`) gets the same group.
OPTIONAL_PYTHON_GROUPS = {
    "chex": "jax",
    "distrax": "jax",
    "flax": "jax",
    "haiku": "jax",
    "jax": "jax",
    "mctx": "jax",
    "optax": "jax",
    "orbax": "jax",
    "rlax": "jax",
    "torch": "pytorch",
    "cma": "python_misc",
    "cvxpy": "python_misc",
    "matplotlib": "python_misc",
    "nashpy": "python_misc",
    "networkx": "python_misc",
    "pandas": "python_misc",
    "pyarrow": "python_misc",
    "pygraphviz": "python_misc",
}

# CMake switch that enables the tests needing each optional group.
PYTHON_GROUP_OPTIONS = {
    "jax": "OPEN_SPIEL_ENABLE_JAX",
    "pytorch": "OPEN_SPIEL_ENABLE_PYTORCH",
    "python_misc": "OPEN_SPIEL_ENABLE_PYTHON_MISC",
}

# Top-level Python packages whose dependents CMake never runs, and why. A copy
# of MANUAL_PACKAGES in bazel/python/py_deps_table.bzl (tagged "manual" there).
PYTHON_PACKAGES_NOT_RUN = {
    "pygraphviz": "needs pygraphviz, which ships no wheels",
}


@dataclasses.dataclass(frozen=True)
class ExternalLabel:
  """How one external label translates to CMake.

  Attributes:
    source: Label as written in the source BUILD files.
    exported: Label as written in the exported BUILD.bazel files.
    cmake: CMake target to link, or None to drop the dependency.
    option: CMake option the dependency needs; dependents are only built when
      it is ON.
    skip: If set, CMake does not build targets depending on this label; the
      text says why (it appears in the generated files).
  """

  source: str
  exported: str
  cmake: Optional[str] = None
  option: Optional[str] = None
  skip: Optional[str] = None


def normalize(label: str) -> str:
  """Spells out the implicit target name: `//a/b` -> `//a/b:b`, `@x` -> `@x//:x`."""
  if label.startswith("@") and "//" not in label:
    return f"{label}//:{label[1:]}"
  if ":" in label:
    return label
  return f"{label}:{label.rsplit('/', 1)[-1]}"


def _absl(path: str, cmake: str) -> ExternalLabel:
  return ExternalLabel(
      source=f"//third_party/absl/{path}",
      exported=f"@abseil-cpp//absl/{path}",
      cmake=cmake)


_ABSL = (
    _absl("algorithm:container", "absl::algorithm_container"),
    _absl("base:core_headers", "absl::core_headers"),
    _absl("container:btree", "absl::btree"),
    _absl("container:flat_hash_map", "absl::flat_hash_map"),
    _absl("container:flat_hash_set", "absl::flat_hash_set"),
    _absl("container:inlined_vector", "absl::inlined_vector"),
    _absl("container:node_hash_map", "absl::node_hash_map"),
    _absl("container:node_hash_set", "absl::node_hash_set"),
    _absl("flags:flag", "absl::flags"),
    _absl("flags:parse", "absl::flags_parse"),
    _absl("flags:usage", "absl::flags_usage"),
    _absl("functional:any_invocable", "absl::any_invocable"),
    _absl("hash", "absl::hash"),
    _absl("log", "absl::log"),
    _absl("log:check", "absl::check"),
    _absl("memory", "absl::memory"),
    _absl("numeric:bits", "absl::bits"),
    _absl("random", "absl::random_random"),
    _absl("random:bit_gen_ref", "absl::random_bit_gen_ref"),
    _absl("random:distributions", "absl::random_distributions"),
    _absl("status", "absl::status"),
    _absl("status:statusor", "absl::statusor"),
    _absl("strings", "absl::strings"),
    _absl("strings:str_format", "absl::str_format"),
    _absl("strings:string_view", "absl::string_view"),
    _absl("synchronization", "absl::synchronization"),
    _absl("time", "absl::time"),
    _absl("types:optional", "absl::optional"),
    _absl("types:span", "absl::span"),
    _absl("types:variant", "absl::variant"),
)

_OTHER = (
    ExternalLabel(
        source="//third_party/json",
        exported="@nlohmann_json//:json",
        cmake="nlohmann_json::nlohmann_json"),
    ExternalLabel(
        source="//third_party/pybind11_json",
        exported="@pybind11_json",
        cmake="open_spiel_ext::pybind11_json"),
    ExternalLabel(
        source="//third_party/pybind11_abseil:absl_casters",
        exported="@pybind11_abseil//pybind11_abseil:absl_casters",
        cmake="open_spiel_ext::pybind11_abseil"),
    ExternalLabel(
        source="//third_party/pybind11_abseil:status_casters",
        exported="@pybind11_abseil//pybind11_abseil:status_casters",
        cmake="open_spiel_ext::pybind11_abseil"),
    ExternalLabel(
        source="//testing/base/public:gunit_main",
        exported="@googletest//:gtest_main",
        skip="needs googletest"),
    ExternalLabel(
        source="//testing/base/public:gunit",
        exported="@googletest//:gtest",
        skip="needs googletest"),
    ExternalLabel(
        source="//third_party/double_dummy_solver:dds",
        exported="@dds//:dds",
        cmake="open_spiel_ext::dds"),
    ExternalLabel(
        source="//third_party/acpc",
        exported="@acpc",
        cmake="open_spiel_ext::acpc",
        option="OPEN_SPIEL_BUILD_WITH_ACPC"),
    ExternalLabel(
        source="//third_party/roshambo",
        exported="@roshambo",
        cmake="open_spiel_ext::roshambo",
        option="OPEN_SPIEL_BUILD_WITH_ROSHAMBO"),
    ExternalLabel(
        source="//third_party/roshambo/BotClasses:rsb_bot",
        exported="@roshambo//:rsb_bot",
        cmake="open_spiel_ext::roshambo",
        option="OPEN_SPIEL_BUILD_WITH_ROSHAMBO"),
    ExternalLabel(
        source="//third_party/nathansttt_hearts:xinxin",
        exported="@xinxin//:xinxin",
        cmake="open_spiel_ext::xinxin_hearts",
        option="OPEN_SPIEL_BUILD_WITH_XINXIN"),
    ExternalLabel(
        source="//third_party/libnop",
        exported="@libnop",
        cmake="open_spiel_ext::libnop",
        option="OPEN_SPIEL_BUILD_WITH_LIBNOP"),
    ExternalLabel(
        source="//third_party/ortools/ortools/linear_solver:linear_solver_glop",
        exported="@ortools//:linear_solver_glop",
        cmake="open_spiel_ext::ortools",
        option="OPEN_SPIEL_BUILD_WITH_ORTOOLS"),
    ExternalLabel(
        source="//third_party/ortools/ortools/linear_solver:linear_solver_lib",
        exported="@ortools//:linear_solver_lib",
        cmake="open_spiel_ext::ortools",
        option="OPEN_SPIEL_BUILD_WITH_ORTOOLS"),
    ExternalLabel(
        source="//third_party/py/hanabi_learning_environment:hanabi",
        exported="@hanabi//:hanabi",
        cmake="open_spiel_ext::hanabi",
        option="OPEN_SPIEL_BUILD_WITH_HANABI"),
    ExternalLabel(
        source="//third_party/py/torch:torch_headers",
        exported="@libtorch//:libtorch",
        cmake="open_spiel_ext::torch",
        option="OPEN_SPIEL_BUILD_WITH_LIBTORCH"),
)

EXTERNAL_LABELS: tuple[ExternalLabel, ...] = _ABSL + _OTHER


def _index() -> dict[str, ExternalLabel]:
  index = {}
  for entry in EXTERNAL_LABELS:
    for label in (entry.source, entry.exported):
      key = normalize(label)
      if key in index:
        raise ValueError(f"Duplicate label in label_map: {label}")
      index[key] = entry
  return index


_INDEX = _index()


def lookup(label: str) -> Optional[ExternalLabel]:
  """Returns the entry for an external label (either form), or None."""
  return _INDEX.get(normalize(label))


def is_open_spiel(label: str) -> bool:
  """Whether `label` is absolute and names an OpenSpiel package."""
  return any(label == p or label.startswith(p + "/") or
             label.startswith(p + ":") for p in OPEN_SPIEL_PREFIXES)


def open_spiel_package_and_name(label: str) -> tuple[str, str]:
  """Splits an absolute OpenSpiel label into (package, name).

  Args:
    label: e.g. `//third_party/open_spiel/games/chess:chess_board`.

  Returns:
    The package relative to the OpenSpiel root ("" for the root) and the
    target name.
  """
  for prefix in OPEN_SPIEL_PREFIXES:
    if label == prefix or label.startswith((prefix + "/", prefix + ":")):
      rest = normalize(label)[len(prefix):]
      package, name = rest.split(":", 1)
      return package.lstrip("/"), name
  raise ValueError(f"Not an OpenSpiel label: {label}")


def python_package(label: str) -> Optional[str]:
  """The top-level Python package a label names, or None if it names none.

  Args:
    label: e.g. `//third_party/py/jax/example_libraries:optimizers` or
      `@py_deps//jax/example_libraries:optimizers`.

  Returns:
    The top-level package (`jax`), or None for labels outside the Python
    package trees.
  """
  for prefix in PYTHON_PACKAGE_PREFIXES:
    if label.startswith(prefix):
      return label[len(prefix):].split(":", 1)[0].split("/", 1)[0]
  return None
