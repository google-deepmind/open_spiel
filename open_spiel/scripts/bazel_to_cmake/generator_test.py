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

"""Tests for generator and the bazel_to_cmake command line."""

import contextlib
import io
import os
import textwrap

from absl.testing import absltest

from open_spiel.scripts.bazel_to_cmake import bazel_to_cmake
from open_spiel.scripts.bazel_to_cmake import export_rules
from open_spiel.scripts.bazel_to_cmake import generator

# A small source tree, with BUILD files written in their source form.
_SOURCE_TREE = {
    "BUILD": """
        load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")
        load("//third_party/bazel_skylib/rules:common_settings.bzl", "bool_flag")

        package(default_visibility = ["//visibility:public"])

        licenses(["notice"])

        bool_flag(
            name = "build_with_acpc",
            build_setting_default = False,
        )

        config_setting(
            name = "build_with_acpc_enabled",
            flag_values = {":build_with_acpc": "True"},
        )

        cc_library(
            name = "open_spiel",
            srcs = ["spiel.cc"],
            hdrs = ["spiel.h"],
            copts = ["-fexceptions"],
            features = ["-use_header_modules"],
            deps = [
                "//third_party/absl/strings",
                "//third_party/json",
            ],
            alwayslink = 1,
        )

        cc_library(
            name = "all_games",
            deps = [
                ":open_spiel",
                "//third_party/open_spiel/games",
            ],
        )

        cc_test(
            name = "spiel_test",
            srcs = ["tests/spiel_test.cc"],
            deps = [":all_games"],
        )
        """,
    "games/BUILD": """
        cc_library(
            name = "games",
            deps = [
                "//third_party/open_spiel/games/foo",
            ] + select({
                "//third_party/open_spiel:build_with_acpc_enabled": [
                    "//third_party/open_spiel/games/poker",
                ],
                "//conditions:default": [],
            }),
        )
        """,
    "games/foo/BUILD": """
        cc_library(
            name = "foo",
            srcs = ["foo.cc"],
            hdrs = ["foo.h"],
            deps = ["//third_party/open_spiel"],
        )

        cc_library(
            name = "foo_headers",
            hdrs = ["foo_headers.h"],
            testonly = 1,
        )

        cc_binary(
            name = "foo_main",
            srcs = ["foo_main.cc"],
            deps = [":foo"],
        )

        cc_test(
            name = "foo_test",
            srcs = ["foo_test.cc"],
            args = [
                "--binary=third_party/open_spiel/games/foo/foo_main",
                "--data=$(location :data.txt)",
                "--name=hello world",
            ],
            data = [
                ":data.txt",
                ":foo_main",
            ],
            deps = [
                ":foo",
                ":foo_headers",
            ],
        )

        cc_test(
            name = "foo_gtest",
            srcs = ["foo_gtest.cc"],
            deps = [
                ":foo",
                "//testing/base/public:gunit_main",
            ],
        )
        """,
    "games/poker/BUILD": """
        cc_library(
            name = "poker",
            srcs = ["poker.cc"],
            hdrs = ["poker.h"],
            deps = [
                "//third_party/acpc",
                "//third_party/open_spiel",
            ],
            alwayslink = 1,
        )

        cc_test(
            name = "foo_test",
            srcs = ["foo_test.cc"],
            deps = [":poker"],
        )
        """,
    "python/BUILD": """
        py_library(
            name = "python",
            srcs = ["__init__.py"],
        )
        """,
    "python/pybind11/BUILD": """
        load("//third_party/pybind11/utils:build_defs.bzl", "pybind_extension", "pybind_library")

        pybind_library(
            name = "games_foo",
            srcs = ["games_foo.cc"],
            hdrs = ["games_foo.h"],
            deps = ["//third_party/open_spiel/games/foo"],
        )

        pybind_extension(
            name = "pyspiel",
            srcs = ["pyspiel.cc"],
            copts = ["-DOPEN_SPIEL_BUILD_WITH_ACPC"],
            deps = [
                ":games_foo",
                "//third_party/open_spiel:all_games",
                "//third_party/pybind11_abseil:absl_casters",
                "//third_party/pybind11_abseil:status_casters",
            ],
        )
        """,
}

_SOURCE_FILES = (
    "spiel.cc", "spiel.h", "tests/spiel_test.cc", "games/foo/foo.cc",
    "games/foo/foo.h", "games/foo/foo_headers.h", "games/foo/foo_main.cc",
    "games/foo/foo_test.cc", "games/foo/foo_gtest.cc", "games/foo/data.txt",
    "games/poker/poker.cc", "games/poker/poker.h", "games/poker/foo_test.cc",
    "python/__init__.py", "python/pybind11/games_foo.cc",
    "python/pybind11/games_foo.h", "python/pybind11/pyspiel.cc",
)

# The export's label rewrites, as far as the tree above needs them.
_EXPORT_REWRITES = (
    ("//third_party/bazel_rules/rules_cc/cc:", "@rules_cc//cc:"),
    ("//third_party/bazel_skylib/", "@bazel_skylib//"),
    ("//third_party/pybind11/utils:build_defs.bzl",
     "//bazel/python:pybind.bzl"),
    ("//third_party/open_spiel", "//open_spiel"),
    ("=third_party/open_spiel/", "=open_spiel/"),
    ("//third_party/absl/", "@abseil-cpp//absl/"),
    ('"//third_party/json"', '"@nlohmann_json//:json"'),
    ('"//third_party/acpc"', '"@acpc"'),
    ("//third_party/pybind11_abseil:", "@pybind11_abseil//pybind11_abseil:"),
    ("//testing/base/public:gunit_main", "@googletest//:gtest_main"),
)

# Unexported additions, which the export scrubs. The marker literals are
# split so that the export's own scrub leaves this file intact.
_BEGIN = "# BEGIN GOOGLE-" + "INTERNAL"
_END = "# END GOOGLE-" + "INTERNAL"
_INTERNAL_ADDITIONS = {
    "BUILD": f"""
        {_BEGIN}
        exports_files(["BUILD", "targets.cmake"])

        cc_test(
            name = "google_only_test",
            srcs = ["tests/google_only_test.cc"],
            deps = ["//util/task:status"],
        )
        {_END}
        """,
}

_POKER_CMAKE = """\
# Generated by scripts/generate_cmake.sh from BUILD. Do not edit.

if(OPEN_SPIEL_BUILD_WITH_ACPC)
  add_library(open_spiel_games_poker_poker_objs OBJECT
    poker.cc
    poker.h
  )
  target_link_libraries(open_spiel_games_poker_poker_objs PUBLIC
    open_spiel_ext::acpc
    open_spiel::open_spiel
  )
  add_library(open_spiel_games_poker_poker INTERFACE)
  target_link_libraries(open_spiel_games_poker_poker INTERFACE
    open_spiel_games_poker_poker_objs
    $<TARGET_OBJECTS:open_spiel_games_poker_poker_objs>
  )
  add_library(open_spiel::games::poker::poker ALIAS open_spiel_games_poker_poker)
  set_property(GLOBAL APPEND PROPERTY OPEN_SPIEL_OBJECT_LIBRARIES open_spiel_games_poker_poker)
endif()

if(OPEN_SPIEL_BUILD_WITH_ACPC)
  add_executable(open_spiel_games_poker_foo_test
    foo_test.cc
  )
  target_link_libraries(open_spiel_games_poker_foo_test PRIVATE
    open_spiel::games::poker::poker
  )
  add_test(NAME games/poker/foo_test COMMAND open_spiel_games_poker_foo_test)
endif()
"""

_FOO_CMAKE = """\
# Generated by scripts/generate_cmake.sh from BUILD. Do not edit.

add_library(open_spiel_games_foo_foo_objs OBJECT
  foo.cc
  foo.h
)
target_link_libraries(open_spiel_games_foo_foo_objs PUBLIC
  open_spiel::open_spiel
)
add_library(open_spiel_games_foo_foo INTERFACE)
target_link_libraries(open_spiel_games_foo_foo INTERFACE
  open_spiel_games_foo_foo_objs
  $<TARGET_OBJECTS:open_spiel_games_foo_foo_objs>
)
add_library(open_spiel::games::foo::foo ALIAS open_spiel_games_foo_foo)
set_property(GLOBAL APPEND PROPERTY OPEN_SPIEL_OBJECT_LIBRARIES open_spiel_games_foo_foo)

add_library(open_spiel_games_foo_foo_headers INTERFACE)
add_library(open_spiel::games::foo::foo_headers ALIAS open_spiel_games_foo_foo_headers)

add_executable(open_spiel_games_foo_foo_main
  foo_main.cc
)
set_target_properties(open_spiel_games_foo_foo_main PROPERTIES OUTPUT_NAME foo_main)
target_link_libraries(open_spiel_games_foo_foo_main PRIVATE
  open_spiel::games::foo::foo
)

add_executable(open_spiel_games_foo_foo_test
  foo_test.cc
)
target_link_libraries(open_spiel_games_foo_foo_test PRIVATE
  open_spiel::games::foo::foo
  open_spiel::games::foo::foo_headers
)
add_test(NAME games/foo/foo_test COMMAND open_spiel_games_foo_foo_test --binary=$<TARGET_FILE:open_spiel_games_foo_foo_main> --data=${open_spiel_SOURCE_DIR}/games/foo/data.txt "--name=hello world")

# cc_test foo_gtest: not built by CMake (needs googletest).
"""

_GAMES_CMAKE = """\
# Generated by scripts/generate_cmake.sh from BUILD. Do not edit.

add_library(open_spiel_games_games INTERFACE)
target_link_libraries(open_spiel_games_games INTERFACE
  open_spiel::games::foo::foo
)
if(OPEN_SPIEL_BUILD_WITH_ACPC)
  target_link_libraries(open_spiel_games_games INTERFACE
    open_spiel::games::poker::poker
  )
endif()
add_library(open_spiel::games::games ALIAS open_spiel_games_games)
set_property(GLOBAL APPEND PROPERTY OPEN_SPIEL_OBJECT_LIBRARIES open_spiel_games_games)

add_subdirectory(foo)
add_subdirectory(poker)
"""

_PYBIND_CMAKE = """\
# Generated by scripts/generate_cmake.sh from BUILD. Do not edit.

if(OPEN_SPIEL_BUILD_WITH_PYTHON)
  add_library(open_spiel_python_pybind11_games_foo_objs OBJECT
    games_foo.cc
    games_foo.h
  )
  target_link_libraries(open_spiel_python_pybind11_games_foo_objs PUBLIC
    open_spiel::games::foo::foo
    open_spiel_ext::pybind11
  )
  add_library(open_spiel_python_pybind11_games_foo INTERFACE)
  target_link_libraries(open_spiel_python_pybind11_games_foo INTERFACE
    open_spiel_python_pybind11_games_foo_objs
    $<TARGET_OBJECTS:open_spiel_python_pybind11_games_foo_objs>
  )
  add_library(open_spiel::python::pybind11::games_foo ALIAS open_spiel_python_pybind11_games_foo)
endif()

if(OPEN_SPIEL_BUILD_WITH_PYTHON)
  add_library(pyspiel MODULE
    pyspiel.cc
  )
  target_link_libraries(pyspiel PRIVATE
    open_spiel::python::pybind11::games_foo
    open_spiel::all_games
    open_spiel_ext::pybind11_abseil
    open_spiel_ext::pybind11
  )
endif()
"""


def _exported(text: str) -> str:
  for before, after in _EXPORT_REWRITES:
    text = text.replace(before, after)
  return text


def _write_tree(root: str, build_name: str, internal: bool,
                export: bool) -> None:
  """Writes the test tree under `root`, in source or exported form."""
  files = {rel: "" for rel in _SOURCE_FILES}
  for rel, text in _SOURCE_TREE.items():
    text = textwrap.dedent(text)
    if internal and rel in _INTERNAL_ADDITIONS:
      text += textwrap.dedent(_INTERNAL_ADDITIONS[rel])
    if export:
      text = _exported(text)
    files[os.path.join(os.path.dirname(rel), build_name)] = text
  for rel, text in files.items():
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      f.write(text)


def _generate(root: str, **kwargs) -> dict[str, str]:
  return generator.Workspace(root, **kwargs).generate()


class GeneratorTest(absltest.TestCase):
  """Translation of BUILD trees to CMake."""

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path
    _write_tree(self.root, "BUILD", internal=False, export=False)

  def write(self, rel: str, text: str) -> None:
    path = os.path.join(self.root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      f.write(textwrap.dedent(text))

  def test_outputs(self):
    outputs = _generate(self.root)
    self.assertCountEqual(outputs, [
        "targets.cmake", "games/CMakeLists.txt", "games/foo/CMakeLists.txt",
        "games/poker/CMakeLists.txt", "python/pybind11/CMakeLists.txt"])
    self.assertEqual(outputs["games/poker/CMakeLists.txt"], _POKER_CMAKE)
    self.assertEqual(outputs["games/foo/CMakeLists.txt"], _FOO_CMAKE)
    self.assertEqual(outputs["games/CMakeLists.txt"], _GAMES_CMAKE)
    self.assertEqual(outputs["python/pybind11/CMakeLists.txt"], _PYBIND_CMAKE)

  def test_root_package(self):
    targets = _generate(self.root)["targets.cmake"]
    self.assertTrue(targets.startswith(generator.HEADER))
    self.assertIn("add_library(open_spiel_open_spiel_objs OBJECT\n", targets)
    self.assertIn("add_library(open_spiel::open_spiel ALIAS "
                  "open_spiel_open_spiel)\n", targets)
    self.assertIn("  absl::strings\n  nlohmann_json::nlohmann_json\n", targets)
    self.assertIn("add_test(NAME spiel_test COMMAND open_spiel_spiel_test)",
                  targets)
    self.assertNotIn("add_subdirectory", targets)
    self.assertNotIn("-fexceptions", targets)

  def test_exported_tree_gives_identical_output(self):
    expected = _generate(self.root)
    internal = bool(export_rules.RULES.build_scrubs)
    source = self.create_tempdir().full_path
    _write_tree(source, "BUILD", internal=internal, export=False)
    exported = self.create_tempdir().full_path
    _write_tree(exported, "BUILD.bazel", internal=False, export=True)
    self.assertEqual(_generate(source), expected)
    self.assertEqual(_generate(exported), expected)

  def test_scrubbed_reference_must_be_alone_on_its_line(self):
    if not export_rules.RULES.scrubbed_prefix:
      self.skipTest("Exported repository: nothing is scrubbed.")
    self.write("games/foo/BUILD", """
        cc_library(
            name = "foo",
            deps = [
                "//third_party/open_spiel/google:base",
            ],
        )
        """)
    self.assertNotIn("google:base", _generate(self.root)[
        "games/foo/CMakeLists.txt"])
    self.write("games/foo/BUILD", """
        cc_library(
            name = "foo",
            deps = ["//third_party/open_spiel/google:base"],
        )
        """)
    with self.assertRaisesRegex(generator.GenerationError, "leak"):
      _generate(self.root)

  def test_excluded_package_is_skipped(self):
    if not export_rules.RULES.excludes:
      self.skipTest("Exported repository: nothing is excluded.")
    self.write("games/catan/BUILD", 'cc_library(name = "catan")')
    self.assertNotIn("games/catan/CMakeLists.txt", _generate(self.root))

  def test_excluded_source_is_an_error(self):
    if not export_rules.RULES.excludes:
      self.skipTest("Exported repository: nothing is excluded.")
    self.write("games/foo/google_foo.cc", "")
    self.write("games/foo/BUILD", """
        cc_library(name = "foo", srcs = ["google_foo.cc"])
        """)
    with self.assertRaisesRegex(generator.GenerationError, "not exported"):
      _generate(self.root)

  def test_missing_source_is_an_error(self):
    self.write("games/foo/BUILD", 'cc_library(name = "foo", srcs = ["x.cc"])')
    with self.assertRaisesRegex(generator.GenerationError, "does not exist"):
      _generate(self.root)
    self.assertIn("games/foo/CMakeLists.txt",
                  _generate(self.root, check_sources=False))

  def test_unknown_label_is_an_error(self):
    self.write("games/foo/BUILD", """
        cc_library(name = "foo", deps = ["//third_party/unknown"])
        """)
    with self.assertRaisesRegex(generator.GenerationError,
                                "unknown label //third_party/unknown"):
      _generate(self.root)

  def test_unknown_attribute_is_an_error(self):
    self.write("games/foo/BUILD",
               'cc_library(name = "foo", linkopts = ["-lm"])')
    with self.assertRaisesRegex(generator.GenerationError, "linkopts"):
      _generate(self.root)

  def test_untranslatable_copt_is_an_error(self):
    self.write("games/foo/BUILD", 'cc_library(name = "foo", copts = ["-O3"])')
    with self.assertRaisesRegex(generator.GenerationError, "-O3"):
      _generate(self.root)

  def test_missing_dependency_is_an_error(self):
    self.write("games/foo/BUILD", 'cc_library(name = "foo", deps = [":bar"])')
    with self.assertRaisesRegex(generator.GenerationError, "no target"):
      _generate(self.root)

  def test_target_compatible_with_guards_target(self):
    self.write("games/foo/BUILD", """
        cc_library(
            name = "foo",
            target_compatible_with = select({
                "//third_party/open_spiel:build_with_acpc_enabled": [],
                "//conditions:default": [
                    "//third_party/bazel_platforms:incompatible",
                ],
            }),
        )
        """)
    outputs = _generate(self.root)
    self.assertTrue(outputs["games/foo/CMakeLists.txt"].startswith(
        generator.HEADER + "\nif(OPEN_SPIEL_BUILD_WITH_ACPC)\n"))
    # The guard propagates to unconditional dependents.
    self.assertIn("if(OPEN_SPIEL_BUILD_WITH_ACPC)\n  add_library("
                  "open_spiel_games_games INTERFACE)",
                  outputs["games/CMakeLists.txt"])

  def test_platform_incompatible_target_is_guarded(self):
    self.write(
        "games/foo/BUILD",
        """
        cc_library(
            name = "foo",
            srcs = ["foo.cc"],
            target_compatible_with = select({
                "//third_party/bazel_platforms/os:windows": [
                    "//third_party/bazel_platforms:incompatible",
                ],
                "//conditions:default": [],
            }),
        )
        """,
    )
    self.write("games/foo/foo.cc", "")
    outputs = _generate(self.root)
    self.assertTrue(
        outputs["games/foo/CMakeLists.txt"].startswith(
            generator.HEADER + "\nif(NOT WIN32)\n"
        )
    )
    # The guard propagates to unconditional dependents.
    self.assertIn(
        "if(NOT WIN32)\n  add_library(open_spiel_games_games",
        outputs["games/CMakeLists.txt"],
    )

  def test_platform_select_on_deps(self):
    self.write(
        "games/BUILD",
        """
        cc_library(
            name = "games",
            deps = select({
                "@platforms//os:windows": [],
                "//conditions:default": ["//open_spiel/games/foo"],
            }),
        )
        """,
    )
    self.write("games/foo/BUILD", 'cc_library(name = "foo")')
    text = _generate(self.root)["games/CMakeLists.txt"]
    self.assertIn(
        "if(NOT WIN32)\n"
        "  target_link_libraries(open_spiel_games_games INTERFACE\n"
        "    open_spiel::games::foo::foo\n"
        "  )\n"
        "endif()\n",
        text,
    )

  def test_select_default_branch(self):
    self.write("games/foo/BUILD", """
        cc_library(
            name = "foo",
            srcs = ["foo.cc"] + select({
                "//third_party/open_spiel:build_with_acpc_enabled": ["poker.cc"],
                "//conditions:default": ["no_poker.cc"],
            }),
        )
        """)
    self.write("games/foo/poker.cc", "")
    self.write("games/foo/no_poker.cc", "")
    text = _generate(self.root)["games/foo/CMakeLists.txt"]
    self.assertIn(
        "if(NOT OPEN_SPIEL_BUILD_WITH_ACPC)\n"
        "  target_sources(open_spiel_games_foo_foo_objs PRIVATE\n"
        "    no_poker.cc\n"
        "  )\n"
        "endif()\n"
        "if(OPEN_SPIEL_BUILD_WITH_ACPC)\n"
        "  target_sources(open_spiel_games_foo_foo_objs PRIVATE\n"
        "    poker.cc\n"
        "  )\n"
        "endif()\n", text)

  def test_skip_propagates_to_dependents(self):
    self.write("games/foo/BUILD", """
        cc_library(
            name = "foo",
            testonly = 1,
            deps = ["//testing/base/public:gunit"],
        )
        cc_test(name = "foo_test", srcs = ["foo_test.cc"], deps = [":foo"])
        """)
    text = _generate(self.root)["games/foo/CMakeLists.txt"]
    self.assertIn("# cc_library foo: not built by CMake", text)
    self.assertIn("# cc_test foo_test: not built by CMake", text)
    self.assertNotIn("add_", text)


class PythonTestsTest(absltest.TestCase):
  """Generation of python/python_tests.cmake."""

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path
    _write_tree(self.root, "BUILD", internal=False, export=False)

  def write(self, rel: str, text: str) -> None:
    path = os.path.join(self.root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      f.write(textwrap.dedent(text))

  def python_tests(self, build: str, **files: str) -> str:
    """Generates python_tests.cmake for `build` as python/algorithms/BUILD."""
    self.write("python/algorithms/BUILD", build)
    for name, text in files.items():
      self.write(f"python/algorithms/{name}.py", text)
    return _generate(self.root)[generator.PYTHON_TESTS_PATH]

  def test_no_python_tests_no_file(self):
    self.assertNotIn(generator.PYTHON_TESTS_PATH, _generate(self.root))

  def test_unconditional_test(self):
    text = self.python_tests(
        """
        py_test(
            name = "cfr_test",
            srcs = ["cfr_test.py"],
            args = ["--iterations=2 --game=kuhn_poker", "--name='a b'"],
            deps = [
                "//third_party/open_spiel/python/pybind11:pyspiel",
                "//third_party/py/absl/testing:absltest",
                "//third_party/py/numpy",
            ],
        )
        """, cfr_test="")
    self.assertEqual(
        text,
        generator.HEADER + generator.PYTHON_TESTS_COMMENT + "\n"
        "open_spiel_py_test(algorithms/cfr_test.py --iterations=2 "
        '--game=kuhn_poker "--name=a b")\n')

  def test_main_and_paths_outside_python(self):
    self.write("games/foo/foo_py_test.py", "")
    self.write("games/foo/BUILD", textwrap.dedent(_SOURCE_TREE[
        "games/foo/BUILD"]) + textwrap.dedent("""
        pytype_strict_contrib_test(
            name = "foo_example_test",
            srcs = ["foo_py_test.py"],
            args = ["--data=third_party/open_spiel/games/foo/data.txt"],
            main = "foo_py_test.py",
        )
        """))
    text = _generate(self.root)[generator.PYTHON_TESTS_PATH]
    self.assertIn(
        "open_spiel_py_test(../games/foo/foo_py_test.py "
        "--data=${open_spiel_SOURCE_DIR}/games/foo/data.txt)\n", text)

  def test_python_group_guards(self):
    text = self.python_tests(
        """
        py_library(
            name = "dqn",
            srcs = ["dqn.py"],
            deps = ["//third_party/py/jax/example_libraries:optimizers"],
        )
        py_test(
            name = "dqn_test",
            srcs = ["dqn_test.py"],
            deps = [":dqn", "//third_party/py/torch:pytorch"],
        )
        py_test(
            name = "lp_test",
            srcs = ["lp_test.py"],
            deps = ["@py_deps//cvxpy"],
        )
        py_test(
            name = "graph_test",
            srcs = ["graph_test.py"],
            deps = ["//third_party/py/pygraphviz"],
        )
        """, dqn="", dqn_test="", lp_test="", graph_test="")
    self.assertIn(
        "if(OPEN_SPIEL_ENABLE_JAX AND OPEN_SPIEL_ENABLE_PYTORCH)\n"
        "  open_spiel_py_test(algorithms/dqn_test.py)\n"
        "endif()\n", text)
    self.assertIn(
        "if(OPEN_SPIEL_ENABLE_PYTHON_MISC)\n"
        "  open_spiel_py_test(algorithms/lp_test.py)\n"
        "endif()\n", text)
    self.assertIn("# py_test //python/algorithms:graph_test", text)
    self.assertNotIn("open_spiel_py_test(algorithms/graph_test.py", text)

  def test_cc_option_guards_propagate(self):
    text = self.python_tests(
        """
        py_library(
            name = "poker_utils",
            srcs = ["poker_utils.py"],
            deps = ["//third_party/open_spiel/games/poker"],
        )
        py_library(
            name = "poker_agent",
            srcs = ["poker_agent.py"],
            deps = [":poker_utils"],
        )
        py_test(
            name = "poker_agent_test",
            srcs = ["poker_agent_test.py"],
            deps = [
                ":poker_agent",
                "//third_party/open_spiel/python/pybind11:pyspiel",
            ],
        )
        py_test(
            name = "xinxin_test",
            srcs = ["xinxin_test.py"],
            target_compatible_with = select({
                "//third_party/open_spiel:build_with_acpc_enabled": [],
                "//conditions:default": [
                    "//third_party/bazel_platforms:incompatible",
                ],
            }),
        )
        """, poker_utils="", poker_agent="", poker_agent_test="",
        xinxin_test="")
    # pyspiel needs OPEN_SPIEL_BUILD_WITH_PYTHON, which every Python test does.
    self.assertIn(
        "if(OPEN_SPIEL_BUILD_WITH_ACPC)\n"
        "  open_spiel_py_test(algorithms/poker_agent_test.py)\n"
        "  open_spiel_py_test(algorithms/xinxin_test.py)\n"
        "endif()\n", text)
    self.assertNotIn("BUILD_WITH_PYTHON", text)

  def test_conditional_python_dep_adds_no_guard(self):
    text = self.python_tests(
        """
        py_test(
            name = "a_test",
            srcs = ["a_test.py"],
            deps = select({
                "//third_party/open_spiel:build_with_acpc_enabled": [
                    "//third_party/py/jax",
                ],
                "//conditions:default": [],
            }),
        )
        """, a_test="")
    self.assertIn("\nopen_spiel_py_test(algorithms/a_test.py)\n", text)

  def test_main_must_be_unambiguous(self):
    with self.assertRaisesRegex(generator.GenerationError, "set main"):
      self.python_tests(
          """
          py_test(name = "a_test", srcs = ["b.py", "c.py"])
          """, b="", c="")

  def test_unknown_attribute_is_an_error(self):
    with self.assertRaisesRegex(generator.GenerationError, "env"):
      self.python_tests(
          """
          py_test(name = "a_test", srcs = ["a_test.py"], env = {"A": "1"})
          """, a_test="")

  def test_unknown_python_package_is_an_error(self):
    with self.assertRaisesRegex(generator.GenerationError, "a_lib"):
      self.python_tests(
          """
          py_test(name = "a_test", srcs = ["a_test.py"], deps = [":a_lib"])
          """, a_test="")

  def test_colliding_tests_are_an_error(self):
    with self.assertRaisesRegex(generator.GenerationError, "collides"):
      self.python_tests(
          """
          py_test(name = "a_test", srcs = ["a_test.py"])
          py_test(name = "b_test", srcs = ["a_test.py"])
          """, a_test="")

  def test_pytest_single_file(self):
    text = self.python_tests(
        """
        pytest_test(name = "a_test", tests = ["a_test.py"])
        """, a_test="")
    self.assertIn(
        "\nopen_spiel_pytest(algorithms/a_test FILES "
        "python/algorithms/a_test.py ARGS -s)\n", text)

  def test_pytest_files_deselect_and_args(self):
    text = self.python_tests(
        """
        pytest_test(
            name = "ab_test",
            srcs = ["helper.py"],
            args = ["-q", "-k 'not slow'"],
            deselect = ["a_test.py::test_x", "b_test.py::B::test_y"],
            tests = ["a_test.py", "b_test.py"],
        )
        """, a_test="", b_test="", helper="")
    self.assertIn(
        "\nopen_spiel_pytest(algorithms/ab_test "
        "FILES python/algorithms/a_test.py python/algorithms/b_test.py "
        "DESELECT python/algorithms/a_test.py::test_x "
        "python/algorithms/b_test.py::B::test_y "
        'ARGS -q -k "not slow")\n', text)

  def test_pytest_without_tests_runs_the_package(self):
    text = self.python_tests(
        """
        pytest_test(name = "all_test", args = [])
        """)
    self.assertIn(
        "\nopen_spiel_pytest(algorithms/all_test FILES python/algorithms)\n",
        text)

  def test_pytest_missing_test_file_is_an_error(self):
    with self.assertRaisesRegex(generator.GenerationError, "does not exist"):
      self.python_tests(
          """
          pytest_test(name = "a_test", tests = ["a_test.py"])
          """)

  def test_pytest_names_collide_with_py_test_names(self):
    with self.assertRaisesRegex(generator.GenerationError,
                                "python/algorithms/b_test.py collides"):
      self.python_tests(
          """
          py_test(name = "b_test", srcs = ["b_test.py"])
          pytest_test(name = "b_test.py", tests = ["b_test.py"])
          """, b_test="")

  def test_py_test_rejects_pytest_attributes(self):
    for attr in ("tests", "deselect"):
      with self.subTest(attr):
        with self.assertRaisesRegex(generator.GenerationError,
                                    f"unsupported attributes .*{attr}"):
          self.python_tests(
              f"""
              py_test(name = "a_test", srcs = ["a_test.py"],
                      {attr} = ["a_test.py"])
              """, a_test="")

  def test_manual_tests_are_guarded(self):
    text = self.python_tests(
        """
        py_test(name = "a_test", srcs = ["a_test.py"], tags = ["manual"])
        pytest_test(
            name = "b_test",
            tags = ["manual"],
            tests = ["b_test.py"],
            deps = ["//third_party/py/jax"],
        )
        py_test(
            name = "c_test",
            srcs = ["c_test.py"],
            tags = ["notap"],
            timeout = "long",
        )
        """, a_test="", b_test="", c_test="")
    self.assertIn(
        "if(OPEN_SPIEL_RUN_MANUAL_TESTS)\n"
        "  open_spiel_py_test(algorithms/a_test.py)\n"
        "endif()\n", text)
    self.assertIn(
        "if(OPEN_SPIEL_ENABLE_JAX AND OPEN_SPIEL_RUN_MANUAL_TESTS)\n"
        "  open_spiel_pytest(algorithms/b_test FILES "
        "python/algorithms/b_test.py ARGS -s)\n"
        "endif()\n", text)
    self.assertIn("\nopen_spiel_py_test(algorithms/c_test.py)\n", text)
    self.assertNotIn("TIMEOUT", text)

  def test_manual_cc_test_is_guarded(self):
    self.write("games/foo/manual_test.cc", "")
    self.write("games/foo/BUILD", textwrap.dedent(_SOURCE_TREE[
        "games/foo/BUILD"]) + textwrap.dedent("""
        cc_test(
            name = "manual_test",
            srcs = ["manual_test.cc"],
            tags = ["manual"],
            deps = [":foo"],
        )
        """))
    text = _generate(self.root)["games/foo/CMakeLists.txt"]
    self.assertIn(
        "if(OPEN_SPIEL_RUN_MANUAL_TESTS)\n"
        "  add_executable(open_spiel_games_foo_manual_test\n", text)
    self.assertIn("\nadd_executable(open_spiel_games_foo_foo_test\n", text)

  def test_wheel_exclusion_and_cmake_contract(self):
    # python/CMakeLists.txt defines open_spiel_py_test, skips the tests in
    # WHEEL_EXCLUDED_PYTHON_TESTS when building a wheel, and includes the
    # generated file. Each excluded test must still be generated.
    cmake_lists = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "python", "CMakeLists.txt")
    if not os.path.exists(cmake_lists):
      self.skipTest("python/CMakeLists.txt is not in the runfiles.")
    with open(cmake_lists) as f:
      text = f.read()
    self.assertIn(f"function({generator.PYTHON_TEST_FUNCTION} py_test_file)",
                  text)
    self.assertIn(f"function({generator.PYTEST_FUNCTION} name)", text)
    self.assertIn("-m pytest", text)
    self.assertIn("IN_LIST WHEEL_EXCLUDED_PYTHON_TESTS", text)
    self.assertIn(f"include({generator.PYTHON_TESTS_OUTPUT})", text)


class CommandLineTest(absltest.TestCase):
  """Writing and checking generated files on disk."""

  def setUp(self):
    super().setUp()
    self.root = self.create_tempdir().full_path
    _write_tree(self.root, "BUILD", internal=False, export=False)

  def run_main(self, *args: str) -> tuple[int, str]:
    stderr = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        stderr):
      code = bazel_to_cmake.main(["--root", self.root, *args])
    return code, stderr.getvalue()

  def test_write_then_check(self):
    self.assertEqual(self.run_main("--check")[0], 1)
    self.assertEqual(self.run_main()[0], 0)
    code, stderr = self.run_main("--check")
    self.assertEqual(code, 0)
    self.assertNotIn("out of date", stderr)
    with open(os.path.join(self.root, "games/poker/CMakeLists.txt")) as f:
      self.assertEqual(f.read(), _POKER_CMAKE)

  def test_check_reports_stale_and_orphaned_files(self):
    self.run_main()
    with open(os.path.join(self.root, "games/foo/CMakeLists.txt"), "a") as f:
      f.write("# edited\n")
    orphan = os.path.join(self.root, "games/old/CMakeLists.txt")
    os.makedirs(os.path.dirname(orphan))
    with open(orphan, "w") as f:
      f.write(generator.HEADER)
    code, stderr = self.run_main("--check")
    self.assertEqual(code, 1)
    self.assertIn("stale: games/foo/CMakeLists.txt", stderr)
    self.assertIn("no longer generated: games/old/CMakeLists.txt", stderr)
    self.assertIn("scripts/generate_cmake.sh", stderr)
    self.assertEqual(self.run_main()[0], 0)
    self.assertFalse(os.path.exists(orphan))

  def test_refuses_to_overwrite_hand_written_file(self):
    path = os.path.join(self.root, "games/foo/CMakeLists.txt")
    with open(path, "w") as f:
      f.write("project(hand_written)\n")
    code, stderr = self.run_main()
    self.assertEqual(code, 2)
    self.assertIn("hand-written games/foo/CMakeLists.txt", stderr)
    self.assertFalse(
        os.path.exists(os.path.join(self.root, "games/CMakeLists.txt")))

  def test_reports_skipped_targets(self):
    code, stderr = self.run_main()
    self.assertEqual(code, 0)
    self.assertIn("note: not built by CMake: games/foo/BUILD:foo_gtest "
                  "(needs googletest)", stderr)



if __name__ == "__main__":
  absltest.main()
