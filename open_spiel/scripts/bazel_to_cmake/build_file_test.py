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


import os

from absl.testing import absltest

from open_spiel.scripts.bazel_to_cmake import build_file


def _evaluate(text: str, package_dir: str = "/nonexistent") -> list[
    build_file.Rule]:
  return build_file.evaluate(text, "pkg", package_dir, "pkg/BUILD")


class EvaluateTest(absltest.TestCase):
  """Evaluation of BUILD files with stub rules."""

  def test_records_rules_in_order(self):
    rules = _evaluate("""
load("//some:defs.bzl", "cc_library", my_test = "cc_test")
package(default_visibility = ["//visibility:public"])
licenses(["notice"])
exports_files(["BUILD"])
cc_library(name = "a", srcs = ["a.cc"])
my_test(name = "a_test", deps = [":a"])
py_library(name = "py", srcs = ["py.py"])
""")
    self.assertEqual([(r.kind, r.name) for r in rules],
                     [("cc_library", "a"), ("cc_test", "a_test"),
                      ("py_library", "py")])
    self.assertEqual(rules[1].attrs["deps"], [":a"])
    self.assertEqual(rules[0].package, "pkg")

  def test_select_concatenation(self):
    rules = _evaluate("""
cc_library(
    name = "a",
    deps = [":x"] + select({":on": [":y"], "//conditions:default": []}) + [":z"],
)
""")
    deps = rules[0].attrs["deps"]
    self.assertIsInstance(deps, build_file.Selectable)
    self.assertEqual(deps.parts, (
        [":x"],
        build_file.Select(((":on", [":y"]), ("//conditions:default", []))),
        [":z"],
    ))

  def test_variables_and_comprehensions(self):
    rules = _evaluate("""
GAMES = ["a", "b"]
[cc_library(name = g, srcs = [g + ".cc"]) for g in GAMES]
""")
    self.assertEqual([r.name for r in rules], ["a", "b"])

  def test_opaque_loaded_symbol(self):
    rules = _evaluate("""
load("//x:torch.bzl", "TORCH_COPTS")
cc_library(name = "a", copts = TORCH_COPTS)
""")
    self.assertEqual(rules[0].attrs["copts"], build_file.Opaque("TORCH_COPTS"))

  def test_unknown_rule_is_an_error(self):
    with self.assertRaisesRegex(build_file.Error, "genrule"):
      _evaluate('genrule(name = "g")')

  def test_unknown_loaded_symbol_is_an_error(self):
    with self.assertRaisesRegex(build_file.Error, "my_macro"):
      _evaluate('load("//x:defs.bzl", "my_macro")')

  def test_pytest_test_is_recorded(self):
    rules = _evaluate("""
load("//third_party/py/pytest:pytest_defs.bzl", "pytest_test")
pytest_test(name = "p_test", tests = ["p_test.py"])
""")
    self.assertEqual([(r.kind, r.name) for r in rules],
                     [("pytest_test", "p_test")])
    self.assertIn("pytest_test", build_file.PY_TEST_RULES)

  def test_glob(self):
    root = self.create_tempdir().full_path
    for rel in ("a.txt", "b.txt", "c.cc", "sub/d.txt", "pkg2/BUILD",
                "pkg2/e.txt"):
      path = os.path.join(root, rel)
      os.makedirs(os.path.dirname(path), exist_ok=True)
      with open(path, "w") as f:
        f.write("")
    rules = _evaluate(
        'filegroup(name = "f", srcs = glob(["**/*.txt"], exclude = ["b.txt"]))',
        root)
    srcs = rules[0].attrs["srcs"]
    self.assertIsInstance(srcs, build_file.GlobResult)
    self.assertEqual(list(srcs), ["a.txt", "sub/d.txt"])


if __name__ == "__main__":
  absltest.main()
