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

"""The build units of this package and of the games package import as a DAG.

testing.import_graph reads every import statement from the AST (no module
is imported). A cycle fails with the import sites that close it. Every
`from P import n` with P a package names a submodule of P, and no module
imports a test file. Recorded paths into
the moved modules resolve to the modules' new homes.
"""

import importlib.util
import os
import sys

import pytest

from open_spiel.python.scale import deps
from open_spiel.python.scale import registry
from open_spiel.python.scale.ckpt import layout
from open_spiel.python.scale.core import schedules
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.testing import import_graph

PACKAGE = import_graph.__name__.rsplit(".", 2)[0]


def _packages() -> list[str]:
  """The packages to check: the framework, and the games package when installed.

  Returns:
    The package names.
  """
  out = [PACKAGE]
  if importlib.util.find_spec(deps.GAMES_PACKAGE.split(".")[0]) is not None:
    out.append(deps.GAMES_PACKAGE)
  return out


@pytest.mark.parametrize("package", _packages())
def test_units_import_as_a_dag(package):
  """Each package's units form a DAG (the report shows any cycle's sites)."""
  text, dag = import_graph.report(package)
  assert dag, text


def test_a_cycle_is_reported_with_its_sites(tmp_path):
  """A lazy import back up the layers closes a cycle, and both sites show."""
  pkg = tmp_path / "toy"
  (pkg / "a").mkdir(parents=True)
  (pkg / "b").mkdir()
  for f in ("__init__.py", "a/__init__.py", "b/__init__.py"):
    (pkg / f).write_text("")
  (pkg / "a" / "x.py").write_text("from toy.b import y\n")
  (pkg / "b" / "y.py").write_text(
      "def f():\n  from toy.a import x\n  return x\n"
  )
  edges = import_graph.edges(str(pkg), "toy")
  units = import_graph.units(str(pkg))
  assert import_graph.cycles(units, edges) == [["a/", "b/"]]
  assert {e.site for e in edges} == {
      os.path.join("a", "x.py") + ":1",
      os.path.join("b", "y.py") + ":2",
  }


def test_a_library_import_of_a_test_module_is_refused(tmp_path):
  """A library module may not import the testing subpackage."""
  pkg = tmp_path / "toy"
  (pkg / "testing").mkdir(parents=True)
  (pkg / "__init__.py").write_text("")
  (pkg / "testing" / "__init__.py").write_text("")
  (pkg / "testing" / "helper.py").write_text("")
  (pkg / "lib.py").write_text("from toy.testing import helper\n")
  with pytest.raises(ValueError, match="test-only"):
    import_graph.edges(str(pkg), "toy")


def test_from_imports_name_submodules():
  """Every `from P import n` (tests included) names a submodule of P."""
  trees = {p: import_graph.package_dir_of(p) for p in _packages()}
  bad = import_graph.attribute_imports(trees)
  assert not bad, "\n".join(bad)


@pytest.mark.parametrize("package", _packages())
def test_no_module_imports_a_test_file(package):
  """What tests share lives in fixtures modules, never in a test file."""
  bad = import_graph.test_file_imports(
      import_graph.package_dir_of(package), package
  )
  assert not bad, "\n".join(bad)


def test_an_import_of_a_test_file_is_reported(tmp_path):
  """A test importing another test file is reported; a fixtures module not."""
  pkg = tmp_path / "toy"
  (pkg / "tests").mkdir(parents=True)
  (pkg / "__init__.py").write_text("")
  (pkg / "tests" / "__init__.py").write_text("")
  (pkg / "tests" / "a_test.py").write_text("X = 1\n")
  (pkg / "tests" / "fixtures.py").write_text("X = 1\n")
  (pkg / "tests" / "b_test.py").write_text(
      "from toy.tests import fixtures\nfrom toy.tests import a_test\n"
  )
  site = os.path.join("tests", "b_test.py")
  assert import_graph.test_file_imports(str(pkg), "toy") == [
      f"toy:{site}:2: imports test file toy.tests.a_test"
  ]


def test_an_attribute_import_from_a_package_is_reported(tmp_path):
  """An attribute of a package's __init__ is reported; modules are not."""
  pkg = tmp_path / "toy"
  (pkg / "sub" / "tests").mkdir(parents=True)
  (pkg / "__init__.py").write_text("THING = 1\n")
  (pkg / "sub" / "__init__.py").write_text("from toy.sub.mod import f\n")
  (pkg / "sub" / "mod.py").write_text("def f():\n  return 1\n")
  (pkg / "sub" / "tests" / "mod_test.py").write_text(
      "from toy import sub\n"  # a subpackage
      "from toy.sub import mod\n"  # a submodule
      "from toy.sub.mod import f\n"  # a module, not a package: any name
      "from toy import THING\n"  # an attribute of toy/__init__.py
      "from os import path\n"  # outside the tree
      "from .. import mod as m\n"  # relative: toy.sub.mod
      "from .. import f as g\n"  # relative: an attribute of toy.sub
  )
  got = import_graph.attribute_imports({"toy": str(pkg)})
  site = os.path.join("sub", "tests", "mod_test.py")
  assert got == [
      f"toy:{site}:4: from toy import THING (THING is not a module of toy)",
      f"toy:{site}:7: from toy.sub import f (f is not a module of toy.sub)",
  ]


def test_main_exits_1_on_an_attribute_import(tmp_path, monkeypatch, capsys):
  """Running import_graph as a script fails on an attribute import."""
  pkg = tmp_path / "toyattr"
  pkg.mkdir()
  (pkg / "__init__.py").write_text("X = 1\n")
  (pkg / "a.py").write_text("from toyattr import X\n")
  monkeypatch.syspath_prepend(str(tmp_path))
  assert import_graph.main(["prog", "toyattr"]) == 1
  assert "from toyattr import X" in capsys.readouterr().out
  (pkg / "a.py").write_text("import toyattr\n")
  assert import_graph.main(["prog", "toyattr"]) == 0


def test_recorded_paths_into_moved_modules_resolve(monkeypatch):
  """A path recorded before a move resolves to the module's new home."""
  monkeypatch.setattr(registry, "_RESOLVERS", [])
  pkg = PACKAGE
  assert registry.resolve_path(f"{pkg}.learners.schedules:linear") == (
      f"{pkg}.core.schedules:linear"
  )
  assert registry.resolve_path(f"{pkg}.configs.common") == (
      f"{pkg}.league.config_common"
  )
  assert registry.resolve_path(f"{pkg}.configs.commonx:f") == (
      f"{pkg}.configs.commonx:f"
  )
  assert registry.resolve_path(f"{pkg}.core.schedules:x") == (
      f"{pkg}.core.schedules:x"
  )
  for old, new in registry.MOVED_MODULES.items():
    assert importlib.util.find_spec(f"{pkg}.{new}") is not None, new
    parent, _, name = old.rpartition(".")
    old_file = os.path.join(
        import_graph.package_dir_of(f"{pkg}.{parent}"), name + ".py"
    )
    assert not os.path.exists(old_file), old
  assert registry.load_object(f"{pkg}.learners.schedules:__name__") == (
      schedules.__name__
  )
  assert registry.load_object(f"{pkg}.league.layout:__name__") == (
      layout.__name__
  )
  assert registry.load_object(f"{pkg}.configs.common:__name__") == (
      config_common.__name__
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
