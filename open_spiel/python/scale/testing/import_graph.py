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

"""The import graph of a package's build units, read from the AST.

A build unit is what one build target holds:

  * each subpackage directory (`league`, `envs/gin`, ...), one BUILD file
    each; a directory in MERGED belongs to its parent's unit instead;
  * each module at the package's top level (`train`, `plugins`, ...,
    `__init__`), one target each in the top-level BUILD file.

Library modules only: test files (`*_test.py`, `conftest.py`, anything
under a `tests/` directory) and the `testing/` subpackage are left out, and
a library module that imports one of them is an error. Every import
statement counts, at module level or inside a function, so a lazy import is
an edge too. A registry that imports a module by a computed name (a string
table) is not an edge: it resolves late, after the importer is loaded. The
parent packages' `__init__.py` files that Python runs before a submodule are
not edges either; an explicit `import open_spiel.python.scale` is.

A second rule covers every module, tests included: `from P import n`, with
`P` a package (a directory with `__init__.py`) of one of the checked packages,
must name a submodule or subpackage of `P`, never an attribute of
`P/__init__.py`. A build that keeps one target per module cannot tell an
attribute from a missing dependency. A third: no module imports a test file
(`*_test.py`); what tests share lives in a fixtures module.

  python -m open_spiel.python.scale.testing.import_graph [package ...]

prints each package's units in layer order (a unit imports only units of
earlier layers) and its edges, then every `from P import n` that names no
submodule and every import of a test file, and exits 1 if any package has a
cycle or any such import.
"""

import ast
import collections
import dataclasses
import importlib.util
import os
import sys

_EXCLUDED_DIRS = frozenset({"tests", "testing", "__pycache__"})
#: Subpackage directories that are part of their parent's unit, with why.
MERGED = {
    # The learner kinds and the league loop call each other (a kind runs
    # the loop's commit and sampler; the loop dispatches on the kinds).
    "league/kinds": "league",
}


@dataclasses.dataclass(frozen=True)
class Edge:
  """One import of a module in unit `dst` by a module in unit `src`.

  Attributes:
    src: The importing unit.
    dst: The imported unit.
    site: `path:line` of the import statement, relative to the package.
  """

  src: str
  dst: str
  site: str


def _is_library(rel: str) -> bool:
  """Whether a package file is library code (not a test or conftest).

  Args:
    rel: The package-relative path.

  Returns:
    True for library code.
  """
  parts = rel.split(os.sep)
  if any(p in _EXCLUDED_DIRS for p in parts[:-1]):
    return False
  name = parts[-1]
  return not (name.endswith("_test.py") or name == "conftest.py")


def _unit(rel: str) -> str:
  """The import-graph unit of a file: its (merged) folder, or its module.

  Args:
    rel: The package-relative path.

  Returns:
    The unit.
  """
  parts = rel.split(os.sep)
  if len(parts) == 1:
    return parts[0][: -len(".py")]
  unit = "/".join(parts[:-1])
  while unit in MERGED:
    unit = MERGED[unit]
  return unit + "/"


def library_modules(package_dir: str) -> list[str]:
  """Returns the package-relative paths of the library modules, sorted.

  Args:
    package_dir: The package's directory.

  Returns:
    Paths of `.py` files relative to `package_dir`.
  """
  out = []
  for root, dirs, files in os.walk(package_dir):
    dirs[:] = sorted(d for d in dirs if d not in _EXCLUDED_DIRS)
    for f in files:
      if f.endswith(".py"):
        rel = os.path.relpath(os.path.join(root, f), package_dir)
        if _is_library(rel):
          out.append(rel)
  return sorted(out)


def _module_file(package_dir: str, parts: list[str]) -> str | None:
  """Returns the package-relative file of module `parts`, or None.

  Args:
    package_dir: The package folder.
    parts: The module path.

  Returns:
    The file, or None.
  """
  base = os.path.join(package_dir, *parts)
  if os.path.isfile(base + ".py"):
    return os.path.relpath(base + ".py", package_dir)
  init = os.path.join(base, "__init__.py")
  if os.path.isfile(init):
    return os.path.relpath(init, package_dir)
  return None


def _imported_modules(
    node: ast.AST, rel: str, package: str, package_dir: str
) -> list[list[str]]:
  """Returns the package-relative module paths one import statement loads.

  Args:
    node: The import statement.
    rel: The importing file.
    package: The package name.
    package_dir: The package folder.

  Returns:
    The module paths.
  """
  root = package.split(".")
  if isinstance(node, ast.Import):
    names = [a.name.split(".") for a in node.names]
    return [n[len(root) :] for n in names if n[: len(root)] == root]
  assert isinstance(node, ast.ImportFrom)
  if node.level:
    here = rel.split(os.sep)[:-1]
    if node.level > 1:
      here = here[: len(here) - (node.level - 1)]
    stem = here + (node.module.split(".") if node.module else [])
  else:
    parts = (node.module or "").split(".")
    if parts[: len(root)] != root:
      return []
    stem = parts[len(root) :]
  out = []
  for alias in node.names:
    sub = stem + [alias.name]
    if alias.name != "*" and _module_file(package_dir, sub) is not None:
      out.append(sub)
    else:
      out.append(stem)
  return out


def edges(package_dir: str, package: str) -> list[Edge]:
  """Returns every import edge between different units of a package.

  Args:
    package_dir: The package's directory.
    package: The package's dotted import name, as its code imports it.

  Returns:
    One Edge per import site whose target is a library module in another
    unit, sorted.

  Raises:
    ValueError: An import names a module of the package that does not exist.
  """
  out = set()
  for rel in library_modules(package_dir):
    with open(os.path.join(package_dir, rel), encoding="utf-8") as f:
      tree = ast.parse(f.read(), filename=rel)
    src = _unit(rel)
    for node in ast.walk(tree):
      if not isinstance(node, (ast.Import, ast.ImportFrom)):
        continue
      for parts in _imported_modules(node, rel, package, package_dir):
        target = (
            "__init__.py" if not parts else _module_file(package_dir, parts)
        )
        if target is None:
          raise ValueError(
              f"{rel}:{node.lineno} imports {'.'.join([package] + parts)},"
              " which is not a module of the package"
          )
        if not _is_library(target):
          raise ValueError(
              f"{rel}:{node.lineno} imports test-only module {target}"
          )
        dst = _unit(target)
        if dst != src:
          out.add(Edge(src, dst, f"{rel}:{node.lineno}"))
  return sorted(out, key=lambda e: (e.src, e.dst, e.site))


def units(package_dir: str) -> list[str]:
  """Returns the units of a package, sorted.

  Args:
    package_dir: The package's directory.

  Returns:
    The unit names.
  """
  return sorted({_unit(rel) for rel in library_modules(package_dir)})


def cycles(all_units: list[str], all_edges: list[Edge]) -> list[list[str]]:
  """Returns the strongly connected components with more than one unit.

  Args:
    all_units: The units.
    all_edges: The edges between them.

  Returns:
    Each cyclic component as a sorted list of units; empty for a DAG.
  """
  succ = collections.defaultdict(set)
  for e in all_edges:
    succ[e.src].add(e.dst)
  index, low, stack, on_stack, out = {}, {}, [], set(), []

  def visit(v: str) -> None:
    index[v] = low[v] = len(index)
    stack.append(v)
    on_stack.add(v)
    for w in sorted(succ[v]):
      if w not in index:
        visit(w)
        low[v] = min(low[v], low[w])
      elif w in on_stack:
        low[v] = min(low[v], index[w])
    if low[v] == index[v]:
      comp = []
      while True:
        w = stack.pop()
        on_stack.discard(w)
        comp.append(w)
        if w == v:
          break
      if len(comp) > 1:
        out.append(sorted(comp))

  for v in sorted(all_units):
    if v not in index:
      visit(v)
  return sorted(out)


def layers(all_units: list[str], all_edges: list[Edge]) -> list[list[str]]:
  """Returns the units of a DAG in layers, leaves first.

  Args:
    all_units: The units.
    all_edges: The edges between them, which must form a DAG.

  Returns:
    Lists of units; each unit depends only on units of earlier lists.

  Raises:
    ValueError: The edges have a cycle.
  """
  deps = {u: set() for u in all_units}
  for e in all_edges:
    deps[e.src].add(e.dst)
  done, out = set(), []
  while len(done) < len(deps):
    layer = sorted(u for u, d in deps.items() if u not in done and d <= done)
    if not layer:
      raise ValueError(f"cycle among {sorted(set(deps) - done)}")
    out.append(layer)
    done.update(layer)
  return out


def package_dir_of(package: str) -> str:
  """Returns the directory of an importable package, without importing it.

  Args:
    package: A dotted package name.

  Returns:
    The package's directory.

  Raises:
    ModuleNotFoundError: The package is not importable.
  """
  spec = importlib.util.find_spec(package)
  if spec is None or not spec.submodule_search_locations:
    raise ModuleNotFoundError(f"no package {package!r}")
  return list(spec.submodule_search_locations)[0]


def report(package: str) -> tuple[str, bool]:
  """Returns a package's graph as text, and whether it is a DAG.

  Args:
    package: A dotted package name, as the package's code imports itself.

  Returns:
    The text and True when the package has no cycle.
  """
  package_dir = package_dir_of(package)
  all_units = units(package_dir)
  all_edges = edges(package_dir, package)
  lines = [f"## {package}", ""]
  found = cycles(all_units, all_edges)
  if found:
    lines.append("CYCLES:")
    for comp in found:
      lines.append("  " + " <-> ".join(comp))
      for e in all_edges:
        if e.src in comp and e.dst in comp:
          lines.append(f"    {e.src} -> {e.dst}  {e.site}")
    return "\n".join(lines), False
  lines.append("Layers (a unit imports only units of earlier layers):")
  for i, layer in enumerate(layers(all_units, all_edges)):
    lines.append(f"  {i}: {' '.join(layer)}")
  lines += ["", "Edges (unit -> unit: import sites):"]
  by_pair = collections.defaultdict(list)
  for e in all_edges:
    by_pair[(e.src, e.dst)].append(e.site)
  for (src, dst), sites in sorted(by_pair.items()):
    lines.append(f"  {src} -> {dst}: {len(sites)}  e.g. {sites[0]}")
  return "\n".join(lines), True


def all_modules(package_dir: str) -> list[str]:
  """Returns the package-relative paths of every `.py` file, tests included.

  Args:
    package_dir: The package's directory.

  Returns:
    Sorted paths relative to `package_dir`.
  """
  out = []
  for root, dirs, files in os.walk(package_dir):
    dirs[:] = sorted(d for d in dirs if d != "__pycache__")
    for f in files:
      if f.endswith(".py"):
        out.append(os.path.relpath(os.path.join(root, f), package_dir))
  return sorted(out)


def _package_of(
    trees: dict[str, str], node: ast.ImportFrom, package: str, rel: str
) -> tuple[str, list[str]] | None:
  """Returns (package, path parts) of the module an ImportFrom reads from.

  Args:
    trees: Checked packages, dotted name to directory.
    node: The statement.
    package: The importing module's package.
    rel: The importing module's path relative to that package.

  Returns:
    The checked package holding the module `node` imports from and the
    module's parts relative to it, or None when it is outside every
    checked package.
  """
  if node.level:
    here = rel.split(os.sep)[:-1]
    if node.level > 1:
      here = here[: len(here) - (node.level - 1)]
    return package, here + (node.module.split(".") if node.module else [])
  parts = (node.module or "").split(".")
  for name in sorted(trees, key=len, reverse=True):
    root = name.split(".")
    if parts[: len(root)] == root:
      return name, parts[len(root) :]
  return None


def attribute_imports(trees: dict[str, str]) -> list[str]:
  """Returns every `from P import n` whose `n` is not a module of package P.

  Args:
    trees: The packages to check, dotted import name to directory. Every
      module of each is read, tests included; an import from any of them
      is checked.

  Returns:
    One `package:path:line: from P import n` line per offending name,
    sorted. Imports from modules (not packages), from outside the trees
    and `import *` are not checked.
  """
  out = []
  for package, package_dir in sorted(trees.items()):
    for rel in all_modules(package_dir):
      with open(os.path.join(package_dir, rel), encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=rel)
      for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
          continue
        found = _package_of(trees, node, package, rel)
        if found is None:
          continue
        owner, parts = found
        pkg_dir = os.path.join(trees[owner], *parts)
        if not os.path.isfile(os.path.join(pkg_dir, "__init__.py")):
          continue  # a module, or not in the tree (edges() reports that)
        src = ".".join([owner] + parts) if parts else owner
        for alias in node.names:
          if alias.name == "*":
            continue
          if _module_file(pkg_dir, [alias.name]) is None:
            out.append(
                f"{package}:{rel}:{node.lineno}: from {src} import"
                f" {alias.name} ({alias.name} is not a module of {src})"
            )
  return sorted(out)


def test_file_imports(package_dir: str, package: str) -> list[str]:
  """Returns every import of a test file (`*_test.py`) in a package.

  Args:
    package_dir: The package's directory.
    package: The package's dotted import name.

  Returns:
    One `package:path:line: imports <module>` line per site, sorted.
  """
  out = []
  for rel in all_modules(package_dir):
    with open(os.path.join(package_dir, rel), encoding="utf-8") as f:
      tree = ast.parse(f.read(), filename=rel)
    for node in ast.walk(tree):
      if not isinstance(node, (ast.Import, ast.ImportFrom)):
        continue
      for parts in _imported_modules(node, rel, package, package_dir):
        target = _module_file(package_dir, parts) if parts else None
        if target is not None and target.endswith("_test.py"):
          out.append(
              f"{package}:{rel}:{node.lineno}: imports test file"
              f" {'.'.join([package] + parts)}"
          )
  return sorted(out)


def main(argv: list[str]) -> int:
  """Prints each package's graph and its attribute imports.

  Args:
    argv: [program, package...] (default: this package).

  Returns:
    The exit code: 1 if any package has a cycle, any `from P import n`
    names no submodule of P or any module imports a test file, else 0.
  """
  packages = argv[1:] or [__name__.rsplit(".", 2)[0]]
  ok = True
  for package in packages:
    text, dag = report(package)
    print(text + "\n")
    ok = ok and dag
  bad = attribute_imports({p: package_dir_of(p) for p in packages})
  if bad:
    print("FROM-IMPORTS THAT NAME NO SUBMODULE:")
    print("\n".join("  " + b for b in bad))
    ok = False
  else:
    print(
        f"Every `from P import n` names a submodule ({len(packages)} packages)."
    )
  tests = []
  for package in packages:
    tests += test_file_imports(package_dir_of(package), package)
  if tests:
    print("IMPORTS OF TEST FILES:")
    print("\n".join("  " + t for t in tests))
    ok = False
  else:
    print("No module imports a test file.")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main(sys.argv))
