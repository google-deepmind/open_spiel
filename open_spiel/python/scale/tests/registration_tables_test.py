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

"""Every name a registry resolves loads with no import by the caller.

An artifact or a config records names (an obs_format, a torso or head kind,
a catalog network, a loss, a policy architecture, a belief sampler, a
learner kind). A module that registers such a name when it is imported is
reachable only if the registry knows to import it on a lookup: through its
built-in module list, its lazy name -> module table, the game's declared
extras or components, or a module those always import. A registering module
that none of them names loads only when some caller happens to import it
first, so a released checkpoint fails to load in a fresh process ("unknown
encoder hearts_card_planes").

The test parses every non-test module of the package, finds each top-level
call to a register function of another package module, and requires (a)
that the (registry, function) pair is one REGISTRIES judges or one
NOT_RESOLVED_BY_NAME explains, and (b) for a judged pair, that the calling
module is in that registry's lookup set. A seeded negative control shows
the scan reports a module missing from a table.
"""

import ast
import importlib
import json
import os
import pathlib
import sys

import pytest

from open_spiel.python.scale.belief import registry as belief_registry
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.networks import actor_critic
from open_spiel.python.scale.networks import catalog
from open_spiel.python.scale.policies import registry as policy_registry

# The judged lookup sets include the registries' own private module lists.
# pylint: disable=protected-access

PKG_DIR = pathlib.Path(os.path.abspath(enc_registry.__file__)).parents[1]
# The package's dotted name, from the registry's own __name__ (encoders/
# registry.py is two components below it): the folder name alone is wrong
# wherever the package is a subpackage (open_spiel.python.scale).
PKG = enc_registry.__name__.rsplit(".", 2)[0]


def _module_name(path: pathlib.Path, root: pathlib.Path, pkg: str) -> str:
  """The dotted name of `path`, a module of package `pkg` at folder `root`.

  Args:
    path: a .py file under root.
    root: the package folder.
    pkg: the package's dotted name.

  Returns:
    The module name.
  """
  parts = list(path.relative_to(root).with_suffix("").parts)
  if parts[-1] == "__init__":
    parts.pop()
  return ".".join([pkg] + parts)


def _is_test(path: pathlib.Path) -> bool:
  """Whether `path` (relative to the package) is a test module.

  Args:
    path: the module's path.

  Returns:
    True for a tests/ module, a *_test.py file or conftest.py.
  """
  return (
      "tests" in path.parts
      or path.name.endswith("_test.py")
      or (path.name == "conftest.py")
  )


def _aliases(tree: ast.Module, module: str) -> dict[str, str]:
  """Top-level import aliases of a module: local name -> module name.

  Args:
    tree: the module's AST.
    module: the module's dotted name (for relative imports).

  Returns:
    {local name: module name}.
  """
  out = {}
  for node in tree.body:
    if isinstance(node, ast.Import):
      for a in node.names:
        out[a.asname or a.name.split(".")[0]] = (
            a.name if a.asname else (a.name.split(".")[0])
        )
    elif isinstance(node, ast.ImportFrom):
      base = node.module or ""
      if node.level:
        pkg = module.split(".")
        base = ".".join(pkg[: len(pkg) - node.level] + ([base] if base else []))
      for a in node.names:
        out[a.asname or a.name] = f"{base}.{a.name}"
  return out


def _top_level_callees(tree: ast.Module):
  """What the module calls when imported.

  The func of each call outside a def or class body, and each decorator of a
  def or class (a bare `@reg.register` is called with the def).

  Args:
    tree: the module's AST.

  Yields:
    Each callee expression.
  """
  stack = list(tree.body)
  while stack:
    node = stack.pop()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      for deco in node.decorator_list:
        yield deco.func if isinstance(deco, ast.Call) else deco
        stack.extend(ast.iter_child_nodes(deco))
      continue
    if isinstance(node, ast.Lambda):
      continue
    if isinstance(node, ast.Call):
      yield node.func
    stack.extend(ast.iter_child_nodes(node))


def register_calls(root: pathlib.Path, pkg: str = PKG):
  """(calling module, registry module, function) of each import-time call.

  Only calls to a register* function of another module of the package rooted
  at `root` count (a module calling its own register function defines that
  registry): `alias.register*(...)`, a bare `register*(...)` imported from
  the registry module, and either one as a decorator.

  Args:
    root: the package folder.
    pkg: the package's dotted name.

  Yields:
    (module, registry module, function name).
  """
  for path in sorted(root.rglob("*.py")):
    if _is_test(path.relative_to(root)):
      continue
    module = _module_name(path, root, pkg)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _aliases(tree, module)
    for f in _top_level_callees(tree):
      if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        target, fn = aliases.get(f.value.id), f.attr  # registry.register(...)
      elif isinstance(f, ast.Name) and f.id in aliases:
        # register(...) after `from <registry> import register [as name]`.
        target, _, fn = aliases[f.id].rpartition(".")
      else:
        continue
      if (
          fn.startswith("register")
          and target
          and target.startswith(pkg + ".")
          and target != module
      ):
        yield module, target, fn


def static_imports(module: str) -> set[str]:
  """The package modules `module` imports at its top level.

  Args:
    module: a package module's dotted name.

  Returns:
    The dotted names.
  """
  assert module.startswith(PKG + "."), module
  path = PKG_DIR / pathlib.Path(*module[len(PKG) + 1 :].split("."))
  path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
  tree = ast.parse(path.read_text(encoding="utf-8"))
  out = set()
  for node in tree.body:
    if isinstance(node, ast.Import):
      out |= {a.name for a in node.names}
    elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
      # `from a.b import c`: a.b, and a.b.c when c is a module.
      out |= {node.module} | {f"{node.module}.{a.name}" for a in node.names}
  return {m for m in out if m.startswith(PKG + ".")}


def _game_declared() -> set[str]:
  """Modules the env registry imports for a game: its extras and components.

  Returns:
    The dotted names.
  """
  out = set()
  for g in env_registry.games():
    out |= set(g.extras or ())
    out |= {c[2] for c in (g.components or ())}
  return out


def _network_policy_closure() -> set[str]:
  """The built-in policy modules and what they import.

  A network config is built only through a network policy, a built-in
  policy architecture; what those modules import is always loaded first.

  Returns:
    The dotted names.
  """
  out = set()
  for mod in policy_registry._BUILTIN_MODULES:
    out |= {mod} | static_imports(mod)
  return out


def _kinds() -> set[str]:
  """The league kind modules (league.kinds imports them all).

  Returns:
    The dotted names.
  """
  return static_imports(f"{PKG}.league.kinds")


def _telemetry() -> set[str]:
  """The training telemetry modules train.py imports by name.

  Returns:
    The dotted names.
  """
  train = importlib.import_module(f"{PKG}.train")
  return {f"{PKG}.{m}" for m in train.TRAINING_TELEMETRY.values()}


#: (registry module, register function) -> the modules a lookup in that
#: registry imports (or that are always imported before one).
REGISTRIES = {
    (f"{PKG}.encoders.registry", "register"): lambda: (
        set(enc_registry._BUILTIN_MODULES) | set(enc_registry.LAZY.values())
    ),
    (f"{PKG}.encoders.registry", "register_family"): lambda: (
        set(enc_registry._BUILTIN_MODULES) | set(enc_registry.LAZY.values())
    ),
    (f"{PKG}.networks.actor_critic", "register_torso"): lambda: (
        set(actor_critic.LAZY_TORSOS.values()) | _network_policy_closure()
    ),
    (f"{PKG}.networks.actor_critic", "register_policy_head"): lambda: (
        set(actor_critic.LAZY_POLICY_HEADS.values()) | _network_policy_closure()
    ),
    (f"{PKG}.networks.catalog", "register"): lambda: set(catalog.LAZY.values()),
    (f"{PKG}.learners.pg", "register_loss"): lambda: set(
        pg.LAZY_LOSSES.values()
    ),
    (f"{PKG}.policies.registry", "register"): lambda: (
        set(policy_registry._BUILTIN_MODULES)
        | set(policy_registry.LAZY.values())
        | _game_declared()
    ),
    (f"{PKG}.belief.registry", "register"): lambda: (
        set(belief_registry._BUILTIN_MODULES)
        | set(belief_registry._LAZY.values())
        | _game_declared()
    ),
    # A translator runs before every policy build, so its registrar must be
    # a module every build loads.
    (f"{PKG}.policies.registry", "register_kwargs_translator"): lambda: set(
        policy_registry._BUILTIN_MODULES
    ),
    (f"{PKG}.eval.registry", "register"): _game_declared,
    (f"{PKG}.league.kinds.base", "register_kind"): _kinds,
    (f"{PKG}.league.runner", "register_run_hook"): _telemetry,
}

#: (registry module, register function) -> why a module registering there
#: need not be reachable from a recorded name. Adding a pair is a decision:
#: say why nothing recorded resolves through it.
NOT_RESOLVED_BY_NAME = {
    (f"{PKG}.ckpt.artifact", "register_meta_extension"): (
        "a manifest extension key is validated when its module is loaded and "
        "is otherwise read as plain JSON (check_extensions allow_unknown); its "
        "only registrar, league.siblings, is imported by the league that "
        "writes and reads it"
    ),
    (f"{PKG}.core.auxiliary", "register_aux_key"): (
        "an aux output key a policy module declares for its own outputs; it "
        "is registered by the module that produces the key"
    ),
    (f"{PKG}.league.slot_factories", "register"): (
        "testing fixtures only (testing/host_operator_fixtures), registered "
        "for the session by the package's conftest"
    ),
}


def unresolved(calls, registries, explained):
  """Failure lines: unjudged register pairs and modules no table names.

  Args:
    calls: (module, registry module, function) triples (register_calls).
    registries: REGISTRIES-shaped (pair -> lookup-set function).
    explained: NOT_RESOLVED_BY_NAME-shaped (pair -> reason).

  Returns:
    The sorted failure lines; empty when every registration is reachable.
  """
  bad, sets = [], {}
  for module, reg, fn in calls:
    pair = (reg, fn)
    if pair in explained:
      continue
    if pair not in registries:
      bad.append(
          f"{module}: {reg}.{fn} is neither in REGISTRIES nor in "
          "NOT_RESOLVED_BY_NAME"
      )
      continue
    if pair not in sets:
      sets[pair] = registries[pair]()
    if module not in sets[pair]:
      bad.append(
          f"{module} registers through {reg}.{fn} only when imported: "
          "add it to that registry's built-in or lazy table"
      )
  return sorted(set(bad))


def test_every_registering_module_is_in_a_lookup_table():
  calls = list(register_calls(PKG_DIR))
  assert calls, "the scan found no registration (the scan is broken)"
  bad = unresolved(calls, REGISTRIES, NOT_RESOLVED_BY_NAME)
  assert not bad, bad


def test_the_scan_sees_the_card_move_transformer_family():
  modules = {m for m, _, _ in register_calls(PKG_DIR)}
  for m in (
      "encoders.hearts_card_planes",
      "encoders.euchre_card_planes",
      "networks.card_move_transformer",
      "learners.ppo_magnet",
  ):
    assert f"{PKG}.{m}" in modules, m


def test_every_lazy_table_entry_registers_its_name():
  # The inverse: a lazy entry names a module that really registers the name.
  for name, mod in enc_registry.LAZY.items():
    importlib.import_module(mod)
    assert enc_registry.known(name), name
  for kind, mod in actor_critic.LAZY_TORSOS.items():
    importlib.import_module(mod)
    assert kind in actor_critic.TORSOS, kind
  for kind, mod in actor_critic.LAZY_POLICY_HEADS.items():
    importlib.import_module(mod)
    assert kind in actor_critic.POLICY_HEADS, kind
  for name, mod in catalog.LAZY.items():
    importlib.import_module(mod)
    assert name in catalog.NAMED, name
  for loss, mod in pg.LAZY_LOSSES.items():
    importlib.import_module(mod)
    assert pg.extension(loss) is not None, loss


#: In a fresh interpreter that imports only the registries, every lazy name is
#: looked up through its registry's own lookup, and the module the table
#: names must be what loaded it. argv[1] is the package's import name.
FRESH_LOOKUP = """
import json, sys
pkg = sys.argv[1]
mods = lambda: sorted(m for m in sys.modules if m.startswith(pkg + "."))
import importlib
enc = importlib.import_module(pkg + ".encoders.registry")
ac = importlib.import_module(pkg + ".networks.actor_critic")
cat = importlib.import_module(pkg + ".networks.catalog")
pg = importlib.import_module(pkg + ".learners.pg")
before = mods()
out = {"before": before, "missing": [], "loaded": {}}
def check(table, name, mod, lookup):
  ok = lookup()
  if not ok:
    out["missing"].append(f"{table}:{name}")
  out["loaded"][f"{table}:{name}"] = mod in sys.modules
for name, mod in enc.LAZY.items():
  check("encoders", name, mod, lambda: enc.get(name) is not None)
for kind, mod in ac.LAZY_TORSOS.items():
  check("torsos", kind, mod, lambda: (
      ac._load_lazy(kind, ac.TORSOS, ac.LAZY_TORSOS), kind in ac.TORSOS)[1])
for kind, mod in ac.LAZY_POLICY_HEADS.items():
  check("policy_heads", kind, mod, lambda: (
      ac._load_lazy(kind, ac.POLICY_HEADS, ac.LAZY_POLICY_HEADS),
      kind in ac.POLICY_HEADS)[1])
for name, mod in cat.LAZY.items():
  check("catalog", name, mod, lambda: cat.get(name) is not None)
for loss, mod in pg.LAZY_LOSSES.items():
  check("losses", loss, mod, lambda: pg.extension(loss) is not None)
print(json.dumps(out))
"""


def _fresh_lookup():
  """Runs FRESH_LOOKUP in a fresh interpreter.

  Returns:
    Its JSON report.
  """
  r = subprocess_env.run_python(
      ["-c", FRESH_LOOKUP, subprocess_env.PACKAGE],
      env={"JAX_PLATFORMS": "cpu"},
      timeout=300,
  )
  assert r.returncode == 0, r.stdout + r.stderr
  return json.loads(r.stdout.strip().splitlines()[-1])


def test_every_lazy_name_resolves_through_its_registry_in_a_fresh_process():
  out = _fresh_lookup()
  assert not out["missing"], out["missing"]
  assert out["loaded"] and all(out["loaded"].values()), out["loaded"]
  lazy_modules = set().union(*(
      t.values()
      for t in (
          enc_registry.LAZY,
          actor_critic.LAZY_TORSOS,
          actor_critic.LAZY_POLICY_HEADS,
          catalog.LAZY,
          pg.LAZY_LOSSES,
      )
  ))
  # the registries alone import none of the lazily registered modules
  for m in lazy_modules:
    assert m.startswith(PKG + "."), m
    assert m not in out["before"], m


def test_negative_control_reports_a_module_missing_from_its_table(tmp_path):
  root = tmp_path / PKG
  (root / "encoders").mkdir(parents=True)
  (root / "__init__.py").write_text("")
  (root / "encoders" / "__init__.py").write_text("")
  (root / "encoders" / "seeded_planes.py").write_text(
      f"from {PKG}.encoders import registry\n"
      "ENC = registry.register(object())\n"
  )
  (root / "encoders" / "registry.py").write_text(
      "def register(e):\n  return e\n"
  )
  (root / "encoders" / "unknown_kind.py").write_text(
      f"from {PKG}.encoders import registry as r\nr.register_widget('w')\n"
  )
  (root / "encoders" / "decorated.py").write_text(
      f"from {PKG}.encoders import registry\n"
      "@registry.register\n"
      "def enc(info):\n  return info\n"
  )
  (root / "encoders" / "bare_name.py").write_text(
      f"from {PKG}.encoders.registry import register as reg\n"
      "ENC = reg(object())\n"
  )
  calls = list(register_calls(root))
  for m in ("seeded_planes", "decorated", "bare_name"):
    assert (
        f"{PKG}.encoders.{m}",
        f"{PKG}.encoders.registry",
        "register",
    ) in calls, m
  bad = unresolved(calls, REGISTRIES, NOT_RESOLVED_BY_NAME)
  for m in ("seeded_planes", "decorated", "bare_name"):
    assert any(b.startswith(f"{PKG}.encoders.{m} registers") for b in bad), (
        m,
        bad,
    )
  assert any("register_widget is neither" in b for b in bad), bad


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
