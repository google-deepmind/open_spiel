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

"""registry: relative paths, package aliases, object aliases and unpickling.

Each with a known answer: a path or pickle recorded under another name of
this package (or relative to it) resolves to the object this install has, and
no module is altered to make it so.
"""

import dataclasses
import pickle
import sys

import pytest

from open_spiel.python.scale import registry
from open_spiel.python.scale.envs import kuhn

PKG = registry.PACKAGE
ALIAS = "acme_recorded_pkg"


@pytest.fixture(name="clean")
def _clean(monkeypatch):
  """Empty resolver, package alias and object alias tables for one test."""
  monkeypatch.setattr(registry, "_RESOLVERS", [])
  monkeypatch.setattr(registry, "_PACKAGE_ALIASES", [])
  monkeypatch.setattr(registry, "_OBJECTS", {})


@dataclasses.dataclass(frozen=True)
class Point:
  """A picklable class of this test module."""

  x: int


def _renamed(blob: bytes, old: str, new: str) -> bytes:
  """A protocol-0 pickle with module `old` renamed to `new` in its globals.

  Args:
    blob: the protocol-0 pickle.
    old: the module name it records.
    new: the name to record instead.

  Returns:
    The edited pickle.
  """
  assert old.encode() + b"\n" in blob, (old, blob)
  return blob.replace(old.encode() + b"\n", new.encode() + b"\n")


def test_package_is_the_import_name():
  assert PKG == registry.__name__.rpartition(".")[0]
  assert sys.modules[PKG + ".registry"] is registry


def test_a_relative_path_names_a_module_of_this_package(clean):
  del clean
  assert registry.absolute_path(".envs.kuhn:KuhnEnv") == (
      f"{PKG}.envs.kuhn:KuhnEnv"
  )
  assert registry.absolute_path("acme.mod:f") == "acme.mod:f"
  assert registry.resolve_path(".envs.kuhn") == f"{PKG}.envs.kuhn"
  assert registry.load_object(".envs.kuhn:KuhnEnv") is kuhn.KuhnEnv
  seen = []
  registry.register_path_resolver(seen.append)  # returns None: no answer
  registry.resolve_path(".envs.kuhn:KuhnEnv")
  assert seen == [f"{PKG}.envs.kuhn:KuhnEnv"]  # resolvers see it absolute
  for bad in (".", ".:f", "..envs.kuhn:KuhnEnv"):
    with pytest.raises(ValueError, match="relative path"):
      registry.resolve_path(bad)


def test_a_package_alias_resolves_into_this_package(clean):
  del clean
  path = f"{ALIAS}.envs.kuhn:KuhnEnv"
  assert registry.resolve_path(path) == path  # no alias yet: as given
  with pytest.raises(ModuleNotFoundError):
    registry.load_object(path)
  registry.register_package_alias(ALIAS)
  registry.register_package_alias(ALIAS)  # the same alias again: no effect
  registry.register_package_alias(PKG)  # this package's own name: no effect
  assert registry.package_aliases() == (ALIAS,)
  assert registry.package_path(path) == f"{PKG}.envs.kuhn:KuhnEnv"
  assert registry.package_path(ALIAS) == PKG
  assert registry.package_path(ALIAS + "_other.x:y") == ALIAS + "_other.x:y"
  assert registry.load_object(path) is kuhn.KuhnEnv
  seen = []
  registry.register_path_resolver(seen.append)
  registry.resolve_path(path)
  assert seen == [f"{PKG}.envs.kuhn:KuhnEnv"]  # resolvers see it as PKG's
  # MOVED_MODULES apply to aliased paths too.
  assert registry.resolve_path(f"{ALIAS}.eval.deadline:x") == (
      f"{PKG}.core.deadline:x"
  )


def test_a_bad_package_alias_raises(clean):
  del clean
  for bad in ("", "1abc", "a..b", "a.b:c", 3, None):
    with pytest.raises(ValueError, match="dotted module name"):
      registry.register_package_alias(bad)
  with pytest.raises(ValueError, match="ambiguous"):
    registry.register_package_alias(PKG + ".envs")
  if "." in PKG:
    with pytest.raises(ValueError, match="ambiguous"):
      registry.register_package_alias(PKG.rpartition(".")[0])
  registry.register_package_alias("acme.rl")
  for overlap in ("acme", "acme.rl.sub"):
    with pytest.raises(ValueError, match="overlaps"):
      registry.register_package_alias(overlap)


def test_an_object_alias_names_one_object_and_sets_nothing(clean):
  del clean

  class Moved:  # pylint: disable=too-few-public-methods
    pass

  path = f"{PKG}.envs.kuhn:MovedForTest"
  registry.register_object_alias(path, Moved)
  registry.register_object_alias(path, Moved)  # the same object: no effect
  assert registry.load_object(path) is Moved
  assert not hasattr(kuhn, "MovedForTest")
  # Under a package alias registered later, and relative, it is the same path.
  registry.register_package_alias(ALIAS)
  assert registry.load_object(f"{ALIAS}.envs.kuhn:MovedForTest") is Moved
  assert registry.load_object(".envs.kuhn:MovedForTest") is Moved
  with pytest.raises(ValueError, match="already names"):
    registry.register_object_alias(path, object)
  for bad in ("no_attr", "mod:", ":attr", "..x:y"):
    with pytest.raises(ValueError):
      registry.register_object_alias(bad, Moved)
  with pytest.raises(TypeError):
    registry.register_object_alias(3, Moved)


def test_unpickling_resolves_package_aliases_and_moved_modules(clean):
  del clean
  env = kuhn.KuhnEnv()
  blob = pickle.dumps({"cls": kuhn.KuhnEnv, "p": Point(3)}, protocol=0)
  recorded = _renamed(blob, f"{PKG}.envs.kuhn", f"{ALIAS}.envs.kuhn")
  with pytest.raises(ModuleNotFoundError):
    registry.loads(recorded)
  registry.register_package_alias(ALIAS)
  got = registry.loads(recorded)
  assert got["cls"] is kuhn.KuhnEnv and got["p"] == Point(3)
  assert registry.loads(blob)["cls"] is type(env)  # current names: unchanged
  # A path resolver remaps a global, as it remaps a recorded string.
  moved = _renamed(blob, __name__, "acme.old_tests")
  registry.register_path_resolver(
      lambda p: p.replace("acme.old_tests", __name__, 1)
      if p.startswith("acme.old_tests")
      else None
  )
  assert registry.loads(moved)["p"] == Point(3)
  # The stdlib unpickler does not: what this module adds is the find_class.
  # (S301: these are bytes this test pickled; it shows the plain loader fail.)
  with pytest.raises(ModuleNotFoundError):
    pickle.loads(moved)  # noqa: S301


def test_unpickling_returns_an_object_alias(clean):
  del clean
  blob = pickle.dumps(Point(4), protocol=0)
  recorded = _renamed(blob, __name__, f"{ALIAS}.league.gone")
  registry.register_package_alias(ALIAS)
  with pytest.raises(ModuleNotFoundError):
    registry.loads(recorded)
  registry.register_object_alias(f"{PKG}.league.gone:Point", Point)
  assert registry.loads(recorded) == Point(4)
  assert registry.Unpickler.__mro__[1] is pickle.Unpickler


def test_a_resolver_that_drops_the_attribute_raises(clean):
  del clean
  blob = pickle.dumps(Point(1), protocol=0)
  registry.register_path_resolver(
      lambda p: "acme.mod" if p.startswith(__name__ + ":") else None
  )
  with pytest.raises(TypeError, match="names no attribute"):
    registry.loads(blob)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
