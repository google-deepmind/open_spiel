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

"""The extension points a deployment registers into, each with a known answer.

plugin loading, path resolvers, config packages and resolvers, hash transforms,
learner kinds, run ids and code-repo provenance, and the reference-eval
registry. The compile cache's shared-dir rule is tested beside its module
(launch/tests/compile_cache_test.py).
"""

import ast
import builtins
import dataclasses
import os
import re
import sys

import pytest

from open_spiel.python.scale import plugins
from open_spiel.python.scale import registry
from open_spiel.python.scale.ckpt import codecs
from open_spiel.python.scale.ckpt import formats
from open_spiel.python.scale.ckpt import provenance
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league.kinds import base as kinds_base


def _module(tmp_path, monkeypatch, name, body):
  """Writes module `name` of a temporary plugin package.

  Args:
    tmp_path: The folder.
    monkeypatch: The pytest fixture.
    name: The module name.
    body: The source.

  Returns:
    The module's import name.
  """
  pkg = tmp_path / "plugpkg"
  pkg.mkdir(exist_ok=True)
  (pkg / "__init__.py").write_text("")
  (pkg / f"{name}.py").write_text(body)
  monkeypatch.syspath_prepend(str(tmp_path))
  return f"plugpkg.{name}"


def test_plugins_load_each_module_once_and_from_the_environment(
    tmp_path, monkeypatch
):
  """Plugins load each module once, also when named in the environment."""
  mod = _module(
      tmp_path,
      monkeypatch,
      "counter",
      "import builtins\n"
      "builtins.PLUGIN_TEST_LOADS = getattr(builtins, 'PLUGIN_TEST_LOADS', 0) "
      "+ 1\n",
  )
  monkeypatch.setattr(plugins, "_LOADED", [])
  monkeypatch.setattr(builtins, "PLUGIN_TEST_LOADS", 0, raising=False)
  assert plugins.load_env({plugins.ENV: f" {mod}, "}) == (mod,)
  assert plugins.load([mod]) == (mod,) and builtins.PLUGIN_TEST_LOADS == 1
  with pytest.raises(ValueError, match="dotted"):
    plugins.load(["not a module"])
  with pytest.raises(ModuleNotFoundError):
    plugins.load(["plugpkg.missing"])


def test_path_resolvers_rewrite_recorded_paths_in_order(monkeypatch):
  """Registered path resolvers rewrite recorded paths, in order."""
  monkeypatch.setattr(registry, "_RESOLVERS", [])
  assert registry.resolve_path("acme.old_pkg.kuhn_jax:KuhnJaxEnv") == (
      "acme.old_pkg.kuhn_jax:KuhnJaxEnv"
  )
  envs = registry.__name__.rpartition(".")[0] + ".envs"

  def first(p):
    """Rewrites acme.old_pkg paths to the envs package.

    Args:
      p: The recorded path.

    Returns:
      The rewritten path, or None.
    """
    if p.startswith("acme."):
      return p.replace("acme.old_pkg", envs, 1)
    return None

  registry.register_path_resolver(first)
  with pytest.raises(ValueError, match="module:attr"):
    registry.load_object(envs)
  registry.register_path_resolver(lambda p: "never.reached:x")
  registry.register_path_resolver(first)  # a second registration has no effect
  assert (
      registry.load_object("acme.old_pkg.kuhn_jax:KuhnJaxEnv").name
      == "kuhn_poker_jax"
  )
  assert registry.resolve_path("other.mod:f") == "never.reached:x"
  with pytest.raises(TypeError):
    registry.register_path_resolver("not callable")


def test_config_packages_resolvers_and_hash_transforms(tmp_path, monkeypatch):
  """Config packages, config resolvers and hash transforms take effect."""
  pkg = tmp_path / "acmecfg"
  pkg.mkdir()
  (pkg / "__init__.py").write_text("HELPER_MODULES = ('helper.py',)\n")
  (pkg / "helper.py").write_text("X = 1\n")
  (pkg / "recipes.py").write_text(
      "from acmecfg.acme_kuhn import get_config\n"
  )  # named like a helper of the library's package
  # A file a test runner generates beside the configs: no get_config.
  (pkg / "acme_kuhn_test_wrapper.py").write_text(
      "import runpy\nrunpy.run_module('acmecfg.acme_kuhn')\n"
  )
  # The framework's configs package, derived from this package's name.
  configs_pkg = registry.__name__.rpartition(".")[0] + ".configs"
  (pkg / "acme_kuhn.py").write_text(
      "import dataclasses\n"
      f"from {configs_pkg} import mmd_kuhn_poker_jax as base\n"
      "def get_config():\n  return dataclasses.replace(base.get_config(), "
      "name='acme_kuhn')\n"
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  for attr, copy in (
      ("_PACKAGES", list),
      ("_RESOLVERS", dict),
      ("_HASH_TRANSFORMS", list),
  ):
    monkeypatch.setattr(config_common, attr, copy(getattr(config_common, attr)))
  assert "acme_kuhn" not in config_common.config_names()
  config_common.register_config_package("acmecfg")
  names = config_common.config_names()
  assert "acme_kuhn" in names and "helper" not in names
  assert "acme_kuhn_test_wrapper" not in names
  assert (
      config_common.config_modules()["recipes"] == "acmecfg.recipes"
  )  # only its own helpers skip
  cfg = config_common.load_config("acme_kuhn")
  assert cfg.name == "acme_kuhn"
  config_common.register_config_resolver(
      "acme.", lambda rest: dataclasses.replace(cfg, name=rest)
  )
  assert config_common.load_config("acme.x7").name == "x7"
  before = config_common.config_hash(cfg)
  resume_before = config_common.resume_hash(cfg)

  def renamed(d):
    """A hash transform that records the learner kind as recorded_name.

    Args:
      d: the config dict.

    Returns:
      the transformed dict.
    """
    return dict(
        d, learner=dict(d["learner"], recorded_name=d["learner"]["kind"])
    )

  config_common.register_hash_transform(renamed)
  assert config_common.config_dict(cfg)["learner"]["recorded_name"] == "pg"
  assert config_common.plain(cfg)["learner"].get("recorded_name") is None
  assert (
      config_common.config_hash(cfg) != before
      and config_common.resume_hash(cfg) != resume_before
  )
  with pytest.raises(KeyError, match="unknown config"):
    config_common.load_config("no_such_config")


def test_learner_kind_registry(monkeypatch):
  """A registered learner kind builds its driver from its spec."""
  # Reaches private _KINDS of the code under test.
  monkeypatch.setattr(kinds_base, "_KINDS", dict(kinds_base._KINDS))  # pylint: disable=protected-access

  @dataclasses.dataclass(frozen=True)
  class Toy:
    """A toy learner spec.

    Attributes:
      kind: the learner kind's registered name.
    """

    kind: str = "toy"

  made = []

  class ToyKind(kinds.KindBase):
    """A toy learner kind."""

    name = "toy"
    spec_cls = Toy

    def build_driver(self, cfg, env, **kw):
      """Records cfg.

      Args:
        cfg: the config.
        env: unused.
        **kw: unused.

      Returns:
        the stand-in driver "R".
      """
      del env, kw
      made.append(cfg)
      return "R"

  kinds.register_kind(ToyKind())
  assert "toy" in kinds.names() and isinstance(kinds.of(Toy()), ToyKind)
  assert kinds.make_driver(Toy(), None, {}) == "R" and made == [Toy()]
  with pytest.raises(ValueError, match="already registered"):
    kinds.register_kind(ToyKind())
  with pytest.raises(TypeError, match="LearnerKind"):
    kinds.register_kind(object())
  with pytest.raises(TypeError, match="spec"):
    kinds.of(object())


def test_artifact_format_and_codec_hooks(monkeypatch):
  """Artifact format and codec hooks register."""
  monkeypatch.setattr(formats, "_REGISTERED", [])
  monkeypatch.setattr(formats, "_WRITER", {"name": formats.NATIVE})

  class Fmt:
    """An artifact format that detects nothing."""

    name = "acme.format"

    def detect(self, d, step):
      """Detects nothing.

      Args:
        d: unused.
        step: unused.

      Returns:
        False.
      """
      del d, step
      return False

  formats.register_format(Fmt())
  assert formats.names() == ("acme.format", formats.NATIVE)
  with pytest.raises(ValueError, match="already registered"):
    formats.register_format(Fmt())
  with pytest.raises(TypeError, match="read-only"):
    formats.set_default_writer("acme.format")
    formats.writer()
  formats.set_default_writer(formats.NATIVE)
  codec = codecs.params_codec("jax")
  codecs.register_params_codec("jax", codec)  # the same codec again: a no-op
  with pytest.raises(ValueError, match="already registered"):
    codecs.register_params_codec("jax", object())


def test_run_ids_and_code_repo(monkeypatch):
  # A fresh state: a deployment plugin loaded in this process may have set both.
  """Run ids and the code repo resolve from the registered env and hooks."""
  monkeypatch.setattr(
      provenance,
      "_STATE",
      {
          "attempt_id_env": None,
          # Reaches private _as_string of the code under test.
          "parse": provenance._as_string,  # pylint: disable=protected-access
          "code_repo": None,
      },
  )
  monkeypatch.setenv("ACME_ATTEMPT", "42")
  monkeypatch.delenv(provenance.CODE_REPO_ENV, raising=False)
  assert provenance.resolve_run_ids(None) == provenance.RunIds(
      None, None
  )  # no env named
  provenance.set_attempt_id_env("ACME_ATTEMPT")
  assert provenance.resolve_run_ids("run-7") == provenance.RunIds("run-7", "42")
  provenance.set_attempt_id_env("ACME_ATTEMPT", id_parser=int)
  assert provenance.resolve_run_ids(7) == provenance.RunIds(7, 42)
  monkeypatch.setenv("ACME_ATTEMPT", "x")
  with pytest.raises(ValueError, match="ACME_ATTEMPT"):
    provenance.resolve_run_ids(None)
  assert provenance.code_repo() is None
  provenance.set_code_repo("acme/rl")
  assert provenance.code_repo() == "acme/rl"
  monkeypatch.setenv(provenance.CODE_REPO_ENV, "acme/other")
  assert provenance.code_repo() == "acme/other"


def test_reference_eval_registry(monkeypatch):
  """A registered reference eval resolves for an InLoopEval."""
  # Reaches private _REFERENCE_EVALS of the code under test.
  monkeypatch.setattr(runner, "_REFERENCE_EVALS", dict(runner._REFERENCE_EVALS))  # pylint: disable=protected-access

  @dataclasses.dataclass(frozen=True)
  class Ref:
    """A stand-in reference eval.

    Attributes:
      label: the eval's label.
    """

    label: str

  built = []
  runner.register_reference_eval(
      "toy_h2h",
      "kuhn_poker_jax",
      Ref,
      lambda ev, env, mesh, cache: built.append(ev.key) or "E",
  )
  ev = runner.InLoopEval("toy_h2h", every=1, deals=2, reference=Ref("r1"))
  assert (
      ev.key == "h2h_r1"
      and runner.build_eval(ev, None) == "E"
      and built == ["h2h_r1"]
  )
  with pytest.raises(ValueError, match="reference"):
    runner.InLoopEval("toy_h2h", every=1, deals=2)
  with pytest.raises(TypeError, match="Ref"):
    runner.InLoopEval("toy_h2h", every=1, deals=2, reference="r1")


# Module-level register_* / set_* functions that are not extension points.
_NOT_EXTENSION_POINTS = {"register_params", "set_native"}


def _doc_blocks() -> dict[str, str]:
  """The plugins doc's "Extension points" list as {package: entry text}.

  Returns:
    {package: entry text}.
  """
  text = plugins.__doc__.split("Extension points, by package", 1)[1]
  blocks: dict[str, str] = {}
  current = None
  for line in text.splitlines():
    m = re.match(r"^  (\S+)\s+(.*)$", line)
    if m:
      current = m.group(1)
      blocks[current] = m.group(2)
    elif current and line.startswith("    ") and line.strip():
      blocks[current] += " " + line.strip()
  return blocks


def test_plugins_doc_lists_every_registration_function():
  """The plugins doc lists every registration function."""
  pkg_dir = os.path.dirname(os.path.abspath(plugins.__file__))
  pkg = plugins.__name__.rpartition(".")[0]
  blocks = _doc_blocks()
  missing = []
  for dirpath, dirnames, filenames in os.walk(pkg_dir):
    dirnames[:] = [
        d for d in dirnames if d not in ("tests", "testing", "__pycache__")
    ]
    for fn in filenames:
      if not fn.endswith(".py") or fn.endswith("_test.py"):
        continue
      path = os.path.join(dirpath, fn)
      rel = os.path.relpath(path, pkg_dir)[:-3].split(os.sep)
      parts = rel[:-1] if rel[-1] == "__init__" else rel
      with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
      for node in tree.body:
        if not (
            isinstance(node, ast.FunctionDef)
            and re.match(r"(register|set)_", node.name)
            and node.name not in _NOT_EXTENSION_POINTS
        ):
          continue
        heads = ["(root)"] + [
            ".".join(parts[:i]) for i in range(1, len(parts) + 1)
        ]
        found = False
        for head in heads:
          for m in re.finditer(
              r"([\w.]*?)\b" + node.name + r"\b", blocks.get(head, "")
          ):
            qual = [q for q in m.group(1).split(".") if q]
            if all(q in parts or q in ("plugins", "registry") for q in qual):
              found = True
        if not found:
          missing.append(".".join([pkg] + parts + [node.name]))
  assert not missing, f"plugins' extension point list misses {missing}"


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
