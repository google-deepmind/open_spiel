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

"""The recorded-name setters, the ArenaSource.next override, scoped registries.

The set_* setters of recorded names, the ArenaSource.next override and the
scoped registrations in testing.registries (consumer-contract gaps 5, 6, 13,
23: a deployment sets these through a documented call, never by assigning a
module global or replacing a method).
"""

import ast
import dataclasses
import os
import sys
import types as pytypes

import numpy as np
import pytest

from open_spiel.python.scale import metrics
from open_spiel.python.scale import plugins
from open_spiel.python.scale.ckpt import codecs
from open_spiel.python.scale.core import arena
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.launch import subprocess_env
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.sources import deals
from open_spiel.python.scale.sources import shards
from open_spiel.python.scale.testing import registries


@pytest.fixture(name="restore")
def _restore(monkeypatch):
  # Records each global so the setters' writes roll back after the test.
  for mod, name in (
      (codecs, "TREE_SHA_TAG"),
      (deals, "HASH_TAG"),
      (deals, "INDEX_FORMAT"),
      (shards, "MANIFEST_FORMAT"),
      (runner, "EVAL_SERIES"),
  ):
    monkeypatch.setattr(mod, name, getattr(mod, name))


@pytest.mark.usefixtures("restore")
def test_set_tree_sha_tag_changes_the_digest():
  tree = {"a": np.arange(3, dtype=np.float32)}
  codecs.set_tree_sha_tag(b"params_tree_sha256\0")
  base = codecs.tree_sha256(tree)
  codecs.set_tree_sha_tag(b"other.tree\0")
  assert codecs.TREE_SHA_TAG == b"other.tree\0"
  assert codecs.tree_sha256(tree) != base
  assert codecs.tree_sha256(tree, tag=b"params_tree_sha256\0") == base


@pytest.mark.usefixtures("restore")
def test_deal_and_shard_setters_set_what_is_read():
  deals.set_hash_tag(b"other.deal\0")
  deals.set_index_format("other.deal_index")
  shards.set_manifest_format("other.shards")
  assert deals.HASH_TAG == b"other.deal\0"
  assert deals.INDEX_FORMAT == "other.deal_index"
  assert shards.MANIFEST_FORMAT == "other.shards"
  index = deals.DealIndex("g", "r", ["ab"])
  assert index.to_json()["format"] == "other.deal_index"


@pytest.mark.usefixtures("restore")
def test_set_eval_series_names_the_series_mean():
  est = pytypes.SimpleNamespace(n=4, mean=2.5, se=0.5)
  # A deployment's plugin may have set it in this process: set the default.
  runner.set_eval_series("eval")
  assert runner.eval_series_mean("k", est)["name"] == "eval/k"
  runner.set_eval_series("tracking")
  assert runner.eval_series_mean("k", est)["name"] == "tracking/k"


@pytest.mark.usefixtures("restore")
@pytest.mark.parametrize(
    "call,bad",
    [
        (codecs.set_tree_sha_tag, b"no_nul"),
        (codecs.set_tree_sha_tag, "str\0"),
        (codecs.set_tree_sha_tag, b"\0"),
        (deals.set_hash_tag, b"no_nul"),
        (deals.set_hash_tag, "str\0"),
        (deals.set_index_format, ""),
        (deals.set_index_format, b"bytes"),
        (shards.set_manifest_format, ""),
        (shards.set_manifest_format, None),
        (runner.set_eval_series, ""),
        (runner.set_eval_series, "a/b"),
    ],
)
def test_setters_refuse_bad_values(call, bad):
  with pytest.raises(ValueError):
    call(bad)


def test_override_source_next_runs_first_and_restores():
  src = object.__new__(arena.ArenaSource)  # next never reaches the arena here
  seen = []

  def fn(source, key):
    """The override: records its call.

    Args:
      source: the ArenaSource.
      key: the PRNG key.

    Returns:
      A stand-in batch.
    """
    seen.append((source, key))
    return "batch"

  with arena.override_source_next(fn):
    assert src.next("k") == "batch"
    with pytest.raises(RuntimeError, match="do not nest"):
      with arena.override_source_next(fn):
        pass
  assert seen == [(src, "k")]
  assert not arena._SOURCE_NEXT_OVERRIDE  # pylint: disable=protected-access


def test_override_source_next_propagates_and_clears_on_raise():
  src = object.__new__(arena.ArenaSource)

  class StopError(Exception):
    pass

  def fn(source, key):
    del key
    raise StopError(source)

  with pytest.raises(StopError):
    with arena.override_source_next(fn):
      src.next("k")
  assert not arena._SOURCE_NEXT_OVERRIDE  # pylint: disable=protected-access
  with pytest.raises(ValueError):
    with arena.override_source_next("not callable"):
      pass


def test_override_source_next_none_lets_the_rollout_run():
  src = object.__new__(arena.ArenaSource)  # no arena: the rollout fails loud
  with arena.override_source_next(lambda source, key: None):
    with pytest.raises(AttributeError):
      src.next("k")


def test_scoped_metrics_replaces_and_restores():
  before = registries.registered_metrics()
  got = []
  with registries.scoped_metrics(series_emitters=[got.append]):
    assert registries.registered_metrics()["series_emitters"] == (got.append,)
    assert registries.registered_metrics()["auto_sinks"] == before["auto_sinks"]
    metrics.emit_series(3, {"a/b": 1.0})

    def extra(vals):
      """A params publisher that publishes nothing.

      Args:
        vals: the params (unused).

      Returns:
        False.
      """
      del vals
      return False

    metrics.register_params_publisher(extra)
  # the block's registration rolled back
  assert registries.registered_metrics() == before
  assert got and got[0]["step"] == 3
  with pytest.raises(TypeError):
    with registries.scoped_metrics(series_emitters=[1]):
      pass
  assert registries.registered_metrics() == before


def test_scoped_loss_registers_and_restores():
  def never(*a, **k):
    raise AssertionError("not called")

  spec = pg.LossSpec(
      name="setters_test_loss",
      config_type=dict,
      targets=never,
      loss=never,
      temperature=never,
  )
  assert "setters_test_loss" not in pg.losses()
  with registries.scoped_loss(spec):
    assert pg.extension("setters_test_loss") is spec
    other = dataclasses.replace(spec)
    with registries.scoped_loss(other):
      assert pg.extension("setters_test_loss") is other
    assert pg.extension("setters_test_loss") is spec
  assert "setters_test_loss" not in pg.losses()
  with pytest.raises(ValueError, match="built-in"):
    with registries.scoped_loss(dataclasses.replace(spec, name="ppo")):
      pass
  with pytest.raises(TypeError):
    with registries.scoped_loss("setters_test_loss"):
      pass


def test_open_view_undoes_a_plugin_and_restores_it(monkeypatch):
  """open_view: the built-in rows, the setters' defaults, no plugin in the env.

  A plugin loaded in the same process (another package's conftest) replaced
  gin_rummy, set the rules_sha tag, the eval series and the tree sha tag,
  registered an alias and named itself in plugins.ENV; inside open_view all
  of that is the package's own, after it the plugin's again.

  Args:
    monkeypatch: pytest's.
  """
  # pylint: disable=protected-access
  builtin = {g.name: g for g in env_registry.BUILTIN}["gin_rummy"]
  assert env_registry.game("gin_rummy") is builtin  # the conftest's view
  kuhn = env_registry.make("kuhn_poker")
  open_sha = env_registry.rules_sha(kuhn)
  # The plugin, undone by monkeypatch after the test.
  monkeypatch.setitem(env_registry._GAMES, "gin_rummy", builtin)
  monkeypatch.setattr(
      env_registry, "_SOURCE_ALIASES", dict(env_registry._SOURCE_ALIASES)
  )
  tag = env_registry._RULES_SHA_TAG
  monkeypatch.setitem(tag, "tag", tag["tag"])
  monkeypatch.setitem(tag, "source", tag["source"])
  monkeypatch.setenv(plugins.ENV, "acme.plugin")
  monkeypatch.setattr(runner, "EVAL_SERIES", runner.EVAL_SERIES)
  monkeypatch.setattr(codecs, "TREE_SHA_TAG", codecs.TREE_SHA_TAG)
  runner.set_eval_series("acme_eval")
  codecs.set_tree_sha_tag(b"acme_tree\0")
  plugin_row = dataclasses.replace(builtin, default_kwargs={"acme": 1})
  env_registry.register(plugin_row, replace=True)
  env_registry.set_rules_sha_tag("acme.rules/1", source=env_registry.RAW_SOURCE)
  env_registry.register_rules_source_alias("a" * 64, "b" * 64)
  plugin_sha = env_registry.rules_sha(kuhn)
  assert plugin_sha != open_sha
  with registries.open_view():
    assert env_registry.game("gin_rummy") is builtin
    assert env_registry._RULES_SHA_TAG == registries.OPEN_RULES_SHA_TAG
    assert not env_registry.rules_source_aliases()
    assert plugins.ENV not in os.environ
    assert env_registry.rules_sha(kuhn) == open_sha
    assert runner.EVAL_SERIES == "eval"
    assert codecs.TREE_SHA_TAG == codecs.DEFAULT_TREE_SHA_TAG
    assert not registries.setting_drift()
  assert runner.EVAL_SERIES == "acme_eval"
  assert codecs.TREE_SHA_TAG == b"acme_tree\0"
  assert env_registry.game("gin_rummy") is plugin_row
  assert env_registry._RULES_SHA_TAG == {
      "tag": "acme.rules/1",
      "source": env_registry.RAW_SOURCE,
  }
  assert env_registry.rules_source_aliases()["a" * 64] == "b" * 64
  assert os.environ[plugins.ENV] == "acme.plugin"
  assert env_registry.rules_sha(kuhn) == plugin_sha
  # Restored on an exception too; with nothing to undo it changes nothing.
  with pytest.raises(RuntimeError):
    with registries.open_view():
      raise RuntimeError("boom")
  assert env_registry.game("gin_rummy") is plugin_row
  assert os.environ[plugins.ENV] == "acme.plugin"
  monkeypatch.delenv(plugins.ENV)
  with registries.open_view():
    pass
  assert plugins.ENV not in os.environ
  # pylint: enable=protected-access


def test_open_view_defaults_are_those_of_a_plugin_free_process():
  """Every setting open_view holds is its value in a process with no plugin.

  A child with every setting module imported and plugins.ENV unset has no
  drift; in this module (the conftest's open view) neither has this process.
  """
  assert not registries.setting_drift()
  env = {k: v for k, v in os.environ.items() if k != plugins.ENV}
  pkg = subprocess_env.PACKAGE
  r = subprocess_env.run_python(
      [
          "-c",
          (
              "import importlib\n"
              f"from {pkg}.testing import registries\n"
              "for m in registries.SETTING_MODULES:\n"
              "  importlib.import_module(m)\n"
              "d = registries.setting_drift()\n"
              "assert not d, d\n"
              "print('NO_DRIFT')\n"
          ),
      ],
      base_env=env,
      timeout=300,
  )
  assert r.returncode == 0 and "NO_DRIFT" in r.stdout, r.stderr[-3000:]


def test_every_set_function_is_held_by_open_view_or_says_why_not():
  """Every top-level set_* of the package is in HELD_SETTERS or UNHELD_SETTERS.

  A new set_* extension point that open_view does not hold would leak a
  plugin's setting into these tests again.
  """
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  found = set()
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [
        d for d in dirnames if d not in ("tests", "testing", "__pycache__")
    ]
    for fn in filenames:
      if not fn.endswith(".py"):
        continue
      path = os.path.join(dirpath, fn)
      rel = os.path.relpath(path, root)[: -len(".py")].replace(os.sep, ".")
      rel = rel[: -len(".__init__")] if rel.endswith(".__init__") else rel
      with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
      found |= {
          f"{rel}.{node.name}"
          for node in tree.body
          if isinstance(node, ast.FunctionDef) and node.name.startswith("set_")
      }
  held = set()
  for rel, names in registries.HELD_SETTERS.items():
    held |= {f"{rel}.{name}" for name in names}
  assert held.isdisjoint(registries.UNHELD_SETTERS)
  assert found == held | set(registries.UNHELD_SETTERS), (
      "not in HELD_SETTERS or UNHELD_SETTERS:",
      sorted(found - held - set(registries.UNHELD_SETTERS)),
      "listed but not found:",
      sorted(held | set(registries.UNHELD_SETTERS) - found),
  )
  # A held setter's module has its settings in the open view.
  for rel in registries.HELD_SETTERS:
    assert f"{subprocess_env.PACKAGE}.{rel}" in registries.SETTING_MODULES, rel


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
