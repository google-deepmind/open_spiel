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


"""league.config_common: config overlays, and ambiguous names raise.

* an overlay replaces load_config of its exact name only, and leaves
  config_modules / config_module alone;
* a hidden config name is not listed, and still loads through its overlay;
* two resolver prefixes that nest are refused at registration;
* a name that both an overlay and a prefix, or both a config module and a
  prefix, would answer raises at load_config instead of picking one.
"""

import dataclasses
import sys

import pytest

from open_spiel.python.scale.league import config_common as common

_BASE = "mmd_kuhn_poker_jax"


@pytest.fixture(autouse=True)
def _own_registries(monkeypatch):
  """Gives each test its own copy of the resolver and overlay tables."""
  monkeypatch.setattr(common, "_RESOLVERS", dict(common._RESOLVERS))  # pylint: disable=protected-access
  monkeypatch.setattr(common, "_OVERLAYS", dict(common._OVERLAYS))  # pylint: disable=protected-access
  monkeypatch.setattr(common, "_HIDDEN", set(common._HIDDEN))  # pylint: disable=protected-access


_BUILT = {}


def _named(name):
  """The base config under another name (built once, before any overlay).

  Args:
    name: the new name.

  Returns:
    The renamed base config.
  """
  if _BASE not in _BUILT:
    _BUILT[_BASE] = common.load_config(_BASE)
  return dataclasses.replace(_BUILT[_BASE], name=name)


def test_an_overlay_replaces_its_exact_name_only():
  """load_config(name) is the overlay's; other names and listings are not."""
  module = common.config_modules()[_BASE]
  base_name = common.load_config(_BASE).name
  _named("warm")  # builds the base config before the overlay takes its name
  common.register_config_overlay(_BASE, lambda: _named("overlaid"))
  assert common.load_config(_BASE).name == "overlaid"
  assert common.config_modules()[_BASE] == module
  assert common.config_module(_BASE).__name__ == module
  assert base_name != "overlaid"
  with pytest.raises(KeyError, match="unknown config"):
    common.load_config(_BASE + "_large")


def test_an_overlay_may_refuse_its_name():
  """An overlay that raises makes loading that name raise, loudly."""

  def refuse():
    raise RuntimeError("not in this process")

  common.register_config_overlay("refused_cfg", refuse)
  with pytest.raises(RuntimeError, match="not in this process"):
    common.load_config("refused_cfg")


def test_a_second_overlay_of_a_name_is_refused():
  """Registering another fn for the same name raises; the same fn is a no-op."""
  fn = lambda: _named("a")  # pylint: disable=unnecessary-lambda-assignment
  common.register_config_overlay("twice_cfg", fn)
  common.register_config_overlay("twice_cfg", fn)
  with pytest.raises(ValueError, match="already registered"):
    common.register_config_overlay("twice_cfg", lambda: _named("b"))


def test_a_hidden_name_is_not_listed_and_loads_only_through_its_overlay():
  """register_hidden_config unlists a name; load_config still loads it."""
  assert _BASE in common.config_names()
  _named("warm")
  common.register_hidden_config(_BASE)
  assert _BASE not in common.config_names()
  assert _BASE not in common.config_modules()
  assert common.config_module(_BASE) is None
  with pytest.raises(KeyError, match="unknown config"):
    common.load_config(_BASE)
  common.register_config_overlay(_BASE, lambda: _named("hidden_overlay"))
  assert common.load_config(_BASE).name == "hidden_overlay"


def test_nested_resolver_prefixes_are_refused():
  """Two prefixes where one starts with the other could both answer a name."""
  common.register_config_resolver("acme.", _named)
  common.register_config_resolver("acme.", common._RESOLVERS["acme."])  # pylint: disable=protected-access
  for nested in ("acme.x", "acm"):
    with pytest.raises(ValueError, match="nest"):
      common.register_config_resolver(nested, _named)
  assert common.load_config("acme.r7").name == "r7"


def test_a_name_both_an_overlay_and_a_prefix_answer_raises():
  """An overlay name that a resolver prefix also matches is ambiguous."""
  common.register_config_resolver("ovl_", _named)
  common.register_config_overlay("ovl_one", lambda: _named("overlay"))
  with pytest.raises(ValueError, match="has an overlay and matches"):
    common.load_config("ovl_one")
  assert common.load_config("ovl_two").name == "two"


def test_a_config_module_shadowed_by_a_prefix_raises():
  """A prefix that would answer a config module's name raises at load."""
  common.register_config_resolver(_BASE[:4], _named)
  with pytest.raises(ValueError, match="is the config module"):
    common.load_config(_BASE)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
