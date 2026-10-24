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

"""envs.registry is the one game table.

Pins: every registered game builds an Env of its name at its default
options; EnvSpecs are complete and canonical (defaults filled, ints for float
options normalised, unknown / non-JSON / out-of-range options refused) and
round-trip through JSON and through spec_of(make(spec)); a replaced spec is
used and logged, a conflicting one refused; extras load on first build;
rules_sha is sha256(tag, module source sha, EnvSpec key) and separates
variants; gin's covers() table; the serving, league and eval tables are
views of the registry.
"""

import dataclasses
import hashlib
import inspect
import json
import sys

import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry
from open_spiel.python.scale.envs.gin import search as gin_search
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league import slot_factories
from open_spiel.python.scale.serve import games

_PKG = registry.__name__.rpartition(".envs.")[0]  # the package's import name

GIN_DEFAULTS = {
    "oklahoma_frac": None,
    "allow_upcard_throwback": True,
    "wall_knock_pick": True,
    "spade_doubles": False,
}


@pytest.mark.parametrize("name", registry.builtin_names())
def test_every_name_builds_its_env_at_its_defaults(name):
  """Every name builds its env at its defaults.

  Args:
    name: The parametrized case.
  """
  env = registry.make(name)
  assert env.name == name
  assert registry.spec_of(env) == registry.spec(name)
  assert registry.spec(name).kwargs == dict(registry.game(name).default_kwargs)


def test_gin_defaults_are_openspiel_rules_and_standard_gin():
  """Gin defaults are openspiel rules and standard gin."""
  assert dict(registry.game("gin_rummy").default_kwargs) == GIN_DEFAULTS
  env = registry.make("gin_rummy")
  assert env.oklahoma_frac is None and env.options == GIN_DEFAULTS
  assert registry.make("gin_rummy", oklahoma_frac=1).oklahoma_frac == 1.0


def test_gin_rule_options_are_validated():
  """Gin rule options are validated."""
  with pytest.raises(ValueError, match="spade_doubles"):
    registry.make("gin_rummy", spade_doubles=True)
  assert registry.make(
      "gin_rummy", oklahoma_frac=1.0, spade_doubles=True
  ).spade_doubles
  with pytest.raises(ValueError, match="wall_knock_pick"):
    registry.make("gin_rummy", oklahoma_frac=1.0, wall_knock_pick=False)
  with pytest.raises(TypeError, match="must be a bool"):
    registry.spec("gin_rummy", allow_upcard_throwback=1)


def test_unknown_name_fails_loud_and_lists_names():
  """Unknown name fails loud and lists names."""
  with pytest.raises(KeyError, match="no env named.*known"):
    registry.make("chess")


def test_spec_is_complete_canonical_and_validated():
  """Spec is complete canonical and validated."""
  s = registry.spec("gin_rummy", oklahoma_frac=1)
  assert s == registry.spec("gin_rummy", oklahoma_frac=1.0)
  assert s.to_json() == {
      "name": "gin_rummy",
      "kwargs": {**GIN_DEFAULTS, "oklahoma_frac": 1.0},
  }
  assert hash(s) == hash(registry.spec("gin_rummy", oklahoma_frac=1.0))
  assert s != registry.spec("gin_rummy")
  with pytest.raises(TypeError, match="unknown env option"):
    registry.spec("gin_rummy", oklahoma=True)
  with pytest.raises(TypeError, match="unknown env option"):
    registry.spec("kuhn_poker", players=2)
  with pytest.raises(ValueError, match="in \\[0, 1\\]"):
    registry.spec("gin_rummy", oklahoma_frac=1.5)
  with pytest.raises(TypeError, match="JSON"):
    registry.spec("gin_rummy", oklahoma_frac=[1.0])
  with pytest.raises(TypeError, match="gomoku connect must be an int"):
    registry.spec("gomoku", connect=4.0)


def test_spec_json_round_trip_and_refuses_non_canonical():
  """Spec json round trip and refuses non canonical."""
  for s in (
      registry.spec("gin_rummy", oklahoma_frac=0.5),
      registry.spec("kuhn_poker"),
      registry.spec("gomoku", board_size=9, connect=4),
  ):
    d = json.loads(json.dumps(s.to_json()))
    assert registry.EnvSpec.from_json(d) == s
  with pytest.raises(ValueError, match="not a canonical"):
    # defaults missing
    registry.EnvSpec.from_json({"name": "gin_rummy", "kwargs": {}})
  with pytest.raises(ValueError, match="name, kwargs"):
    registry.EnvSpec.from_json({"name": "kuhn_poker"})


def test_make_and_spec_of_round_trip_non_default_options():
  """Make and spec of round trip non default options."""
  s = registry.spec("gomoku", board_size=9, connect=4)
  env = registry.make(s)
  assert env.board_size == 9 and env.connect == 4
  assert registry.spec_of(env) == s
  g = registry.make("gin_rummy", oklahoma_frac=0.25)
  assert registry.spec_of(g) == registry.spec("gin_rummy", oklahoma_frac=0.25)


def _manual_rules_sha(env, spec_key, tag="rules/1"):
  """The rules digest computed by hand from the game's source.

  Args:
    env: The env.
    spec_key: The env spec's key.
    tag: The digest tag.

  Returns:
    The hex digest.
  """
  with open(inspect.getsourcefile(type(env.game)), "rb") as f:
    src = hashlib.sha256(f.read()).hexdigest()
  return hashlib.sha256(
      tag.encode() + b"\0" + src.encode() + b"\0" + spec_key.encode() + b"\0"
  ).hexdigest()


def test_rules_sha_known_answer_and_covers_options():
  """Rules sha known answer and covers options."""
  std = registry.make("gin_rummy")
  okl = registry.make("gin_rummy", oklahoma_frac=1.0)
  assert registry.rules_sha(std) == _manual_rules_sha(
      std, registry.spec_of(std).key()
  )
  assert registry.rules_sha(std) != registry.rules_sha(okl)
  assert registry.rules_sha(okl) == registry.rules_sha(
      registry.make("gin_rummy", oklahoma_frac=1)
  )
  # one implementation
  assert config_common.rules_sha(okl) == registry.rules_sha(okl)
  small = registry.make("gomoku", board_size=9)
  big = registry.make("gomoku")
  assert registry.rules_sha(small) != registry.rules_sha(big)


def test_rules_sha_tag_is_settable(monkeypatch):
  """Rules sha tag is settable.

  Args:
    monkeypatch: pytest monkeypatch fixture.
  """
  env = registry.make("kuhn_poker")
  # Reaches private _RULES_SHA_TAG of the code under test.
  tag = registry._RULES_SHA_TAG  # pylint: disable=protected-access
  monkeypatch.setitem(tag, "tag", tag["tag"])
  registry.set_rules_sha_tag("acme.rules/7")
  assert registry.rules_sha(env) == _manual_rules_sha(
      env, registry.spec_of(env).key(), "acme.rules/7"
  )
  with pytest.raises(ValueError):
    registry.set_rules_sha_tag("")


@pytest.mark.parametrize(
    "trained,served,want",
    [
        (None, None, True),
        (1.0, 1.0, True),
        (None, 1.0, False),  # a standard-gin artifact does not play Oklahoma
        (1.0, None, False),
        (0.5, None, True),  # a mixed artifact played both variants
        (0.5, 1.0, True),
        (0.0, None, True),  # frac 0 deals only standard hands
        (0.0, 1.0, False),
    ],
)
def test_gin_covers_known_answers(trained, served, want):
  """Gin covers known answers.

  Args:
    trained: The parametrized case.
    served: The parametrized case.
    want: The parametrized case.
  """
  t = registry.spec("gin_rummy", oklahoma_frac=trained)
  s = registry.spec("gin_rummy", oklahoma_frac=served)
  assert registry.covers(t, s) is want


def test_gin_covers_needs_equal_rules():
  """Gin covers needs equal rules."""
  t = registry.spec("gin_rummy", oklahoma_frac=0.5)
  s = registry.spec(
      "gin_rummy", oklahoma_frac=None, allow_upcard_throwback=False
  )
  assert not registry.covers(t, s)


def test_default_covers_is_equality_and_needs_the_same_game():
  """Default covers is equality and needs the same game."""
  a = registry.spec("gomoku")
  assert registry.covers(a, registry.spec("gomoku"))
  assert not registry.covers(a, registry.spec("gomoku", connect=4))
  assert not registry.covers(
      registry.spec("kuhn_poker"), registry.spec("leduc_poker")
  )


def test_check_serves():
  """Check serves."""
  okl = registry.make("gin_rummy", oklahoma_frac=1.0)
  std = registry.make("gin_rummy")
  trained = registry.spec("gin_rummy", oklahoma_frac=1.0).to_json()
  registry.check_serves(trained, okl)
  registry.check_serves(None, std)  # a manifest without an env_spec: game only
  with pytest.raises(ValueError, match="allow_cross_variant"):
    registry.check_serves(trained, std)
  registry.check_serves(trained, std, allow_cross_variant=True)
  with pytest.raises(ValueError, match="trained on 'gin_rummy'"):
    registry.check_serves(
        trained, registry.make("kuhn_poker"), allow_cross_variant=True
    )


def test_register_is_idempotent_refuses_a_conflict_and_replaces_on_request(
    monkeypatch,
):
  """Register is idempotent refuses a conflict and replaces on request.

  Args:
    monkeypatch: pytest monkeypatch fixture.
  """
  # Reaches private _GAMES of the code under test.
  monkeypatch.setattr(registry, "_GAMES", dict(registry._GAMES))  # pylint: disable=protected-access
  g = registry.game("kuhn_poker")
  assert registry.register(g) is g
  other = dataclasses.replace(
      g, extras=g.extras + (f"{_PKG}.policies.uniform",)
  )
  with pytest.raises(ValueError, match="already registered"):
    registry.register(other)
  assert registry.register(other, replace=True) is other
  assert registry.game("kuhn_poker") is other


def test_extras_load_on_first_build_and_by_component_prefix(monkeypatch):
  """Extras load on first build and by component prefix.

  Args:
    monkeypatch: pytest monkeypatch fixture.
  """
  monkeypatch.setattr(registry, "_EXTRAS_LOADED", set())
  registry.make("connect_four")
  assert f"{_PKG}.envs.connect_four.teacher" in sys.modules
  assert registry.load_extras_for("connect_four.prefer_center")
  assert not registry.load_extras_for("network")
  assert not registry.load_extras_for("chess.bot")


def test_spec_fields_are_validated():
  """Spec fields are validated."""
  with pytest.raises(ValueError, match="module:attr"):
    registry.GameSpec("g", "no_colon")
  with pytest.raises(ValueError, match="canon"):
    registry.GameSpec("g", "m:a", canon="nope")
  with pytest.raises(TypeError, match="extras"):
    registry.GameSpec("g", "m:a", extras=["a"])
  with pytest.raises(TypeError, match="components"):
    registry.GameSpec("g", "m:a", components=(("belief", "x"),))
  with pytest.raises(TypeError, match="components"):
    registry.GameSpec("g", "m:a", components=[("belief", "x", "m")])


def test_game_components_live_with_their_game():
  """Game components live with their game.

  A game's own belief samplers and slot factories are declared on its
  GameSpec, in its envs/<game>/ package: the generic registries name no game.
  """
  assert (
      registry.component_module("belief", "leduc")
      == f"{_PKG}.envs.leduc.belief"
  )
  assert (
      registry.component_module("belief", "leduc_card_removal")
      == f"{_PKG}.envs.leduc.belief"
  )
  assert registry.component_module("belief", "gin") == f"{_PKG}.envs.gin.belief"
  assert (
      registry.component_module("slot_factory", "gin_mctx_pimc")
      == f"{_PKG}.envs.gin.search"
  )
  assert registry.component_module("belief", "nope") is None
  assert set(registry.component_names("belief")) >= {
      "leduc",
      "leduc_card_removal",
      "gin",
  }
  pkg = registry.__name__.rsplit(".", 2)[0]  # <package>.envs.registry
  for g in registry.games():
    for _, _, module in g.components + tuple((None, None, m) for m in g.extras):
      if module.startswith(pkg + "."):
        assert module.startswith((pkg + ".envs.", pkg + ".encoders.")), (
            g.name,
            module,
        )


def test_slot_factories_find_a_game_factory_through_the_game_table():
  """Slot factories find a game factory through the game table."""
  assert "gin_mctx_pimc" not in slot_factories.LAZY
  with pytest.raises(KeyError, match="gin_mctx_pimc"):
    # the message lists the declared names
    slot_factories.get("no_such_factory", {})
  assert (
      registry.component_module("slot_factory", "gin_mctx_pimc")
      == gin_search.__name__
  )
  assert "gin_mctx_pimc" in slot_factories.names()


def test_a_component_declared_twice_with_different_modules_fails(monkeypatch):
  """A component declared twice with different modules fails.

  Args:
    monkeypatch: pytest monkeypatch fixture.
  """
  # Reaches private _GAMES of the code under test.
  monkeypatch.setattr(registry, "_GAMES", dict(registry._GAMES))  # pylint: disable=protected-access
  g = registry.game("kuhn_poker")
  registry.register(
      dataclasses.replace(g, components=(("belief", "gin", "elsewhere.mod"),)),
      replace=True,
  )
  with pytest.raises(ValueError, match="several modules"):
    registry.component_module("belief", "gin")


def test_no_env_name_is_registered_twice():
  """No env name is registered twice."""
  # register() accepts an equal row twice, so the tables themselves must list
  # each name once (two rows under one name would merge silently).
  names = [g.name for g in registry.BUILTIN]
  assert len(names) == len(set(names)), sorted(
      n for n in set(names) if names.count(n) > 1
  )
  assert set(names) <= set(registry.names())
  g = registry.game("kuhn_poker")
  with pytest.raises(ValueError, match="listed twice"):
    # Reaches private _unique_names of the code under test.
    registry._unique_names((g, dataclasses.replace(g)))  # pylint: disable=protected-access


def test_the_other_tables_are_views_of_the_registry():
  """The other tables are views of the registry."""
  assert set(config_common.env_names()) == set(registry.names())
  assert (
      registry.by_field("reference_evals")["gin_rummy.rule_bot"] == "gin_rummy"
  )
  assert runner.eval_names()["gin_rummy.rule_bot"] == "gin_rummy"
  assert (
      registry.by_field("endgame")["gin_rummy.scripted_endgame"] == "gin_rummy"
  )
  assert set(games.served_games()) >= set(registry.builtin_names()) - {
      "kuhn_poker_jax",
      "bridge",
  }
  assert (
      "kuhn_poker_jax" not in games.served_games()
      and "bridge" not in games.served_games()
  )
  for name in registry.builtin_names():
    assert config_common.build_env_by_name(name).name == name


def test_pyspiel_history_translators():
  """Pyspiel history translators."""
  gin_fn = registry.resolve(registry.game("gin_rummy").pyspiel_history_to_raw)
  assert gin_fn([5, 6], {"starting_player": 1}) == [1, 5, 6]
  with pytest.raises(ValueError, match="starting_player"):
    gin_fn([], {"starting_player": 2})
  # bargaining's chance ids are pyspiel's: its histories need no translation.
  assert registry.game("bargaining").pyspiel_history_to_raw is None
  assert registry.game("kuhn_poker").pyspiel_history_to_raw is None


def test_served_gin_variants():
  """Served gin variants."""
  assert registry.spec_of(
      games.env_for("gin_rummy", {"oklahoma": True})
  ) == registry.spec("gin_rummy", oklahoma_frac=1.0)
  assert registry.spec_of(games.env_for("gin_rummy", {})) == registry.spec(
      "gin_rummy"
  )
  with pytest.raises(ValueError, match="knock_card"):
    games.env_for("gin_rummy", {"knock_card": 9})


def test_run_config_refuses_bad_env_options():
  """Run config refuses bad env options."""
  cfg = config_common.load_config("mmd_kuhn_poker")
  with pytest.raises(TypeError, match="unknown env option"):
    dataclasses.replace(cfg, env_kwargs={"oklahoma_frac": 1.0})
  assert config_common.env_spec(cfg) == registry.spec("kuhn_poker")


def test_training_env_of_a_manifest():
  """Training env of a manifest."""
  m = types.Manifest(
      schema_version=2,
      game="gin_rummy",
      rules_sha="r",
      obs_format="o",
      arch_name="a",
      arch_kwargs={},
      framework="jax",
      capability=types.Capability.NATIVE_JAX,
      default_rule=types.ARGMAX,
      params_sha256="p",
      step=0,
      parents=(),
  )
  # Without an env_spec, a game with constructor options cannot be inferred;
  # a game without options is its one env.
  assert m.env_spec is None
  with pytest.raises(ValueError, match="no env_spec"):
    registry.training_env(m)
  assert (
      registry.training_env(m._replace(game="kuhn_poker")).name == "kuhn_poker"
  )
  m = m._replace(
      env_spec=registry.spec("gin_rummy", oklahoma_frac=1.0).to_json()
  )
  assert registry.training_env(m).oklahoma_frac == 1.0
  with pytest.raises(ValueError, match="not its game"):
    registry.training_env(m._replace(game="kuhn_poker"))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
