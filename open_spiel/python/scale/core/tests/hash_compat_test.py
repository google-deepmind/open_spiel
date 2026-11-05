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

"""core.hash_compat: added config fields.

They are hashed only away from their default.

* a config class with an added field at its default hashes exactly as the
  class without the field; set to anything else, the field is hashed;
* registered hash transforms still see the full key set (a transform that
  recognises a dataclass dict by its keys keeps working), and the field is
  dropped after them;
* the N-player fields are declared (pinned list) and every in-repo config
  hashes without them;
* every non-default value of a declared-but-unimplemented field is refused.
"""

import dataclasses
import sys
from typing import Any, Optional

import pytest

from open_spiel.python.scale.core import hash_compat
from open_spiel.python.scale.core import sides
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import controller
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.testing import registries


@dataclasses.dataclass(frozen=True)
class _Old:
  a: int = 1
  b: str = "x"


@dataclasses.dataclass(frozen=True)
class _New:
  a: int = 1
  b: str = "x"
  c: Optional[tuple[Any, ...]] = hash_compat.added(None)
  d: str = hash_compat.added("auto")


@dataclasses.dataclass(frozen=True)
class _Outer:
  inner: object
  items: tuple[Any, ...] = ()


def test_added_fields_at_default_hash_as_the_old_class():
  """Added fields at default hash as the old class."""
  assert (
      common.config_dict(_New())
      == common.config_dict(_Old())
      == {"a": 1, "b": "x"}
  )
  assert common.config_hash(_New(a=3)) == common.config_hash(_Old(a=3))
  nested_new = _Outer(_New(), (_New(b="y"),))
  nested_old = _Outer(_Old(), (_Old(b="y"),))
  assert common.config_hash(nested_new) == common.config_hash(nested_old)


def test_added_field_away_from_default_is_hashed():
  """Added field away from default is hashed."""
  assert common.config_dict(_New(d="vs_field")) == {
      "a": 1,
      "b": "x",
      "d": "vs_field",
  }
  assert common.config_dict(_New(c=(1, 2)))["c"] == [1, 2]
  assert common.config_hash(_New(d="vs_field")) != common.config_hash(_Old())
  # An equal value counts as the default (plain form).
  assert "c" not in common.config_dict(_New(c=None))


def test_plain_keeps_every_field():
  """Plain keeps every field."""
  assert common.plain(_New()) == {"a": 1, "b": "x", "c": None, "d": "auto"}


def test_transforms_see_the_full_key_set(monkeypatch):
  """Transforms see the full key set."""
  monkeypatch.setattr(common, "_HASH_TRANSFORMS", list(common._HASH_TRANSFORMS))  # pylint: disable=protected-access
  keys = frozenset(f.name for f in dataclasses.fields(_New))
  seen = []

  def by_keyset(d):
    """Renames field b of the dict with exactly `keys`.

    Args:
      d: a hashed dict.

    Returns:
      The dict, transformed when its key set is `keys`.
    """
    if frozenset(d) == keys:
      seen.append(True)
      return dict(d, b="renamed")
    return d

  common.register_hash_transform(by_keyset)
  assert common.config_dict(_New()) == {"a": 1, "b": "renamed"}
  assert seen == [True]


def test_a_transform_that_moves_the_marker_into_a_list_fails_loud(monkeypatch):
  """A transform that moves the marker into a list fails loud."""
  monkeypatch.setattr(common, "_HASH_TRANSFORMS", list(common._HASH_TRANSFORMS))  # pylint: disable=protected-access
  common.register_hash_transform(lambda d: dict(d, moved=[d.get("d")]))
  with pytest.raises(ValueError, match="list element"):
    common.config_dict(_New())


def test_mutable_defaults_are_refused():
  """Mutable defaults are refused."""
  with pytest.raises(TypeError):
    hash_compat.added([])


#: The fields the N-player work added (core.hash_compat.added), pinned:
#: adding one here means it is declared with its default and refused until
#: implemented. PGConfig.teacher_coef is the teacher-distillation coefficient
#: (learners.pg module doc "Teacher distillation"), off at its default.
#: LeagueConfig.fixed_layout is the Arena layout of the fixed opponents
#: (league.loop LeagueConfig doc); its default "stacked" is the layout every
#: recorded run used, so it is hash-neutral there.
DECLARED = {
    loop.LeagueConfig: {
        "lineup": None,
        "payoff_view": "auto",
        "eval_design": "auto",
        "fixed_layout": "stacked",
    },
    runner.InLoopEval: {"design": "auto"},
    pg.PGConfig: {"credit": "seat", "teacher_coef": pg.TEACHER_COEF_OFF},
}


@pytest.mark.parametrize("cls", list(DECLARED), ids=lambda c: c.__name__)
def test_n_player_fields_are_declared(cls):
  """N player fields are declared."""
  assert hash_compat.added_fields(cls) == DECLARED[cls]


def test_gate_request_field_defaults_to_none():
  """Gate request field defaults to none."""
  # Default None keeps the two-sided head-to-head gate; a field is a
  # members-only sampler.Mix (the N-player substitution gate) and anything else
  # is refused.
  req = controller.GateRequest("main", "main@0", 1, 7, 16)
  assert req.field is None
  with pytest.raises(TypeError):
    controller.GateRequest("main", "main@0", 1, 7, 16, field=object())
  with pytest.raises(ValueError):
    controller.GateRequest(
        "main", "main@0", 1, 7, 16, field=sampler.Mix(1.0, {})
    )
  field = sampler.Mix(0.0, {"main@0": 1.0})
  assert (
      controller.GateRequest("main", "main@0", 1, 7, 16, field=field).field
      == field
  )


def _walk_keys(d, out):
  """Adds every dict key under `d` to `out`.

  Args:
    d: A plain config value.
    out: The key set, updated.

  Returns:
    out.
  """
  if isinstance(d, dict):
    for k, v in d.items():
      out.add(k)
      _walk_keys(v, out)
  elif isinstance(d, list):
    for v in d:
      _walk_keys(v, out)
  return out


# The package's own configs: read at collection, before the conftest's open
# view, so in its own (a session that loaded a plugin lists its configs too).
with registries.open_view():
  _CONFIG_NAMES = common.config_names()


@pytest.mark.parametrize("name", _CONFIG_NAMES)
def test_in_repo_configs_hash_without_the_n_player_fields(name):
  """In repo configs hash without the n player fields."""
  cfg = common.load_config(name)
  keys = _walk_keys(common.config_dict(cfg), set())
  assert not keys & {"lineup", "payoff_view", "eval_design", "credit"}, name
  assert hash_compat.OMITTED not in str(common.config_dict(cfg))


def test_unimplemented_values_are_refused():
  """Unimplemented values are refused."""
  base = dict(num_envs=8, num_steps=4, updates_per_iteration=1, pool_capacity=0)
  with pytest.raises(TypeError):
    loop.LeagueConfig(**base, lineup=("learner", "pool"))
  with pytest.raises(ValueError, match="one or the other"):
    loop.LeagueConfig(
        **base, self_play_seats="all", lineup=sides.lineup(("learner", "pool"))
    )
  with pytest.raises(NotImplementedError):
    loop.LeagueConfig(**base, payoff_view="vs_field")
  with pytest.raises(ValueError, match="eval_design"):
    loop.LeagueConfig(**base, eval_design="substitute")
  with pytest.raises(ValueError, match="design"):
    runner.InLoopEval("vs_uniform", every=1, deals=4, design="swap2+partner")
  with pytest.raises(ValueError, match="credit"):
    pg.PGConfig(credit="team")


def test_implemented_values_enter_the_hash_off_their_default():
  """Implemented lineup and eval designs are accepted and hashed off default.

  The lineup and eval designs the league implements (league.loop,
  league.runner) are accepted, and a value off its default is hashed.
  """
  base = dict(num_envs=8, num_steps=4, updates_per_iteration=1, pool_capacity=0)
  lu = sides.lineup(("learner", "pool"))
  assert loop.LeagueConfig(**base, lineup=lu).lineup == lu
  assert loop.LeagueConfig(**base, eval_design="cyclic").eval_design == "cyclic"
  assert (
      runner.InLoopEval("vs_uniform", every=1, deals=4, design="cyclic").design
      == "cyclic"
  )
  plain = loop.hashed_config(loop.LeagueConfig(**base))
  assert not set(plain) & {"lineup", "payoff_view", "eval_design"}
  assert "lineup" in loop.hashed_config(loop.LeagueConfig(**base, lineup=lu))
  # Side credit (N-player S8) is implemented: accepted, and hashed off its
  # default only.
  assert pg.PGConfig(anneal_lr=False, credit="side").credit == "side"
  seat = common.config_dict(pg.PGConfig(anneal_lr=False))
  side = common.config_dict(pg.PGConfig(anneal_lr=False, credit="side"))
  assert "credit" not in seat and side["credit"] == "side"
  assert (
      loop.hashed_config(loop.LeagueConfig(**base, eval_design="cyclic"))[
          "eval_design"
      ]
      == "cyclic"
  )


# -- recorded forms (hash_compat module doc "Recorded forms") ----------------


@pytest.fixture(name="registry")
def _registry(monkeypatch):
  """Gives the test its own copy of the recorded-form registries.

  Args:
    monkeypatch: pytest's monkeypatch fixture.

  Returns:
    The hash_compat module, its registries copied for this test.
  """
  for attr, copy in (
      ("_ADDED", dict),
      ("_CLASS_TRANSFORMS", lambda d: {k: list(v) for k, v in d.items()}),
      ("_KEYSETS", dict),
      ("_STRING_TRANSFORMS", list),
  ):
    monkeypatch.setattr(hash_compat, attr, copy(getattr(hash_compat, attr)))
  return hash_compat


@dataclasses.dataclass(frozen=True)
class _Later:
  a: int = 1
  b: str = "x"
  e: int = 0
  f: tuple[Any, ...] = ()


def test_recorded_form_with_nothing_registered_changes_nothing(registry):
  """With nothing registered for a class the recorded form is plain."""
  assert common.hashed_form(_Later()) == common.plain(_Later())
  before = common.config_hash(_Later(e=2))
  registry.register_string_transform(lambda s: s)
  assert common.config_hash(_Later(e=2)) == before


def test_an_added_group_is_left_out_only_while_all_of_it_is_at_default(
    registry,
):
  """A group leaves the hash at its defaults, and is hashed whole otherwise."""
  registry.register_added_fields(_Later, {"e": 0, "f": ()})
  assert common.config_dict(_Later()) == common.config_dict(_Old())
  assert common.config_hash(_Later(a=4)) == common.config_hash(_Old(a=4))
  # One field away from its default: the whole group is hashed.
  assert common.config_dict(_Later(f=(1,))) == {
      "a": 1,
      "b": "x",
      "e": 0,
      "f": [1],
  }
  assert common.hashed_form(_Later()) == {"a": 1, "b": "x"}


def test_a_group_predicate_must_also_hold(registry):
  """The group's predicate on the original dict must hold as well."""
  registry.register_added_fields(_Later, {"e": 0}, lambda d: d["a"] == 1)
  assert "e" not in common.config_dict(_Later())
  assert common.config_dict(_Later(a=2))["e"] == 0


def test_class_and_string_transforms_rewrite_the_recorded_form(registry):
  """Class transforms rewrite one class's dict; string transforms every str."""
  registry.register_class_transform(
      _Later, lambda d: {("old_b" if k == "b" else k): v for k, v in d.items()}
  )
  registry.register_string_transform(lambda s: s.replace("new.mod", "old.mod"))
  got = common.config_dict(_Outer(_Later(b="new.mod:f"), ("new.mod:g",)))
  assert got == {
      "inner": {"a": 1, "old_b": "old.mod:f", "e": 0, "f": []},
      "items": ["old.mod:g"],
  }


def test_registration_refuses_what_it_cannot_honour(registry):
  """Wrong registrations raise instead of silently hashing differently."""
  with pytest.raises(TypeError, match="not a dataclass"):
    registry.register_added_fields(dict, {"a": 1})
  with pytest.raises(ValueError, match="not fields of _Later"):
    registry.register_added_fields(_Later, {"nope": 0})
  with pytest.raises(TypeError, match="not callable"):
    registry.register_class_transform(_Later, "not callable")
  with pytest.raises(TypeError, match="not callable"):
    registry.register_string_transform(None)

  @dataclasses.dataclass(frozen=True)
  class _Twin:  # the same field names as _Later
    a: int = 1
    b: str = "x"
    e: int = 0
    f: tuple[Any, ...] = ()

  registry.register_added_fields(_Later, {"e": 0})
  with pytest.raises(ValueError, match="ambiguous"):
    registry.register_class_transform(_Twin, lambda d: d)


def test_the_recorded_form_sees_core_added_fields_by_their_keys(registry):
  """A registered class with a core-added field is still recognised.

  The added field holds OMITTED while the recorded form runs, so the class's
  key set is complete; the field is dropped afterwards.

  Args:
    registry: hash_compat with this test's own copy of its registries.
  """
  registry.register_class_transform(_New, lambda d: dict(d, b="renamed"))
  assert common.config_dict(_New()) == {"a": 1, "b": "renamed"}


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
