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

"""Tests for belief.registry.

Samplers are built and recorded by name, game samplers
load on demand, and unknown or unregistered samplers fail loudly.
"""

import dataclasses
import sys

import pytest

from open_spiel.python.scale.belief import enumeration
from open_spiel.python.scale.belief import perfect_information
from open_spiel.python.scale.belief import registry
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs.leduc import belief as leduc_belief
from open_spiel.python.scale.policies import particle_belief as particle
from open_spiel.python.scale.policies import uniform


def _uniform(env):
  """A uniform policy for `env`.

  Args:
    env: The env.

  Returns:
    The policy.
  """
  return uniform.UniformPolicy(env.name, env.num_actions, env.num_players)


def test_a_sampler_a_game_declares_is_found_through_the_game_table(monkeypatch):
  """A sampler a game declares is found through the game table.

  belief.registry names no game module: a game's GameSpec.components
  declares its samplers, and `get` imports the declared module on demand.

  Args:
    monkeypatch: pytest's monkeypatch fixture.
  """
  # Reaches private _LAZY of the code under test.
  assert not {"leduc", "leduc_card_removal", "gin"} & set(registry._LAZY)  # pylint: disable=protected-access
  # Reaches private _GAMES of the code under test.
  monkeypatch.setattr(env_registry, "_GAMES", dict(env_registry._GAMES))  # pylint: disable=protected-access
  # Reaches private _CLASSES of the code under test.
  monkeypatch.setattr(registry, "_CLASSES", dict(registry._CLASSES))  # pylint: disable=protected-access
  g = env_registry.game("kuhn_poker")
  env_registry.register(
      dataclasses.replace(
          g,
          components=(
              ("belief", "kuhn_test_sampler", perfect_information.__name__),
          ),
      ),
      replace=True,
  )
  assert "kuhn_test_sampler" in registry.names()
  with pytest.raises(KeyError, match="kuhn_test_sampler"):
    # the module imports but registers another name
    registry.get("kuhn_test_sampler")
  registry.register(
      "kuhn_test_sampler", perfect_information.PerfectInformationSampler
  )
  assert (
      registry.get("kuhn_test_sampler")
      is perfect_information.PerfectInformationSampler
  )


def test_shipped_samplers_are_registered():
  """Shipped samplers are registered."""
  assert {
      "perfect_information",
      "enumeration",
      "particle",
      "leduc",
      "gin",
  } <= set(registry.names())
  assert (
      registry.get("perfect_information")
      is perfect_information.PerfectInformationSampler
  )
  assert registry.get("enumeration") is enumeration.EnumerationSampler
  assert registry.get("particle") is particle.ParticleSampler
  assert registry.get("leduc") is leduc_belief.LeducBeliefSampler


def test_every_sampler_declares_the_protocol_attributes():
  """Every sampler declares the protocol attributes."""
  for name in (
      "perfect_information",
      "enumeration",
      "particle",
      "leduc",
      "gin",
  ):
    cls = registry.get(name)
    assert cls.is_perfect_information in (True, False), name
    assert cls.params_source in registry.PARAMS_SOURCES, name


def test_build_passes_the_opponent_only_to_samplers_that_take_one():
  """Build passes the opponent only to samplers that take one."""
  c4 = env_registry.make("connect_four")
  assert isinstance(
      registry.build("perfect_information", c4),
      perfect_information.PerfectInformationSampler,
  )
  le = env_registry.make("leduc_poker")
  s = registry.build("leduc", le, opponent=_uniform(le))
  assert isinstance(s, leduc_belief.LeducBeliefSampler)
  assert registry.takes_opponent("leduc") and not registry.takes_opponent(
      "enumeration"
  )
  with pytest.raises(ValueError, match="pass opponent"):
    registry.build("leduc", le)
  with pytest.raises(ValueError, match="takes no opponent"):
    registry.build("perfect_information", c4, opponent=_uniform(c4))


def test_name_and_kwargs_rebuild_the_sampler():
  """Name and kwargs rebuild the sampler."""
  le = env_registry.make("leduc_poker")
  s = registry.build("leduc", le, opponent=_uniform(le))
  name = registry.name_of(s)
  assert name == "leduc"
  again = registry.build(name, le, **s.kwargs)
  assert type(again) is type(s) and again.kwargs == s.kwargs


def test_unknown_and_unregistered_samplers_fail_loud():
  """Unknown and unregistered samplers fail loud."""
  with pytest.raises(KeyError, match="no belief sampler"):
    registry.get("nope")

  class Rogue(perfect_information.PerfectInformationSampler):
    """A sampler subclass that is never registered."""

    pass

  with pytest.raises(KeyError, match="not a registered"):
    registry.name_of(Rogue(env_registry.make("tic_tac_toe")))


def test_register_refuses_a_second_class_under_one_name_and_allows_aliases():
  """Register refuses a second class under one name and allows aliases."""
  cls = registry.get("perfect_information")
  assert registry.register("perfect_information", cls) is cls
  with pytest.raises(ValueError, match="already registered"):
    registry.register("perfect_information", enumeration.EnumerationSampler)
  registry.register("perfect_information_alias_for_test", cls)
  sampler = cls(env_registry.make("tic_tac_toe"))
  assert registry.name_of(sampler) == "perfect_information"


def test_card_removal_sampler_is_built_from_the_env_alone():
  """Card removal sampler is built from the env alone."""
  le = env_registry.make("leduc_poker")
  s = registry.build("leduc_card_removal", le)
  assert isinstance(s, leduc_belief.LeducCardRemovalSampler)
  assert s.params_source == "none" and not s.kwargs
  assert registry.name_of(s) == "leduc_card_removal"
  assert not registry.takes_opponent("leduc_card_removal")


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
