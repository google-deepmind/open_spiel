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

"""game_string_of writes game strings the installed pyspiel loads.

Fast.

A parameter at pyspiel's default is dropped; a parameter the installed
pyspiel's game does not have is refused with a ValueError naming it.
"""

import sys

import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.serve import checkpoint_bot as checkpoint_bot_lib
from open_spiel.python.scale.serve import games
import pyspiel


def _manifest(game, env_spec):
  """A minimal Manifest of `game` and its EnvSpec.

  Args:
    game: The game.
    env_spec: The EnvSpec JSON.

  Returns:
    The Manifest.
  """
  return types.Manifest(
      schema_version=1,
      game=game,
      rules_sha="r",
      obs_format="o",
      arch_name="a",
      arch_kwargs={},
      framework="f",
      capability=None,
      default_rule=types.ARGMAX,
      params_sha256="p",
      step=1,
      parents=(),
      env_spec=env_spec,
  )


def _gin_manifest(**kwargs):
  """A gin Manifest with the given env options.

  Args:
    **kwargs: The env options.

  Returns:
    The Manifest.
  """
  return _manifest(
      "gin_rummy", env_registry.spec("gin_rummy", **kwargs).to_json()
  )


def _gin_spec():
  """gin_rummy's parameter specification in the installed pyspiel.

  Returns:
    The parameter specification.
  """
  return checkpoint_bot_lib.installed_parameter_specification("gin_rummy")


def test_game_string_params_drops_defaults():
  """A parameter at pyspiel's default is dropped; the others are kept."""
  spec = _gin_spec()
  params = {
      "gin_bonus": 25,
      "hand_size": 10,
      "knock_card": 10,
      "num_ranks": 13,
      "num_suits": 4,
      "oklahoma": True,
      "undercut_bonus": 25,
  }
  assert games.game_string_params("gin_rummy", params, spec) == {
      "oklahoma": True
  }
  assert not games.game_string_params("gin_rummy", {"oklahoma": False}, spec)
  # A non-default value is kept; a bool is not an int.
  assert games.game_string_params("gin_rummy", {"knock_card": 9}, spec) == {
      "knock_card": 9
  }
  assert games.game_string_params("gin_rummy", {"oklahoma": 1}, spec) == {
      "oklahoma": 1
  }


def test_game_string_params_refuses_a_parameter_pyspiel_lacks():
  """A parameter the pyspiel game does not have is refused, naming it."""
  spec = {k: v for k, v in _gin_spec().items() if k != "oklahoma"}
  with pytest.raises(ValueError, match="has no 'oklahoma' parameter"):
    games.game_string_params("gin_rummy", {"oklahoma": True}, spec)
  with pytest.raises(ValueError, match="has no 'oklahoma' parameter"):
    checkpoint_bot_lib.game_string_of(
        _gin_manifest(oklahoma_frac=1.0), spec
    )


def test_gin_oklahoma_string_loads():
  """An Oklahoma gin artifact's env (Oklahoma deals, first player at chance).

  The installed pyspiel loads its string, a game in which player 0 moves
  first.
  """
  m = _gin_manifest(oklahoma_frac=1.0)
  s = checkpoint_bot_lib.game_string_of(m, _gin_spec())
  assert s == "gin_rummy(oklahoma=true)"
  assert checkpoint_bot_lib.game_string_of(m) == s
  game = pyspiel.load_game(s)
  assert game.get_parameters()["oklahoma"]
  state = game.new_initial_state()
  while state.is_chance_node():
    state.apply_action(state.chance_outcomes()[0][0])
  assert state.current_player() == 0


def test_gin_default_artifact_is_the_short_name():
  """Gin default artifact is the short name."""
  assert (
      checkpoint_bot_lib.game_string_of(_gin_manifest(), _gin_spec())
      == "gin_rummy"
  )


@pytest.mark.parametrize("oklahoma", [False, True])
def test_installed_gin_parameters_are_served(oklahoma):
  """The installed pyspiel's gin_rummy, all its parameters, is served.

  A game reports every parameter it has (get_parameters).

  Args:
    oklahoma: The parametrized case.
  """
  game = pyspiel.load_game("gin_rummy", {"oklahoma": oklahoma})
  params = dict(game.get_parameters())
  games.check_params("gin_rummy", params)
  assert games.env_spec_for("gin_rummy", params) == env_registry.spec(
      "gin_rummy", oklahoma_frac=1.0 if oklahoma else None
  )


def test_gin_artifact_with_a_non_default_rule_has_no_game_string():
  """An artifact trained under a non-default gin rule has no game string."""
  odd = _gin_manifest(oklahoma_frac=1.0, allow_upcard_throwback=False)
  with pytest.raises(ValueError, match="no pyspiel game string"):
    checkpoint_bot_lib.game_string_of(odd, _gin_spec())


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
