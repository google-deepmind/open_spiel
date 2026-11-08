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

"""Whether a pyspiel BUILD of a served game is the game its env plays.

A served game's name and parameters pick its env (serve.games); its
ServingSpec.game_check (envs.registry) also checks the pyspiel game object
itself, for an engine whose released and corrected builds share a name and
parameters but not the rules (euchre: released OpenSpiel's engine against
the corrected one the env plays). The serving entry points that get a pyspiel
game call this before they decide: the GUI's availability check
(controller.refusal), the checkpoint seat and the Remote bot
(CheckpointBot.check_serves), and any caller that answers pyspiel
states directly through PolicyServer.decide (which should call
check_pyspiel_game on the state's game before deciding; PolicyServer itself
does not).
"""

from typing import Any, Optional

from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.serve import games


def pyspiel_game_problem(game: Any) -> Optional[str]:
  """Why the pyspiel game object is not the game its env plays, or None.

  A served game's ServingSpec.game_check refuses a build of the game whose
  rules are not the env's although its name and parameters are (euchre: a
  released pyspiel's engine against the corrected one the env plays). A game
  none of whose envs has a game_check is None here (never asked for its
  parameters); for one that has, a variant no env plays is refused too.

  Args:
    game: the pyspiel game.

  Returns:
    The refusal, or None.
  """
  name = game.get_type().short_name
  if not any(
      games.pyspiel_name(n) == name
      and env_registry.game(n).serving.game_check is not None
      for n in games.served_games()
  ):
    return None  # no env of this pyspiel game checks the build
  try:
    env_name = games.env_name_for(name, dict(game.get_parameters()))
  except (KeyError, ValueError) as e:
    return f"no env serves this {name}: {e}"
  check = env_registry.game(env_name).serving.game_check
  if check is None:
    return None
  try:
    check(game)
  except ValueError as e:
    return str(e)
  return None


def check_pyspiel_game(game: Any) -> None:
  """Raises when the pyspiel game object is not the game its env plays.

  Args:
    game: the pyspiel game.

  Raises:
    ValueError: pyspiel_game_problem's refusal.
  """
  problem = pyspiel_game_problem(game)
  if problem is not None:
    raise ValueError(problem)
