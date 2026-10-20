# Copyright 2019 DeepMind Technologies Limited
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

"""Conformance tests for the struct/JSON API.

Every game that implements State::ToStruct or State::ToObservationStruct is
checked here, so a game gains this coverage as soon as it adopts the API.
Game-specific expectations belong in the game's own test; what is checked here
are the properties that must hold for every game:

  - the structs serialize to JSON objects, and to_dict() agrees with to_json(),
  - an observation struct reports the player it was requested for,
  - in a perfect information game, all players observe the same thing,
  - a state struct that can be turned back into a state round-trips.
"""

import contextlib
import json
import os
import random
import sys

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel

# Keep the playouts short: this is an API conformance test, not a sim test.
MAX_ACTIONS_PER_GAME = 30
SEED = 87126

# Games known to implement the struct API. Guards against the discovery below
# silently returning nothing, which would turn this test into a no-op.
EXPECTED_STRUCT_GAMES = frozenset(
    ["catch", "connect_four", "gin_rummy", "go", "hearts", "tic_tac_toe"]
)


@contextlib.contextmanager
def _muted_stderr():
  """Mutes the C++ error stream while probing for unimplemented methods.

  Struct support is optional, and the only way to ask a game whether it
  implements a given method is to call it. The C++ layer prints to stderr
  before raising, which would bury the test output.
  """
  sys.stderr.flush()
  stderr_fd = sys.stderr.fileno()
  saved_fd = os.dup(stderr_fd)
  try:
    with open(os.devnull, "w") as devnull:
      os.dup2(devnull.fileno(), stderr_fd)
    yield
  finally:
    os.dup2(saved_fd, stderr_fd)
    os.close(saved_fd)


def _probe_struct_api(game):
  """Returns which parts of the struct API a game implements."""
  state = game.new_initial_state()
  has_state_struct = True
  has_observation_struct = True
  has_new_initial_state = False
  with _muted_stderr():
    try:
      state_struct = state.to_struct()
    except pyspiel.SpielError:
      has_state_struct = False
    else:
      try:
        game.new_initial_state(state_struct)
        has_new_initial_state = True
      except (pyspiel.SpielError, TypeError):
        pass
    try:
      state.to_observation_struct(0)
    except pyspiel.SpielError:
      has_observation_struct = False
  return has_state_struct, has_observation_struct, has_new_initial_state


def _struct_games():
  """All loadable games implementing any part of the struct API."""
  games_list = []
  for game_info in pyspiel.registered_games():
    if not game_info.default_loadable:
      continue
    try:
      game = pyspiel.load_game(game_info.short_name)
      support = _probe_struct_api(game)
    except (pyspiel.SpielError, RuntimeError):
      continue
    if support[0] or support[1]:
      games_list.append((game_info.short_name,) + support)
  return games_list


SPIEL_LOADABLE_GAMES_LIST = [
    game_info
    for game_info in pyspiel.registered_games()
    if game_info.default_loadable
]


class GamesStructTest(parameterized.TestCase):

  def assert_json_object(self, struct, description):
    """A struct must serialize to a JSON object agreeing with to_dict()."""
    parsed = json.loads(struct.to_json())
    self.assertIsInstance(parsed, dict, f"{description} is not a JSON object")
    self.assertEqual(parsed, struct.to_dict(), f"{description} to_dict()")
    return parsed

  def check_state(self, game, state, support):
    """Checks every struct invariant at a single state."""
    has_state_struct, has_observation_struct, has_new_initial_state = support
    if has_state_struct:
      state_struct = state.to_struct()
      state_json = self.assert_json_object(state_struct, "state struct")
      self.assertEqual(json.loads(state.to_json()), state_json)
      # A state struct that the game can turn back into a state must describe
      # that state completely.
      if has_new_initial_state:
        restored = game.new_initial_state(state_struct)
        self.assertEqual(json.loads(restored.to_json()), state_json)

    if not has_observation_struct:
      return
    observations = []
    for player in range(game.num_players()):
      observation = state.to_observation_struct(player)
      observation_json = self.assert_json_object(
          observation, f"observation struct of player {player}"
      )
      # An observation is always from the point of view it was asked for.
      if "observing_player" in observation_json:
        self.assertEqual(observation_json["observing_player"], player)
      observation_json.pop("observing_player", None)
      observations.append(observation_json)

    perfect_information = (
        game.get_type().information
        == pyspiel.GameType.Information.PERFECT_INFORMATION
    )
    if perfect_information:
      for player, observation_json in enumerate(observations):
        self.assertEqual(
            observation_json,
            observations[0],
            f"player {player} observes a different state than player 0 in a "
            "perfect information game",
        )

  @parameterized.named_parameters(
      (game_info.short_name, game_info.short_name)
      for game_info in SPIEL_LOADABLE_GAMES_LIST
  )
  def test_game_structs(self, game_name):
    try:
      game = pyspiel.load_game(game_name)
      support = _probe_struct_api(game)
    except (pyspiel.SpielError, RuntimeError):
      return
    if not (support[0] or support[1]):
      return
    rng = random.Random(SEED)
    state = game.new_initial_state()
    for _ in range(MAX_ACTIONS_PER_GAME):
      self.check_state(game, state, support)
      if state.is_terminal():
        return
      if state.is_chance_node():
        outcomes, probabilities = zip(*state.chance_outcomes())
        state.apply_action(rng.choices(outcomes, probabilities)[0])
      elif state.is_simultaneous_node():
        state.apply_actions([
            rng.choice(state.legal_actions(player))
            for player in range(game.num_players())
        ])
      else:
        state.apply_action(rng.choice(state.legal_actions()))
    # The structs must also work once the game is over.
    while not state.is_terminal() and state.move_number() < 1000:
      if state.is_chance_node():
        outcomes, probabilities = zip(*state.chance_outcomes())
        state.apply_action(rng.choices(outcomes, probabilities)[0])
      elif state.is_simultaneous_node():
        state.apply_actions([
            rng.choice(state.legal_actions(player))
            for player in range(game.num_players())
        ])
      else:
        state.apply_action(rng.choice(state.legal_actions()))
    self.check_state(game, state, support)

  def test_struct_games_are_discovered(self):
    discovered = {game[0] for game in _struct_games()}
    self.assertContainsSubset(EXPECTED_STRUCT_GAMES, discovered)


if __name__ == "__main__":
  absltest.main()
