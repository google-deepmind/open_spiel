# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python spiel example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import pyspiel

# Put a bound on length of game so test does not timeout.
MAX_ACTIONS_PER_GAME = 1000

# All games registered in the main spiel library.
SPIEL_GAMES_LIST = pyspiel.registered_games()


# Check for mandatory parameters.
def _has_mandatory_params(game):
  return any(p.is_mandatory for p in game.parameter_specification.values())


# All games without mandatory parameters.
SPIEL_LOADABLE_GAMES_LIST = [
    g for g in SPIEL_GAMES_LIST if not _has_mandatory_params(g)
]
assert len(SPIEL_LOADABLE_GAMES_LIST) >= 40, len(SPIEL_LOADABLE_GAMES_LIST)

# All simultaneous games.
SPIEL_SIMULTANEOUS_GAMES_LIST = [
    g for g in SPIEL_LOADABLE_GAMES_LIST
    if g.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS
]
assert len(SPIEL_SIMULTANEOUS_GAMES_LIST) >= 14, len(
    SPIEL_SIMULTANEOUS_GAMES_LIST)

# All multiplayer games. This is a list of (game, num_players) pairs to test.
SPIEL_MULTIPLAYER_GAMES_LIST = [
    # pylint: disable=g-complex-comprehension
    (g, p)
    for g in SPIEL_LOADABLE_GAMES_LIST
    for p in range(max(g.min_num_players, 2), 1 + min(g.max_num_players, 6))
    if g.max_num_players > 2 and g.max_num_players > g.min_num_players
]
assert len(SPIEL_MULTIPLAYER_GAMES_LIST) >= 39, len(
    SPIEL_MULTIPLAYER_GAMES_LIST)


class GamesSimTest(parameterized.TestCase):

  def apply_action(self, state, action):
    if state.is_simultaneous_node():
      state.apply_actions(action)
    else:
      state.apply_action(action)

  def apply_action_test_clone(self, state, action):
    """Apply the action and test the clone method if it's implemented."""
    try:
      state_clone = state.clone()
      self.assertEqual(str(state), str(state_clone))
      self.assertEqual(state.history(), state_clone.history())
      self.apply_action(state, action)
      self.apply_action(state_clone, action)
      self.assertEqual(str(state), str(state_clone))
      self.assertEqual(state.history(), state_clone.history())
    except Exception:  # pylint: disable=broad-except
      self.apply_action(state, action)

  def serialize_deserialize(self, game, state):
    ser_str = pyspiel.serialize_game_and_state(game, state)
    new_game, new_state = pyspiel.deserialize_game_and_state(ser_str)
    self.assertEqual(str(game), str(new_game))
    self.assertEqual(str(state), str(new_state))

  def sim_game(self, game):
    min_utility = game.min_utility()
    max_utility = game.max_utility()
    self.assertLess(min_utility, max_utility)

    # Get a new state
    state = game.new_initial_state()
    total_actions = 0

    next_serialize_check = 1

    while not state.is_terminal() and total_actions <= MAX_ACTIONS_PER_GAME:
      total_actions += 1

      # Serialize/Deserialize is costly. Only do it every power of 2 actions.
      if total_actions >= next_serialize_check:
        self.serialize_deserialize(game, state)
        next_serialize_check *= 2

      # The state can be three different types: chance node,
      # simultaneous node, or decision node
      if state.is_chance_node():
        # Chance node: sample an outcome
        outcomes = state.chance_outcomes()
        self.assertNotEmpty(outcomes)
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)
      elif state.is_simultaneous_node():
        # Simultaneous node: sample actions for all players
        chosen_actions = [
            np.random.choice(state.legal_actions(pid))
            for pid in range(game.num_players())
        ]
        # Apply the joint action and test cloning states.
        self.apply_action_test_clone(state, chosen_actions)
      else:
        if state.is_terminal():
          self.assertEmpty(state.legal_actions())
          for player in range(game.num_players()):
            self.assertEmpty(state.legal_actions(player))
        # Decision node: sample action for the single current player
        action = np.random.choice(state.legal_actions(state.current_player()))
        # Apply action and test state cloning.
        self.apply_action_test_clone(state, action)

    # Max sure at least one action was made.
    self.assertGreater(total_actions, 0,
                       "No actions taken in sim of " + str(game))

    # Either the game is now done, or the maximum actions has been taken.
    if state.is_terminal():
      # Print utilities for each player.
      utilities = state.returns()
      # Check that each one is in range
      for utility in utilities:
        self.assertGreaterEqual(utility, game.min_utility())
        self.assertLessEqual(utility, game.max_utility())
      print("Sim of game {} terminated with {} total actions. Utilities: {}"
            .format(game, total_actions, utilities))
    else:
      print(
          "Sim of game {} terminated after maximum number of actions {}".format(
              game, MAX_ACTIONS_PER_GAME))

  @parameterized.parameters(*SPIEL_LOADABLE_GAMES_LIST)
  def test_game_sim(self, game_info):
    game = pyspiel.load_game(game_info.short_name)
    self.assertLessEqual(game_info.min_num_players, game.num_players())
    self.assertLessEqual(game.num_players(), game_info.max_num_players)
    self.sim_game(game)

  @parameterized.parameters(*SPIEL_SIMULTANEOUS_GAMES_LIST)
  def test_simultaneous_game_as_turn_based(self, game_info):
    converted_game = pyspiel.load_game_as_turn_based(game_info.short_name)
    self.sim_game(converted_game)

  @parameterized.parameters(*SPIEL_MULTIPLAYER_GAMES_LIST)
  def test_multiplayer_game(self, game_info, num_players):
    game = pyspiel.load_game(game_info.short_name,
                             {"players": pyspiel.GameParameter(num_players)})
    self.sim_game(game)

  def test_breakthrough(self):
    # make a smaller (6x6) board
    game = pyspiel.load_game("breakthrough(rows=6,columns=6)")
    self.sim_game(game)

  def test_pig(self):
    # make a smaller lower win score
    game = pyspiel.load_game("pig(players=2,winscore=15)")
    self.sim_game(game)


if __name__ == "__main__":
  absltest.main()
