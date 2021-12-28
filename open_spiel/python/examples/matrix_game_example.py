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

"""Python spiel example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from absl import app
import numpy as np

import pyspiel
from open_spiel.python.utils import file_utils


def _manually_create_game():
  """Creates the game manually from the spiel building blocks."""
  game_type = pyspiel.GameType(
      "matching_pennies",
      "Matching Pennies",
      pyspiel.GameType.Dynamics.SIMULTANEOUS,
      pyspiel.GameType.ChanceMode.DETERMINISTIC,
      pyspiel.GameType.Information.ONE_SHOT,
      pyspiel.GameType.Utility.ZERO_SUM,
      pyspiel.GameType.RewardModel.TERMINAL,
      2,  # max num players
      2,  # min_num_players
      True,  # provides_information_state
      True,  # provides_information_state_tensor
      False,  # provides_observation
      False,  # provides_observation_tensor
      dict()  # parameter_specification
  )
  game = pyspiel.MatrixGame(
      game_type,
      {},  # game_parameters
      ["Heads", "Tails"],  # row_action_names
      ["Heads", "Tails"],  # col_action_names
      [[-1, 1], [1, -1]],  # row player utilities
      [[1, -1], [-1, 1]]  # col player utilities
  )
  return game


def _easy_create_game():
  """Uses the helper function to create the same game as above."""
  return pyspiel.create_matrix_game("matching_pennies", "Matching Pennies",
                                    ["Heads", "Tails"], ["Heads", "Tails"],
                                    [[-1, 1], [1, -1]], [[1, -1], [-1, 1]])


def _even_easier_create_game():
  """Leave out the names too, if you prefer."""
  return pyspiel.create_matrix_game([[-1, 1], [1, -1]], [[1, -1], [-1, 1]])


def _import_data_create_game():
  """Creates a game via imported payoff data."""
  payoff_file = file_utils.find_file(
      "open_spiel/data/paper_data/response_graph_ucb/soccer.txt", 2)
  payoffs = np.loadtxt(payoff_file)*2-1
  return pyspiel.create_matrix_game(payoffs, payoffs.T)


def main(_):
  games_list = pyspiel.registered_games()
  print("Registered games:")
  print(games_list)

  # Load a two-player normal-form game as a two-player matrix game.
  blotto_matrix_game = pyspiel.load_matrix_game("blotto")
  print("Number of rows in 2-player Blotto with default settings is {}".format(
      blotto_matrix_game.num_rows()))

  # Several ways to load/create the same game of matching pennies.
  print("Creating matrix game...")
  game = pyspiel.load_matrix_game("matrix_mp")
  game = _manually_create_game()
  game = _import_data_create_game()
  game = _easy_create_game()
  game = _even_easier_create_game()

  # Quick test: inspect top-left utility values:
  print("Values for joint action ({},{}) is {},{}".format(
      game.row_action_name(0), game.col_action_name(0),
      game.player_utility(0, 0, 0), game.player_utility(1, 0, 0)))

  state = game.new_initial_state()

  # Print the initial state
  print("State:")
  print(str(state))

  assert state.is_simultaneous_node()

  # Simultaneous node: sample actions for all players.
  chosen_actions = [
      random.choice(state.legal_actions(pid))
      for pid in range(game.num_players())
  ]
  print("Chosen actions: ", [
      state.action_to_string(pid, action)
      for pid, action in enumerate(chosen_actions)
  ])
  state.apply_actions(chosen_actions)

  assert state.is_terminal()

  # Game is now done. Print utilities for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
  app.run(main)
