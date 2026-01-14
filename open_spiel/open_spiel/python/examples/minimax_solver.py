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
"""Example usage of the minimax solver."""

from absl import app
from absl import flags

from open_spiel.python.algorithms import minimax_solver
import pyspiel

FLAGS = flags.FLAGS

# Game strings can just contain the name or the name followed by parameters
# and arguments, e.g. "breakthrough(rows=6,columns=6)"
flags.DEFINE_string("game_string", "tic_tac_toe", "Game string")


def main(_):
  print("Creating game.")
  game = pyspiel.load_game(FLAGS.game_string)
  state = game.new_initial_state()
  print(state)

  print("Creating minimax oracle and solving.")
  solver = minimax_solver.MinimaxSolver(FLAGS.game_string)
  solver.solve()

  print("Playing game.")
  while not state.is_terminal():
    print("")
    print(state)
    # Decision node: sample action for the single current player
    action_values = solver.action_values_from_state(state)
    best_value = float("-inf")
    best_action = pyspiel.INVALID_ACTION
    for action in state.legal_actions():
      action_value = action_values[action]
      print(
          f"Action {state.action_to_string(action)} "
          + f"has minimax value: {action_value}"
      )
      if action_value > best_value:
        best_value = action_value
        best_action = action
    print(
        f"Applying action {best_action}: "
        + f"{state.action_to_string(best_action)}"
    )
    state.apply_action(best_action)

  # Game is now done. Print utilities for each player
  print("")
  print(str(state))
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
  app.run(main)
