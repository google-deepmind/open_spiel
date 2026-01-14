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

"""Example to traverse an entire game tree."""

from absl import app
from absl import flags

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel

_GAME_STRING = flags.DEFINE_string(
    "game_string", "tic_tac_toe", "Name of the game"
)


class GameStats:
  num_states: int = 0
  num_chance_nodes: int = 0
  num_decision_nodes: int = 0
  num_simultaneous_nodes: int = 0
  num_terminals: int = 0
  info_state_dict: dict[str, list[int]] = {}

  def __str__(self):
    return (f"Number of states {self.num_states} \n" +
            f"Number of chance nodes {self.num_chance_nodes} \n" +
            f"Number of decision nodes {self.num_decision_nodes} \n" +
            f"Number of simultaneous nodes {self.num_simultaneous_nodes} \n" +
            f"Number of terminals {self.num_terminals} \n")


def traverse_game_tree(game: pyspiel.Game,
                       state: pyspiel.State,
                       game_stats: GameStats):
  """Traverses the game tree, collecting information about the game."""

  if state.is_terminal():
    game_stats.num_terminals += 1
  elif state.is_chance_node():
    game_stats.num_chance_nodes += 1
    for outcome in state.legal_actions():
      child = state.child(outcome)
      traverse_game_tree(game, child, game_stats)
  elif state.is_simultaneous_node():
    game_stats.num_simultaneous_nodes += 1
    # Using joint actions for convenience. Can use legal_actions(player) to
    # and state.apply_actions when walking over individual players
    for joint_action in state.legal_actions():
      child = state.child(joint_action)
      traverse_game_tree(game, child, game_stats)
  else:
    game_stats.num_decision_nodes += 1
    legal_actions = state.legal_actions()
    if game.get_type().provides_information_state_string:
      game_stats.info_state_dict[
          state.information_state_string()] = legal_actions
    for action in state.legal_actions():
      # print(f"Decision node: \n {state}")
      # print(f"Taking action {action} ({state.action_to_string(action)}")
      child = state.child(action)
      traverse_game_tree(game, child, game_stats)


def main(_):
  game = pyspiel.load_game(_GAME_STRING.value)
  game_stats = GameStats()
  state = game.new_initial_state()
  traverse_game_tree(game, state, game_stats)
  print(game_stats)
  # for info_state_string in game_stats.info_state_dict:
  #   print(info_state_string)
  #   # print(game_stats.info_state_dict[info_state_string])  # legal actions


if __name__ == "__main__":
  app.run(main)
