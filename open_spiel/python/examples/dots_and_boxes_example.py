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
#
# Contributed by Wannes Meert, Giuseppe Marra, and Pieter Robberechts
# for the KU Leuven course Machine Learning: Project.


"""Python spiel example."""

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 12761381, "The seed to use for the RNG.")

# Supported types of players: "random", "human"
flags.DEFINE_string("player0", "random", "Type of the agent for player 0.")
flags.DEFINE_string("player1", "random", "Type of the agent for player 1.")


def LoadAgent(agent_type, player_id, rng):
  """Return a bot based on the agent type."""
  if agent_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  elif agent_type == "human":
    return human.HumanBot()
  else:
    raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


def main(_):
  rng = np.random.RandomState(FLAGS.seed)
  games_list = pyspiel.registered_names()
  assert "dots_and_boxes" in games_list

  game_string = "dots_and_boxes(num_rows=2,num_cols=2)"
  print("Creating game: {}".format(game_string))
  game = pyspiel.load_game(game_string)

  agents = [
      LoadAgent(FLAGS.player0, 0, rng),
      LoadAgent(FLAGS.player1, 1, rng),
  ]

  state = game.new_initial_state()

  # Print the initial state
  print("INITIAL STATE")
  print(str(state))

  while not state.is_terminal():
    current_player = state.current_player()
    # Decision node: sample action for the single current player
    legal_actions = state.legal_actions()
    for action in legal_actions:
      print(
          "Legal action: {} ({})".format(
              state.action_to_string(current_player, action), action
          )
      )
    action = agents[current_player].step(state)
    action_string = state.action_to_string(current_player, action)
    print("Player ", current_player, ", chose action: ", action_string)
    state.apply_action(action)

    print("")
    print("NEXT STATE:")
    print(str(state))
    if not state.is_terminal():
      print(str(state.observation_tensor()))

  # Game is now done. Print utilities for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
  app.run(main)
