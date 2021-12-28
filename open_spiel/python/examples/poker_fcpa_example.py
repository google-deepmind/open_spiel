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

from absl import app
from absl import flags
import numpy as np

from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 12761381, "The seed to use for the RNG.")

# Supported types of players: "random", "human", "check_call", "fold"
flags.DEFINE_string("player0", "random", "Type of the agent for player 0.")
flags.DEFINE_string("player1", "random", "Type of the agent for player 1.")


def LoadAgent(agent_type, game, player_id, rng):
  """Return a bot based on the agent type."""
  if agent_type == "random":
    return uniform_random.UniformRandomBot(player_id, rng)
  elif agent_type == "human":
    return human.HumanBot()
  elif agent_type == "check_call":
    policy = pyspiel.PreferredActionPolicy([1, 0])
    return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
  elif agent_type == "fold":
    policy = pyspiel.PreferredActionPolicy([0, 1])
    return pyspiel.make_policy_bot(game, player_id, FLAGS.seed, policy)
  else:
    raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


def main(_):
  rng = np.random.RandomState(FLAGS.seed)

  # Make sure poker is compiled into the library, as it requires an optional
  # dependency: the ACPC poker code. To ensure it is compiled in, prepend both
  # the install.sh and build commands with OPEN_SPIEL_BUILD_WITH_ACPC=ON.
  # See here:
  # https://github.com/deepmind/open_spiel/blob/master/docs/install.md#configuration-conditional-dependencies
  # for more details on optional dependencies.
  games_list = pyspiel.registered_names()
  assert "universal_poker" in games_list

  fcpa_game_string = pyspiel.hunl_game_string("fcpa")
  print("Creating game: {}".format(fcpa_game_string))
  game = pyspiel.load_game(fcpa_game_string)

  agents = [
      LoadAgent(FLAGS.player0, game, 0, rng),
      LoadAgent(FLAGS.player1, game, 1, rng)
  ]

  state = game.new_initial_state()

  # Print the initial state
  print("INITIAL STATE")
  print(str(state))

  while not state.is_terminal():
    # The state can be three different types: chance node,
    # simultaneous node, or decision node
    current_player = state.current_player()
    if state.is_chance_node():
      # Chance node: sample an outcome
      outcomes = state.chance_outcomes()
      num_actions = len(outcomes)
      print("Chance node with " + str(num_actions) + " outcomes")
      action_list, prob_list = zip(*outcomes)
      action = rng.choice(action_list, p=prob_list)
      print("Sampled outcome: ",
            state.action_to_string(state.current_player(), action))
      state.apply_action(action)
    else:
      # Decision node: sample action for the single current player
      legal_actions = state.legal_actions()
      for action in legal_actions:
        print("Legal action: {} ({})".format(
            state.action_to_string(current_player, action), action))
      action = agents[current_player].step(state)
      action_string = state.action_to_string(current_player, action)
      print("Player ", current_player, ", chose action: ",
            action_string)
      state.apply_action(action)

    print("")
    print("NEXT STATE:")
    print(str(state))

  # Game is now done. Print utilities for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
  app.run(main)
