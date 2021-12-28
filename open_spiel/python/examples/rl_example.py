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

import logging
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import rl_environment

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "tic_tac_toe", "Name of the game")
flags.DEFINE_integer("num_players", None, "Number of players")


def select_actions(observations, cur_player):
  cur_legal_actions = observations["legal_actions"][cur_player]
  actions = [np.random.choice(cur_legal_actions)]
  return actions


def print_iteration(time_step, actions, player_id):
  """Print TimeStep information."""
  obs = time_step.observations
  logging.info("Player: %s", player_id)
  if time_step.step_type.first():
    logging.info("Info state: %s, - - %s", obs["info_state"][player_id],
                 time_step.step_type)
  else:
    logging.info("Info state: %s, %s %s %s", obs["info_state"][player_id],
                 time_step.rewards[player_id], time_step.discounts[player_id],
                 time_step.step_type)
  logging.info("Action taken: %s", actions)
  logging.info("-" * 80)


def turn_based_example(unused_arg):
  """Example usage of the RL environment for turn-based games."""
  # `rl_main_loop.py` contains more details and simultaneous move examples.
  logging.info("Registered games: %s", rl_environment.registered_games())
  logging.info("Creating game %s", FLAGS.game)

  env_configs = {"players": FLAGS.num_players} if FLAGS.num_players else {}
  env = rl_environment.Environment(FLAGS.game, **env_configs)

  logging.info("Env specs: %s", env.observation_spec())
  logging.info("Action specs: %s", env.action_spec())

  time_step = env.reset()

  while not time_step.step_type.last():
    pid = time_step.observations["current_player"]
    actions = select_actions(time_step.observations, pid)
    print_iteration(time_step, actions, pid)
    time_step = env.step(actions)

  # Print final state of end game.
  for pid in range(env.num_players):
    print_iteration(time_step, actions, pid)


if __name__ == "__main__":
  app.run(turn_based_example)
