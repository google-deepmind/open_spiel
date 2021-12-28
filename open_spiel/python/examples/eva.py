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

"""Ephemeral Value Adjustment example: https://arxiv.org/abs/1810.08163."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import eva
from open_spiel.python.algorithms import exploitability
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", 1000, "Number of iterations")
flags.DEFINE_string("game_name", "kuhn_poker", "Name of the game")


class JointPolicy(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, agents):
    self._agents = agents

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    return self._agents[cur_player].action_probabilities(state)


def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  env = rl_environment.Environment(FLAGS.game_name)
  num_players = env.num_players
  num_actions = env.action_spec()["num_actions"]
  state_size = env.observation_spec()["info_state"][0]
  eva_agents = []
  with tf.Session() as sess:
    for player in range(num_players):
      eva_agents.append(
          eva.EVAAgent(
              sess,
              env,
              player,
              state_size,
              num_actions,
              embedding_network_layers=(64, 32),
              embedding_size=12,
              learning_rate=1e-4,
              mixing_parameter=0.5,
              memory_capacity=int(1e6),
              discount_factor=1.0,
              epsilon_start=1.0,
              epsilon_end=0.1,
              epsilon_decay_duration=int(1e6)))
    sess.run(tf.global_variables_initializer())
    for _ in range(FLAGS.num_episodes):
      time_step = env.reset()
      while not time_step.last():
        current_player = time_step.observations["current_player"]
        current_agent = eva_agents[current_player]
        step_out = current_agent.step(time_step)
        time_step = env.step([step_out.action])

      for agent in eva_agents:
        agent.step(time_step)

    game = pyspiel.load_game(FLAGS.game_name)
    joint_policy = JointPolicy(eva_agents)
    conv = exploitability.nash_conv(game, joint_policy)
    logging.info("EVA in '%s' - NashConv: %s", FLAGS.game_name, conv)


if __name__ == "__main__":
  app.run(main)
