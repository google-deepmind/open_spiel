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

"""DQN agent vs Tabular Q-Learning agents trained on Tic Tac Toe.

The two agents are trained by playing against each other. Then, the game
can be played against the DQN agent from the command line.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of train episodes.")
flags.DEFINE_boolean(
    "iteractive_play", True,
    "Whether to run an interactive play with the agent after training.")


def pretty_board(time_step):
  """Returns the board in `time_step` in a human readable format."""
  info_state = time_step.observations["info_state"][0]
  x_locations = np.nonzero(info_state[9:18])[0]
  o_locations = np.nonzero(info_state[18:])[0]
  board = np.full(3 * 3, ".")
  board[x_locations] = "X"
  board[o_locations] = "0"
  board = np.reshape(board, (3, 3))
  return board


def command_line_action(time_step):
  """Gets a valid action from the user on the command line."""
  current_player = time_step.observations["current_player"]
  legal_actions = time_step.observations["legal_actions"][current_player]
  action = -1
  while action not in legal_actions:
    print("Choose an action from {}:".format(legal_actions))
    sys.stdout.flush()
    action_str = input()
    try:
      action = int(action_str)
    except ValueError:
      continue
  return action


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes


def main(_):
  game = "tic_tac_toe"
  num_players = 2
  env = rl_environment.Environment(game)
  state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [32, 32]
  replay_buffer_capacity = int(1e4)
  train_episodes = FLAGS.num_episodes
  loss_report_interval = 1000

  with tf.Session() as sess:
    dqn_agent = dqn.DQN(
        sess,
        player_id=0,
        state_representation_size=state_size,
        num_actions=num_actions,
        hidden_layers_sizes=hidden_layers_sizes,
        replay_buffer_capacity=replay_buffer_capacity)
    tabular_q_agent = tabular_qlearner.QLearner(
        player_id=1, num_actions=num_actions)
    agents = [dqn_agent, tabular_q_agent]

    sess.run(tf.global_variables_initializer())

    # Train agent
    for ep in range(train_episodes):
      if ep and ep % loss_report_interval == 0:
        logging.info("[%s/%s] DQN loss: %s", ep, train_episodes, agents[0].loss)
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)

    # Evaluate against random agent
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]
    r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
    logging.info("Mean episode rewards: %s", r_mean)

    if not FLAGS.iteractive_play:
      return

    # Play from the command line against the trained DQN agent.
    human_player = 1
    while True:
      logging.info("You are playing as %s", "X" if human_player else "0")
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if player_id == human_player:
          agent_out = agents[human_player].step(time_step, is_evaluation=True)
          logging.info("\n%s", agent_out.probs.reshape((3, 3)))
          logging.info("\n%s", pretty_board(time_step))
          action = command_line_action(time_step)
        else:
          agent_out = agents[1 - human_player].step(
              time_step, is_evaluation=True)
          action = agent_out.action
        time_step = env.step([action])

      logging.info("\n%s", pretty_board(time_step))

      logging.info("End of game!")
      if time_step.rewards[human_player] > 0:
        logging.info("You win")
      elif time_step.rewards[human_player] < 0:
        logging.info("You lose")
      else:
        logging.info("Draw")
      # Switch order of players
      human_player = 1 - human_player


if __name__ == "__main__":
  app.run(main)
