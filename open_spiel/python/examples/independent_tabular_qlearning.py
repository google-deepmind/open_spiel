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

"""Tabular Q-Learner self-play example.

Two Q-Learning agents are trained by playing against each other.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer("num_eval_episodes", int(1e4),
                     "Number of episodes to use during each evaluation.")
flags.DEFINE_integer("eval_freq", int(1e4),
                     "The frequency (in episodes) to run evaluation.")
flags.DEFINE_string(
    "epsilon_schedule", None,
    "Epsilon schedule: e.g. 'linear,init,final,num_steps' or "
    "'constant,0.2'")
flags.DEFINE_string("game", "tic_tac_toe", "Game to load.")


def eval_agents(env, agents, num_episodes):
  """Evaluate the agents, returning a numpy array of average returns."""
  rewards = np.array([0] * env.num_players, dtype=np.float64)
  for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = agents[player_id].step(time_step, is_evaluation=True)
      time_step = env.step([agent_output.action])
    for i in range(env.num_players):
      rewards[i] += time_step.rewards[i]
  rewards /= num_episodes
  return rewards


def create_epsilon_schedule(sched_str):
  """Creates an epsilon schedule from the string as desribed in the flags."""
  values = FLAGS.epsilon_schedule.split(",")
  if values[0] == "linear":
    assert len(values) == 4
    return rl_tools.LinearSchedule(
        float(values[1]), float(values[2]), int(values[3]))
  elif values[0] == "constant":
    assert len(values) == 2
    return rl_tools.ConstantSchedule(float(values[1]))
  else:
    print("Unrecognized schedule string: {}".format(sched_str))
    sys.exit()


def main(_):
  env = rl_environment.Environment(FLAGS.game)
  num_players = env.num_players
  num_actions = env.action_spec()["num_actions"]

  agents = []
  if FLAGS.epsilon_schedule is not None:
    for idx in range(num_players):
      agents.append(
          tabular_qlearner.QLearner(
              player_id=idx,
              num_actions=num_actions,
              epsilon_schedule=create_epsilon_schedule(FLAGS.epsilon_schedule)))
  else:
    agents = [
        tabular_qlearner.QLearner(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

  # 1. Train the agents
  training_episodes = FLAGS.num_train_episodes
  for cur_episode in range(training_episodes):
    if cur_episode % int(FLAGS.eval_freq) == 0:
      avg_rewards = eval_agents(env, agents, FLAGS.num_eval_episodes)
      print("Training episodes: {}, Avg rewards: {}".format(
          cur_episode, avg_rewards))
    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      agent_output = agents[player_id].step(time_step)
      time_step = env.step([agent_output.action])

    # Episode is over, step all agents with final info state.
    for agent in agents:
      agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
