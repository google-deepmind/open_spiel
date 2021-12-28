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

import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import eva
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.environments import catch

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e5), "Number of train episodes.")
flags.DEFINE_integer("eval_every", int(1e3),
                     "How often to evaluate the policy.")
flags.DEFINE_enum("algorithm", "dqn", ["dqn", "rpg", "qpg", "rm", "eva", "a2c"],
                  "Algorithms to run.")


def _eval_agent(env, agent, num_episodes):
  """Evaluates `agent` for `num_episodes`."""
  rewards = 0.0
  for _ in range(num_episodes):
    time_step = env.reset()
    episode_reward = 0
    while not time_step.last():
      agent_output = agent.step(time_step, is_evaluation=True)
      time_step = env.step([agent_output.action])
      episode_reward += time_step.rewards[0]
    rewards += episode_reward
  return rewards / num_episodes


def main_loop(unused_arg):
  """Trains a DQN agent in the catch environment."""
  env = catch.Environment()
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  train_episodes = FLAGS.num_episodes

  with tf.Session() as sess:
    if FLAGS.algorithm in {"rpg", "qpg", "rm", "a2c"}:
      agent = policy_gradient.PolicyGradient(
          sess,
          player_id=0,
          info_state_size=info_state_size,
          num_actions=num_actions,
          loss_str=FLAGS.algorithm,
          hidden_layers_sizes=[128, 128],
          batch_size=128,
          entropy_cost=0.01,
          critic_learning_rate=0.1,
          pi_learning_rate=0.1,
          num_critic_before_pi=3)
    elif FLAGS.algorithm == "dqn":
      agent = dqn.DQN(
          sess,
          player_id=0,
          state_representation_size=info_state_size,
          num_actions=num_actions,
          learning_rate=0.1,
          replay_buffer_capacity=10000,
          hidden_layers_sizes=[32, 32],
          epsilon_decay_duration=2000,  # 10% total data
          update_target_network_every=250)
    elif FLAGS.algorithm == "eva":
      agent = eva.EVAAgent(
          sess,
          env,
          player_id=0,
          state_size=info_state_size,
          num_actions=num_actions,
          learning_rate=1e-3,
          trajectory_len=2,
          num_neighbours=2,
          mixing_parameter=0.95,
          memory_capacity=10000,
          dqn_hidden_layers=[32, 32],
          epsilon_decay_duration=2000,  # 10% total data
          update_target_network_every=250)
    else:
      raise ValueError("Algorithm not implemented!")

    sess.run(tf.global_variables_initializer())

    # Train agent
    for ep in range(train_episodes):
      time_step = env.reset()
      while not time_step.last():
        agent_output = agent.step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
      # Episode is over, step agent with final info state.
      agent.step(time_step)

      if ep and ep % FLAGS.eval_every == 0:
        logging.info("-" * 80)
        logging.info("Episode %s", ep)
        logging.info("Loss: %s", agent.loss)
        avg_return = _eval_agent(env, agent, 100)
        logging.info("Avg return: %s", avg_return)


if __name__ == "__main__":
  app.run(main_loop)
