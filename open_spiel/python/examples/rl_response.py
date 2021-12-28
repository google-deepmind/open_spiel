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

"""RL agents trained against fixed policy/bot as approximate responses.

This can be used to try to find exploits in policies or bots, as described in
Timbers et al. '20 (https://arxiv.org/abs/2004.09677), but only using RL
directly rather than RL+Search.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner

FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string("checkpoint_dir", "/tmp/dqn_test",
                    "Directory to save/load the agent models.")
flags.DEFINE_integer(
    "save_every", int(1e4),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(1e6),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")
flags.DEFINE_integer("eval_episodes", 1000,
                     "How many episodes to run per eval.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [64, 64, 64],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")


# Main algorithm parameters
flags.DEFINE_integer("seed", 0, "Seed to use for everything")
flags.DEFINE_integer("window_size", 30, "Size of window for rolling average")
flags.DEFINE_integer("num_players", 2, "Numebr of players")
flags.DEFINE_string("game", "leduc_poker", "Game string")
flags.DEFINE_string("exploitee", "random", "Exploitee (random | first)")
flags.DEFINE_string("learner", "qlearning", "Learner (qlearning | dqn)")


def eval_against_fixed_bots(env, trained_agents, fixed_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(fixed_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = fixed_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      turn_num = 0
      while not time_step.last():
        turn_num += 1
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes


def create_training_agents(num_players, sess, num_actions, info_state_size,
                           hidden_layers_sizes):
  """Create the agents we want to use for learning."""
  if FLAGS.learner == "qlearning":
    # pylint: disable=g-complex-comprehension
    return [
        tabular_qlearner.QLearner(
            player_id=idx,
            num_actions=num_actions,
            # step_size=0.02,
            step_size=0.1,
            # epsilon_schedule=rl_tools.ConstantSchedule(0.5),
            epsilon_schedule=rl_tools.LinearSchedule(0.5, 0.2, 1000000),
            discount_factor=0.99) for idx in range(num_players)
    ]
  elif FLAGS.learner == "dqn":
    # pylint: disable=g-complex-comprehension
    return [
        dqn.DQN(
            session=sess,
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            discount_factor=0.99,
            epsilon_start=0.5,
            epsilon_end=0.1,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=FLAGS.replay_buffer_capacity,
            batch_size=FLAGS.batch_size) for idx in range(num_players)
    ]
  else:
    raise RuntimeError("Unknown learner")


class FirstActionAgent(rl_agent.AbstractAgent):
  """An example agent class."""

  def __init__(self, player_id, num_actions, name="first_action_agent"):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    # Pick the first legal action.
    cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
    action = cur_legal_actions[0]
    probs = np.zeros(self._num_actions)
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)


class RollingAverage(object):
  """Class to store a rolling average."""

  def __init__(self, size=100):
    self._size = size
    self._values = np.array([0] * self._size, dtype=np.float64)
    self._index = 0
    self._total_additions = 0

  def add(self, value):
    self._values[self._index] = value
    self._total_additions += 1
    self._index = (self._index + 1) % self._size

  def mean(self):
    n = min(self._size, self._total_additions)
    if n == 0:
      return 0
    return self._values.sum() / n


def main(_):
  np.random.seed(FLAGS.seed)
  tf.random.set_random_seed(FLAGS.seed)

  num_players = FLAGS.num_players

  env = rl_environment.Environment(FLAGS.game, include_full_state=True)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # Exploitee agents
  if FLAGS.exploitee == "first":
    exploitee_agents = [
        FirstActionAgent(idx, num_actions) for idx in range(num_players)
    ]
  elif FLAGS.exploitee == "random":
    exploitee_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        # FirstActionAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]
  else:
    raise RuntimeError("Unknown exploitee")

  rolling_averager = RollingAverage(FLAGS.window_size)
  rolling_averager_p0 = RollingAverage(FLAGS.window_size)
  rolling_averager_p1 = RollingAverage(FLAGS.window_size)
  rolling_value = 0
  total_value = 0
  total_value_n = 0

  with tf.Session() as sess:
    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
    # pylint: disable=g-complex-comprehension
    learning_agents = create_training_agents(num_players, sess, num_actions,
                                             info_state_size,
                                             hidden_layers_sizes)
    sess.run(tf.global_variables_initializer())

    print("Starting...")

    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        r_mean = eval_against_fixed_bots(env, learning_agents, exploitee_agents,
                                         FLAGS.eval_episodes)
        value = r_mean[0] + r_mean[1]
        rolling_averager.add(value)
        rolling_averager_p0.add(r_mean[0])
        rolling_averager_p1.add(r_mean[1])
        rolling_value = rolling_averager.mean()
        rolling_value_p0 = rolling_averager_p0.mean()
        rolling_value_p1 = rolling_averager_p1.mean()
        total_value += value
        total_value_n += 1
        avg_value = total_value / total_value_n
        print(("[{}] Mean episode rewards {}, value: {}, " +
               "rval: {} (p0/p1: {} / {}), aval: {}").format(
                   ep + 1, r_mean, value, rolling_value, rolling_value_p0,
                   rolling_value_p1, avg_value))

      agents_round1 = [learning_agents[0], exploitee_agents[1]]
      agents_round2 = [exploitee_agents[0], learning_agents[1]]

      for agents in [agents_round1, agents_round2]:
        time_step = env.reset()
        while not time_step.last():
          player_id = time_step.observations["current_player"]
          if env.is_turn_based:
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
          else:
            agents_output = [agent.step(time_step) for agent in agents]
            action_list = [
                agent_output.action for agent_output in agents_output
            ]
          time_step = env.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
          agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
