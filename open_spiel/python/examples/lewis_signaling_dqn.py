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

"""DQN example on Lewis Signaling Game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn

FLAGS = flags.FLAGS

# Env parameters
flags.DEFINE_integer("num_states", 3, "Number of states and actions")
flags.DEFINE_integer("num_messages", 3, "Number of messages")
flags.DEFINE_string("payoffs", "1, 0, 0, 0, 1, 0, 0, 0, 1",
                    "Payoffs to use ('random' for random [0, 1) payoffs)")

# Alg parameters
flags.DEFINE_integer("num_episodes", 50000, "Number of train episodes")
flags.DEFINE_float("step_size", 0.1, "Step size for updates")
flags.DEFINE_float("eps_init", 1.0, "Initial value of epsilon")
flags.DEFINE_float("eps_final", 0.0, "Final value of epsilon")
flags.DEFINE_integer("eps_decay_steps", 49000,
                     "Number of episodes to decay epsilon")
flags.DEFINE_integer("replay_buffer_capacity", int(1e4),
                     "Size of replay buffer")

# Misc paramters
flags.DEFINE_integer("num_runs", 10, "Number of repetitions")
flags.DEFINE_integer("log_interval", 100,
                     "Number of episodes between each logging")
flags.DEFINE_bool("plot", False, "Set to plot the graphs")


def main(_):
  game = "lewis_signaling"
  num_players = 2

  num_states = FLAGS.num_states
  num_messages = FLAGS.num_messages
  if FLAGS.payoffs == "random":
    payoffs = np.random.random((num_states, num_states))
    payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
  elif FLAGS.payoffs == "climbing":
    # This is a particular payoff matrix that is hard for decentralized
    # algorithms. Introduced in C. Claus and C. Boutilier, "The dynamics of
    # reinforcement learning in cooperative multiagent systems", 1998, for
    # simultaneous action games, but it is difficult even in the case of
    # signaling games.
    payoffs = np.array([[11, -30, 0], [-30, 7, 6], [0, 0, 5]]) / 30
    payoffs_str = ",".join([str(x) for x in payoffs.flatten()])
  else:
    payoffs_str = FLAGS.payoffs
    try:
      payoffs_list = [float(x) for x in payoffs_str.split(",")]
      payoffs = np.array(payoffs_list).reshape((num_states, num_states))
    except ValueError:
      raise ValueError(
          "There should be {} (states * actions) elements in payoff. Found {} elements"
          .format(num_states * num_states, len(payoffs_list)))

  env_configs = {
      "num_states": num_states,
      "num_messages": num_messages,
      "payoffs": payoffs_str
  }

  env = rl_environment.Environment(game, **env_configs)
  state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  replay_buffer_capacity = FLAGS.replay_buffer_capacity

  # Results to store
  num_runs = FLAGS.num_runs
  training_episodes = FLAGS.num_episodes
  log_interval = FLAGS.log_interval
  rewards = np.zeros((num_runs, training_episodes // log_interval))
  opts = np.zeros((num_runs, training_episodes // log_interval))
  converge_point = np.zeros((num_states, num_states))
  percent_opt = 0

  # Repeat the experiment num_runs times
  for i in range(num_runs):
    with tf.Session() as sess:
      # pylint: disable=g-complex-comprehension
      agents = [
          dqn.DQN(
              sess,
              player_id=idx,
              state_representation_size=state_size,
              num_actions=num_actions,
              learning_rate=FLAGS.step_size,
              replay_buffer_capacity=replay_buffer_capacity,
              epsilon_start=FLAGS.eps_init,
              epsilon_end=FLAGS.eps_final,
              epsilon_decay_duration=FLAGS.eps_decay_steps * 2)
          for idx in range(num_players)
      ]

      # 1. Train the agents
      for cur_episode in range(training_episodes):
        time_step = env.reset()
        # Find cur_state for logging. See lewis_signaling.cc for info_state
        # details
        cur_state = time_step.observations["info_state"][0][3:].index(1)
        while not time_step.last():
          player_id = time_step.observations["current_player"]
          agent_output = agents[player_id].step(time_step)
          time_step = env.step([agent_output.action])

        # Episode is over, step all agents with final info state.
        for agent in agents:
          agent.step(time_step)

        # Store rewards
        reward = time_step.rewards[0]
        max_reward = payoffs[cur_state].max()
        cur_idx = (i, cur_episode // log_interval)
        rewards[cur_idx] += reward / log_interval
        opts[cur_idx] += np.isclose(reward, max_reward) / log_interval

      base_info_state0 = [1.0, 0.0, 0.0] + [0.0] * num_states
      base_info_state1 = [0.0, 1.0, 0.0] + [0.0] * num_states

      for s in range(num_states):
        info_state0 = copy.deepcopy(base_info_state0)
        info_state0[3 + s] = 1.0
        # pylint: disable=protected-access
        m, _ = agents[0]._epsilon_greedy(info_state0, np.arange(num_messages),
                                         0)
        info_state1 = copy.deepcopy(base_info_state1)
        info_state1[3 + m] = 1.0
        a, _ = agents[1]._epsilon_greedy(info_state1, np.arange(num_states), 0)
        converge_point[s, a] += 1
        best_act = payoffs[s].argmax()
        percent_opt += int(a == best_act) / num_runs / num_states

  if FLAGS.plot:
    # pylint: disable=g-import-not-at-top
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    params = {
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    }
    mpl.rcParams.update(params)

    def init_fig():
      fig, ax = plt.subplots(1, 1)
      ax.spines["top"].set_visible(False)
      ax.spines["right"].set_visible(False)
      return fig, ax

    def plot_scalars(scalars,
                     repetition_axis=0,
                     scalar_labels=None,
                     title=None,
                     ax_labels=None):
      """Plots scalar on ax by filling 1 standard error.

      Args:
          scalars: List of scalars to plot (mean taken over repetition
            axis)
          repetition_axis: Axis to take the mean over
          scalar_labels: Labels for the scalars (for legend)
          title: Figure title
          ax_labels: Labels for x and y axis (list of 2 strings)
      """
      if not all([len(s.shape) == 2 for s in scalars]):
        raise ValueError("Only 2D arrays supported for plotting")

      if scalar_labels is None:
        scalar_labels = [None] * len(scalars)

      if len(scalars) != len(scalar_labels):
        raise ValueError(
            "Wrong number of scalar labels, expected {} but received {}".format(
                len(scalars), len(scalar_labels)))

      _, plot_axis = init_fig()
      for i, scalar in enumerate(scalars):
        xs = np.arange(scalar.shape[1 - repetition_axis]) * FLAGS.log_interval
        mean = scalar.mean(axis=repetition_axis)
        sem = stats.sem(scalar, axis=repetition_axis)
        plot_axis.plot(xs, mean, label=scalar_labels[i])
        plot_axis.fill_between(xs, mean - sem, mean + sem, alpha=0.5)

      if title is not None:
        plot_axis.set_title(title)
      if ax_labels is not None:
        plot_axis.set_xlabel(ax_labels[0])
        plot_axis.set_ylabel(ax_labels[1])

    def plot_confusion_matrix(cm, cmap=plt.cm.Blues, title=None):
      """Plot the confusion matrix.

      Args:
          cm (np.ndarray): Confusion matrix to plot
          cmap: Color map to be used in matplotlib's imshow
          title: Figure title

      Returns:
          Figure and axis on which the confusion matrix is plotted.
      """
      fig, ax = plt.subplots()
      ax.imshow(cm, interpolation="nearest", cmap=cmap)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_xlabel("Receiver's action", fontsize=14)
      ax.set_ylabel("Sender's state", fontsize=14)
      # Loop over data dimensions and create text annotations.
      fmt = "d"
      thresh = cm.max() / 2.
      for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
          ax.text(
              j,
              i,
              format(cm[i, j], fmt),
              ha="center",
              va="center",
              color="white" if cm[i, j] > thresh else "black")
      fig.tight_layout()
      if title is not None:
        ax.set_title(title)
      return fig, ax

    plot_scalars([rewards],
                 title="Reward graph (DQN)",
                 ax_labels=["Episodes", "Reward per episode"])
    plot_scalars([opts],
                 title="Percentage of optimal actions (DQN)",
                 ax_labels=["Episodes", "% optimal actions"])

    plot_confusion_matrix(
        converge_point.astype(np.int), title="Final policy (DQN)")

    plt.show()

  return percent_opt


if __name__ == "__main__":
  app.run(main)
