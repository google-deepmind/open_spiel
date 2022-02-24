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

"""NFSP agents trained on Leduc Poker."""
import abc
import os
import random
from typing import List

import tensorflow.compat.v1 as tf
from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy, rl_agent
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "leduc_poker",
                    "Name of the game.")
flags.DEFINE_integer("num_players", 2,
                     "Number of players.")
flags.DEFINE_integer("num_train_episodes", int(20e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
  128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01,
                   "Learning rate for avg-policy sl network.")
flags.DEFINE_string("optimizer_str", "sgd",
                    "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse",
                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 19200,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
                     "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.06,
                   "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.001,
                   "Final exploration parameter.")
flags.DEFINE_enum("evaluation_metric", "nash_conv", ["exploitability", "nash_conv", "avg_return"],
                  "Choose from 'exploitability', 'nash_conv', 'avg_return'.")
flags.DEFINE_integer("evaluation_opponent_pool_size", 5,
                     "Only affects the 'avg_return' evaluation metric. How many past checkpoints to use as the pool of opponents.")
flags.DEFINE_enum("evaluation_opponent_pool", "stratified", ["recent", "random", "stratified"],
                  "Only affects the 'avg_return' evaluation metric.  Determines which how to sample the pool of past opponents to use when evaluating average returns.")
flags.DEFINE_enum("evaluation_opponent_sampling", "independent", ["independent", "correlated", "perturbed"],
                  "Only affects the' avg_return' evaluation metric.  Determines how to sample rosters of opponents from the pool of possible opponents."
                  "Options are 'indpendent' to sample each player independently, 'correlated' to copy entire rosters from a previous episode, "
                  "and 'perturbed' to use the current episode's roster with a single opponent copied from a previous episode.")
flags.DEFINE_integer("evaluation_num_samples", 1000,
                     "Only affects the 'avg_return' evaluation metric.  How many episodes of play to sample for calculating the average return.")

flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir", "/tmp/nfsp_test",
                    "Directory to save/load the agent.")


class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = list(range(FLAGS.num_players))
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {
      "info_state": [None] * FLAGS.num_players,
      "legal_actions": [None] * FLAGS.num_players
    }

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
      state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
      observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def latest_checkpoint_dir():
  return os.path.join(FLAGS.checkpoint_dir, "latest")


def checkpoint_dir(episode):
  return os.path.join(FLAGS.checkpoint_dir, str(episode))


def list_saved_checkpoints():
  return list(sorted(int(p) for p in os.listdir(FLAGS.checkpoint_dir) if p != "latest"))


def most_recent_checkpoints(n):
  return list_saved_checkpoints()[-n + 1:-1]


def random_checkpoints(n):
  checkpoints = list_saved_checkpoints()[:-1]
  if n > len(checkpoints):
    n = len(checkpoints)
  return random.sample(checkpoints, n)


def stratified_checkpoints(n):
  checkpoints = list_saved_checkpoints()[:-1]
  if n > len(checkpoints):
    n = len(checkpoints)
  if n == 0:
    return []
  elif n == 1:
    return [checkpoints[0]]
  else:
    return [checkpoints[int(i * (len(checkpoints) - 1) / (n - 1))] for i in range(n)]


class OpponentDistribution(metaclass=abc.ABCMeta):
  """
  Represents a distribution of opponent agents, conditional on a single fixed agent.
  """

  @abc.abstractmethod
  def sample(self, fixed_agent: rl_agent.AbstractAgent, fixed_agent_player_id: int) -> List[rl_agent.AbstractAgent]:
    pass


class IndependentOpponentDistribution(OpponentDistribution):
  """
  A distribution where each agent is chosen independently from a uniform distribution.
  """

  def __init__(self, agents: List[List[rl_agent.AbstractAgent]]):
    """
    :param agents: List of agents for each player id. Note that player_id is the first dimension.
    """
    self.agents = agents

  def sample(self, fixed_agent: rl_agent.AbstractAgent, fixed_agent_player_id: int) -> List[rl_agent.AbstractAgent]:
    selected_agents = [random.choice(choices) for choices in self.agents]
    selected_agents[fixed_agent_player_id] = fixed_agent
    return selected_agents


class CorrelatedOpponentDistribution(OpponentDistribution):
  """
  A distribution where opponents are chosen uniformly from a list of N "rosters".
  """

  def __init__(self, rosters: List[List[rl_agent.AbstractAgent]]):
    """

    :param rosters: A list of agent rosters to randomly select from.  Note that player_id is the second dimension.
    """
    self.rosters = rosters

  def sample(self, fixed_agent: rl_agent.AbstractAgent, fixed_agent_player_id: int) -> List[rl_agent.AbstractAgent]:
    selected_agents = random.choice(self.rosters)
    selected_agents[fixed_agent_player_id] = fixed_agent
    return selected_agents


class PerturbedOpponentDistribution(OpponentDistribution):
  """
  A distribution where there is a baseline roster of agents, and only a single agent is randomized as a deviation from
  that roster.
  """

  def __init__(self, default_roster: List[rl_agent.AbstractAgent],
               other_agents: List[List[rl_agent.AbstractAgent]]) -> List[rl_agent.AbstractAgent]:
    """

    :param default_roster: The default set of agents, one per each player_id.
    :param other_agents: The list of alternative agents, per player id.  Note that player_id is the first dimension.
    """
    self.default_roster = default_roster
    self.other_agents = other_agents or [[] for _ in self.default_roster]

  def sample(self, fixed_agent: rl_agent.AbstractAgent, fixed_agent_player_id: int) -> List[rl_agent.AbstractAgent]:
    selected_agents = self.default_roster
    selected_agents[fixed_agent_player_id] = fixed_agent
    other_positions = [i for i in range(len(selected_agents)) if i != fixed_agent_player_id]
    if other_positions:
      position_to_perturb = random.choice(other_positions)
      if self.other_agents[position_to_perturb]:
        selected_agents[position_to_perturb] = random.choice(self.other_agents[position_to_perturb])
    return selected_agents


def monte_carlo_returns(env: rl_environment.Environment, agents: List[rl_agent.AbstractAgent]) -> List[float]:
  returns = [0.0 for _ in agents]
  discounts = [1.0 for _ in agents]
  time_step = env.reset()
  while True:
    if time_step.rewards:
      returns = [R + r * d for (R, r, d) in zip(returns, time_step.rewards, discounts)]
      discounts = time_step.discounts if time_step.discounts else [1.0 for _ in time_step.rewards]
    if time_step.last():
      break
    player_id = time_step.observations["current_player"]
    agent_output = agents[player_id].step(time_step, True)
    action_list = [agent_output.action]
    time_step = env.step(action_list)

  for agent in agents:
    agent.step(time_step)
  return returns


def average_returns(env: rl_environment.Environment, agents_to_evaluate: List[rl_agent.AbstractAgent],
                    opponent_distribution: OpponentDistribution, n_samples: int) -> List[float]:
  """
  :param env: Game environment
  :param agents_to_evaluate: List of N agents to evaluate, one for each player in the game.
  :param opponent_distribution: The distribution of opponents to evaluate the agents against.
  :param n_samples: Number of games to play
  :return: Length N array of average agent returns
  """
  samples = [0.0 for _ in agents_to_evaluate]
  for i in range(n_samples):
    for player_id, agent in enumerate(agents_to_evaluate):
      agents = opponent_distribution.sample(agent, player_id)
      returns = monte_carlo_returns(env, agents)
      samples[player_id] += returns[player_id]
  return [s / n_samples for s in samples]


def evaluate_monte_carlo(env: rl_environment.Environment, latest_agents: List[rl_agent.AbstractAgent], nfsp_args,
                         opponent_episodes, n_samples, sampling_mode="independent") -> List[float]:
  """

  :param env: Game environment
  :param latest_agents:
  :param nfsp_args: Args for constructing the NFSP agent.  Must match those used to save the previous agents.
  :param opponent_episodes: List of episode numbers to load agents from for use as opponents
  :param n_samples: Number of games to sample.
  :param sampling_mode: How to construct rosters of agents for each game options are "independent", "correlated", and "perturbed"
  :return: List of average returns, per agent in latest_agents.
  """
  opponent_agents = []
  for episode in opponent_episodes:
    episode_agents = []
    for player_id in range(len(latest_agents)):
      agent = nfsp.NFSP(player_id=player_id, **nfsp_args)
      agent.restore(checkpoint_dir(episode))
      episode_agents.append(agent)
    opponent_agents.append(episode_agents)
  if sampling_mode == "independent":
    distribution = IndependentOpponentDistribution(list(zip(*opponent_agents, latest_agents)))
  elif sampling_mode == "correlated":
    distribution = CorrelatedOpponentDistribution(opponent_agents + [latest_agents])
  elif sampling_mode == "perturbed":
    distribution = PerturbedOpponentDistribution(latest_agents, list(zip(*opponent_agents)))
  else:
    raise ValueError("Invalid sampling_mode argument to evaluate_monte_carlo: " + sampling_mode)
  return average_returns(env, latest_agents, distribution, n_samples)


def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  game = FLAGS.game_name

  env_configs = {}
  if FLAGS.num_players:
    env_configs["players"] = FLAGS.num_players
  env = rl_environment.Environment(game, **env_configs)
  num_players = env.num_players
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
    "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
    "reservoir_buffer_capacity": FLAGS.reservoir_buffer_capacity,
    "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
    "anticipatory_param": FLAGS.anticipatory_param,
    "batch_size": FLAGS.batch_size,
    "learn_every": FLAGS.learn_every,
    "rl_learning_rate": FLAGS.rl_learning_rate,
    "sl_learning_rate": FLAGS.sl_learning_rate,
    "optimizer_str": FLAGS.optimizer_str,
    "loss_str": FLAGS.loss_str,
    "update_target_network_every": FLAGS.update_target_network_every,
    "discount_factor": FLAGS.discount_factor,
    "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
    "epsilon_start": FLAGS.epsilon_start,
    "epsilon_end": FLAGS.epsilon_end,
  }

  with tf.Session() as sess:
    nfsp_args = {
      "session": sess,
      "state_representation_size": info_state_size,
      "num_actions": num_actions,
      "hidden_layers_sizes": hidden_layers_sizes,
      **kwargs
    }
    # pylint: disable=g-complex-comprehension
    agents = [
      nfsp.NFSP(player_id=idx, **nfsp_args) for idx in range(num_players)
    ]
    joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    sess.run(tf.global_variables_initializer())

    start_episode = 0
    if FLAGS.use_checkpoints:
      os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)
      for agent in agents:
        if agent.has_checkpoint(latest_checkpoint_dir()):
          agent.restore(latest_checkpoint_dir())
          start_episode = list_saved_checkpoints()[-1]

    for ep in range(start_episode, start_episode + FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        logging.info("Losses: %s", losses)

        if FLAGS.use_checkpoints:
          for agent in agents:
            agent.save(latest_checkpoint_dir())
            agent.save(checkpoint_dir(ep))

        if FLAGS.evaluation_metric == "exploitability":
          # Avg exploitability is implemented only for 2 players constant-sum
          # games, use nash_conv otherwise.
          expl = exploitability.exploitability(env.game, joint_avg_policy)
          logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
        elif FLAGS.evaluation_metric == "nash_conv":
          nash_conv = exploitability.nash_conv(env.game, joint_avg_policy)
          logging.info("[%s] NashConv %s", ep + 1, nash_conv)
        elif FLAGS.evaluation_metric == "avg_return":
          if FLAGS.evaluation_opponent_pool == "recent":
            opponent_checkpoints = most_recent_checkpoints(FLAGS.evaluation_opponent_pool_size)
          elif FLAGS.evaluation_opponent_pool == "random":
            opponent_checkpoints = random_checkpoints(FLAGS.evaluation_opponent_pool_size)
          elif FLAGS.evaluation_opponent_pool == "stratified":
            opponent_checkpoints = stratified_checkpoints(FLAGS.evaluation_opponent_pool_size)
          avg_return = evaluate_monte_carlo(env, agents, nfsp_args, opponent_checkpoints,
                                            FLAGS.evaluation_num_samples, FLAGS.evaluation_opponent_sampling)
          logging.info("[%s] AvgReturn %s", ep + 1, avg_return)
        else:
          raise ValueError(" ".join(("Invalid evaluation metric, choose from",
                                     "'exploitability', 'nash_conv', 'avg_return.")))

        logging.info("_____________________________________________")

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
