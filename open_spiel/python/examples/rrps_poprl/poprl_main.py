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

"""Populatiron RL (PopRL) algorithm for repeated rock-paper-scissors.

For details, see Lanctot et al. 2023 Population-based Evaluation in Repeated
Rock-Paper-Scissors as a Benchmark for Multiagent Reinforcement Learning
https://openreview.net/forum?id=gQnJ7ODIAx
"""

import copy
import sys
import time

from absl import app
from absl import flags
import jax
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.examples.rrps_poprl import impala
from open_spiel.python.examples.rrps_poprl import rl_environment
from open_spiel.python.jax import boltzmann_dqn
from open_spiel.python.jax import dqn
from open_spiel.python.jax import policy_gradient
import pyspiel

FLAGS = flags.FLAGS

# Training parameters
flags.DEFINE_string(
    "checkpoint_dir",
    "/tmp/dqn_test",
    "Directory to save/load the agent models.",
)
flags.DEFINE_integer(
    "save_every",
    int(1e4),
    "Episode frequency at which the DQN agent models are saved.",
)
flags.DEFINE_integer(
    "num_train_episodes", int(1e6), "Number of training episodes."
)
flags.DEFINE_integer(
    "eval_every",
    100,
    "Episode frequency at which the DQN agents are evaluated.",
)
flags.DEFINE_integer("eval_episodes", 1, "How many episodes to run per eval.")

# DQN model hyper-parameters
flags.DEFINE_list(
    "hidden_layers_sizes",
    [256, 128],
    "Number of hidden units in the Q-Network MLP.",
)
flags.DEFINE_integer(
    "replay_buffer_capacity", int(1e5), "Size of the replay buffer."
)
flags.DEFINE_integer(
    "batch_size", 16, "Number of transitions to sample at each learning step."
)
flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
flags.DEFINE_float("eta", 0.2, "BDQN eta param")

# policy_gradient params
flags.DEFINE_float("critic_learning_rate", 0.001, "Critic Learning rate.")
flags.DEFINE_float("pi_learning_rate", 0.0001, "Pi learning rate.")
flags.DEFINE_float("entropy_cost", 0.001, "Entropy cost.")
flags.DEFINE_float("lambda_", 1.0, "PG lambda.")
flags.DEFINE_integer("num_critic_before_pi", 8, "Entropy cost.")

# impala params
flags.DEFINE_integer("unroll_length", 20, "Unroll length.")
flags.DEFINE_float(
    "prediction_weight", 0, "Weight to put on the prediction losses."
)

# Main algorithm parameters
flags.DEFINE_integer("seed", 0, "Seed to use for everything")
flags.DEFINE_integer("window_size", 50, "Size of window for rolling average")
flags.DEFINE_integer("num_players", 2, "Numebr of players")
flags.DEFINE_string("game", "leduc_poker", "Game string")
flags.DEFINE_string("exploitee", "random", "Exploitee (random | first)")
flags.DEFINE_string("learner", "impala", "Learner (qlearning | dqn)")

flags.DEFINE_integer("cp_freq", 10000, "Checkpoint save frequency.")
flags.DEFINE_string("cp_dir", None, "Checkpoint directory")

# Testing against specific members
flags.DEFINE_integer("pop_only", -1, "Create a population of only this bot.")

# Generalization. How many agents to leave ou of the training population and
# use for testing?
flags.DEFINE_integer("leave_out_set_size", 0, "Cross-validation test size.")

# Environment recall
flags.DEFINE_integer("env_recall", 1, "How many timesteps back define obs?")

flags.DEFINE_string("pred_logs_dir", None, "Directory to save prediction logs.")

# Interactive mode
flags.DEFINE_string("interactive_mode", None, 'Bot id or "human".')

flags.DEFINE_float("rm_epsilon", 0.1, "Exploration for regret-matching.")

# Population RL
flags.DEFINE_float("prob_selfplay", 0.2, "Probability that we meet ourself")
flags.DEFINE_string("eval_checkpoint", None, "Evaluate a checkpoint")

# Set this to something specific for testing. List of IDs
# Set back to None to use full population.
FIXED_POPULATION = None


class State:

  def __init__(self):
    self.np_rng_state = None
    self.learning_agents = None
    self.ep = None
    self.rolling_averager = None
    self.expl_rolling_averagers = None


## This does not work, unfortunately. Not sure why. Simple pickle does not work
## because one of the haiku transforms is not serializable. There seems to be
## some nontrivial logic to use checkpointing when working with Haiku.
## Below is my attempt at applying a fix I found based on this thread:
## https://github.com/google-deepmind/dm-haiku/issues/18 but it didn't work.
class Checkpoint(object):
  """A class for saving the state of the agent (and model)."""

  def __init__(self, checkpoint_dir):
    self.checkpoint_dir = checkpoint_dir
    self.state = State()

  def restore_or_save(self):
    assert False, "Not implemented yet."
    # filename = os.path.join(self.checkpoint_dir, "tree.pkl")
    # if os.path.exists(filename):
    #   self.state = self.restore()
    # else:
    #   # pickle.dump(self.state, filename)  # Pickles to any file (even /cns).
    #   self.save()

  def restore(self):
    assert False, "Not implemented yet."
    # print("Restoring checkpoint")
    # with open(os.path.join(self.checkpoint_dir, "tree.pkl"), "rb") as f:
    #   tree_struct = pickle.load(f)
    # leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    # with open(os.path.join(self.checkpoint_dir, "arrays.npy"), "rb") as f:
    #   flat_state = [np.load(f, allow_pickle=False) for _ in leaves]
    # return jax.tree_util.tree_unflatten(treedef, flat_state)

  def save(self):
    assert False, "Not implemented yet."
    # print("Saving checkpoint")
    # # filename = os.path.join(self.checkpoint_dir, "checkpoint.pkl")
    # # pickle.dump(self.state, filename)  # Pickles to any file (even /cns).
    # with open(os.path.join(self.checkpoint_dir, "arrays.npy"), "wb") as f:
    #   for x in jax.tree_util.tree_leaves(self.state):
    #     np.save(f, x, allow_pickle=False)
    # tree_struct = jax.tree_util.tree_map(lambda t: 0, self.state)
    # with open(os.path.join(self.checkpoint_dir, "tree.pkl"), "wb") as f:
    #   pickle.dump(tree_struct, f)


class PredictionLogger(object):
  """A prediction logger."""

  def __init__(self, log_dir):
    self._log_dir = log_dir
    self._enabled = self._log_dir is not None
    self._logs = {}
    self._cur_log = ""
    self._freq = 1000
    self._last_log = 0

  def new_log(self, training_episodes):
    if not self._enabled:
      return
    if training_episodes - self._last_log >= self._freq:
      self._cur_log = ""
      self._cur_step = 0

  def log(self, training_episodes, pop_idx, predictions):
    if not self._enabled:
      return
    if training_episodes - self._last_log >= self._freq:
      line = f"{training_episodes} {pop_idx} {self._cur_step}"
      for i in range(len(predictions)):
        line = line + f" {predictions[i]}"
      self._cur_log += line + "\n"
      self._cur_step += 1

  def end_log(self, training_episodes, pop_idx):
    if not self._enabled:
      return
    if training_episodes - self._last_log >= self._freq:
      key = f"{training_episodes}.{pop_idx}"
      self._logs[key] = self._cur_log

  def update_training_episodes(self, training_episodes):
    if not self._enabled:
      return
    if training_episodes - self._last_log >= self._freq:
      self._last_log = training_episodes


def last_predictions(agent):
  if hasattr(agent, "last_predictions"):
    return agent.last_predictions()
  else:
    return np.zeros(pyspiel.ROSHAMBO_NUM_BOTS)


def eval_agent(
    env,
    num_players,
    num_actions,
    bot_names,
    learning_agent,
    prediction_logger,
    num_training_episodes,
):
  """Evaluate the agent."""
  sum_episode_rewards = np.zeros(num_players)
  pop_expl = np.zeros(pyspiel.ROSHAMBO_NUM_BOTS)
  for pop_idx in range(len(bot_names)):
    bot_id = pop_idx
    bot_name = bot_names[bot_id]
    bot = pyspiel.make_roshambo_bot(0, bot_name)
    pop_agent = BotAgent(num_actions, bot, name=bot_name)

    if hasattr(learning_agent, "restart"):
      learning_agent.restart()

    agents = [pop_agent, learning_agent]
    env.set_prediction_label(pop_idx)

    time_step = env.reset()
    episode_rewards = np.zeros(num_players)
    turn_num = 0
    prediction_logger.new_log(num_training_episodes)

    while not time_step.last():
      turn_num += 1
      player_id = time_step.observations["current_player"]
      if env.is_turn_based:
        agent_output = agents[player_id].step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
      else:
        agents_output = [
            agent.step(time_step, is_evaluation=True) for agent in agents
        ]
        action_list = [agent_output.action for agent_output in agents_output]
      prediction_logger.log(
          num_training_episodes, pop_idx, last_predictions(learning_agent)
      )
      time_step = env.step(action_list)
      episode_rewards += time_step.rewards
    pop_expl[pop_idx] = episode_rewards[0]
    sum_episode_rewards += episode_rewards
    prediction_logger.end_log(num_training_episodes, pop_idx)
  prediction_logger.update_training_episodes(num_training_episodes)
  return sum_episode_rewards / len(bot_names), pop_expl


class HumanAgent(rl_agent.AbstractAgent):
  """Agent class that wraps a bot.

  Note, the environment must include the OpenSpiel state.
  """

  def __init__(self, num_actions, name="human_agent"):
    assert num_actions > 0
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    action = 5
    while action > 2:
      value_str = input("Choose an action: ")
      if value_str == "R":
        action = 0
      if value_str == "P":
        action = 1
      if value_str == "S":
        action = 2
      if value_str == "q":
        action = -1
    probs = np.zeros(self._num_actions)
    if action >= 0:
      probs[action] = 1.0
    return rl_agent.StepOutput(action=action, probs=probs)


def pretty_top10_preds_str(predictions, indices, max_weight=1.01):
  """Pretty string representation of the top 10 predictions."""

  top_10_preds = ""
  sum_weight = 0
  for i in range(10):
    pred_idx = indices[42 - i]
    weight = predictions[pred_idx]
    bar_width = int(weight / 0.01)
    bar_str = "#" * bar_width
    top_10_preds += f"  {pred_idx:2d}: {weight:.5f} {bar_str}\n"
    sum_weight += weight
    if sum_weight > max_weight:
      break
  return top_10_preds


def interactive_episode(
    env, num_players, num_actions, bot_names, learning_agent
):
  """Interactive Episode."""
  print("Starting interactive episode!")
  actions_str = ["R", "P", "S"]
  actions_seq = ["", ""]

  if FLAGS.interactive_mode == "human":
    pop_agent = HumanAgent(num_actions)
    pop_idx = -1
  else:
    test_pop_ids = [int(FLAGS.interactive_mode)]
    pop_agent, pop_idx = sample_bot_agent(bot_names, test_pop_ids, num_actions)
    print(f"Sampled bot {pop_idx} ({bot_names[pop_idx]})")

  agents = [pop_agent, learning_agent]

  time_step = env.reset()
  episode_rewards = np.zeros(num_players)
  turn_num = 0

  while not time_step.last():
    player_id = time_step.observations["current_player"]
    if env.is_turn_based:
      agent_output = agents[player_id].step(time_step, is_evaluation=True)
      action_list = [agent_output.action]
    else:
      agents_output = [
          agent.step(time_step, is_evaluation=True) for agent in agents
      ]
      action_list = [agent_output.action for agent_output in agents_output]
    if action_list[0] == -1:
      # Restart episode.
      print("Restarting episode.")
      interactive_episode(
          env, num_players, num_actions, bot_names, learning_agent
      )
      return
    action_list_str = [actions_str[int(x)] for x in action_list]
    actions_seq[0] += action_list_str[0]
    actions_seq[1] += action_list_str[1]
    predictions = last_predictions(learning_agent)
    indices = np.argsort(predictions)
    top_10_preds = pretty_top10_preds_str(predictions, indices, max_weight=0.75)
    time_step = env.step(action_list)
    episode_rewards += time_step.rewards
    print(
        f"Turn {turn_num}, Prev actions: {action_list_str}, "
        + f"Rewards: {time_step.rewards}, Returns: {episode_rewards} \n"
        + f"Action Seq [0]: {actions_seq[0]} \n"
        + f"Action Seq [1]: {actions_seq[1]}"
    )
    print(f"Top 10 predictions: \n{top_10_preds}")
    if FLAGS.interactive_mode != "human":
      # Prompt to continue.
      input("Press any key:")
    turn_num += 1


class ConstantActionAgent(rl_agent.AbstractAgent):
  """An example agent class."""

  def __init__(
      self, player_id, num_actions, action_idx, name="constant_action_agent"
  ):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions
    self._action_idx = action_idx

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
    action = cur_legal_actions[self._action_idx]
    probs = np.zeros(self._num_actions)
    probs[action] = 1.0
    return rl_agent.StepOutput(action=action, probs=probs)


class RegretMatchingAgent(rl_agent.AbstractAgent):
  """TODO(author5): finish this agent."""

  def __init__(
      self,
      player_id,
      num_actions,
      epsilon,
      constant_observation=None,
      name="regret_matching_agent",
  ):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions
    self._regrets = {}
    self._prev_info_state = None
    self._prev_action = None
    self._prev_sample_policy = None
    self._prev_rm_policy = None
    self._epsilon = epsilon
    self._prev_legal_actions = None
    self._constant_observation = constant_observation

  def _get_info_state_key(self, info_state):
    return (
        self._constant_observation
        if self._constant_observation is not None
        else info_state
    )

  def _get_rm_policy(self, uniform_policy, info_state, legal_actions):
    info_state_key = self._get_info_state_key(info_state)
    regrets = self._regrets.get(info_state_key, None)
    if regrets is None:
      regrets = np.zeros(self._num_actions, dtype=np.float64)
      regrets[legal_actions] = 0.000001
      self._regrets[info_state_key] = regrets
    rm_policy = regrets.copy()
    rm_policy[rm_policy < 0] = 0.0
    denom = rm_policy.sum()
    if denom <= 0:
      rm_policy = uniform_policy
    else:
      rm_policy /= denom
    return rm_policy

  def _get_action_probs(self, info_state, legal_actions, epsilon):
    uniform_policy = np.zeros(self._num_actions, dtype=np.float64)
    uniform_policy[legal_actions] = 1.0 / len(legal_actions)
    rm_policy = self._get_rm_policy(uniform_policy, info_state, legal_actions)
    sample_policy = epsilon * uniform_policy + (1 - epsilon) * rm_policy
    # print(sample_policy)
    action = np.random.choice(np.arange(self._num_actions), p=sample_policy)
    return action, sample_policy, rm_policy

  def step(self, time_step, is_evaluation=False):
    legal_actions = time_step.observations["legal_actions"][self._player_id]
    info_state = str(time_step.observations["info_state"][self._player_id])
    info_state_key = self._get_info_state_key(info_state)
    sampled_action, probs = None, None

    if not time_step.last():
      epsilon = 0.0 if is_evaluation else self._epsilon
      sampled_action, sample_policy, rm_policy = self._get_action_probs(
          info_state, legal_actions, epsilon
      )

    # Learn step: don't learn during evaluation or at first agent steps.
    if self._prev_sample_policy is not None and not is_evaluation:
      reward = time_step.rewards[self._player_id]
      values = np.zeros(self._num_actions, dtype=np.float64)
      values[self._prev_action] = (
          reward / self._prev_sample_policy[self._prev_action]
      )
      exp_value = np.dot(values, self._prev_rm_policy)
      for action in legal_actions:
        self._regrets[self._prev_info_state_key][action] += (
            values[action] - exp_value
        )

      if time_step.last():  # prepare for the next episode.
        self._prev_sample_policy = None
        return

    if not is_evaluation:
      self._prev_info_state_key = info_state_key
      self._prev_action = sampled_action
      self._prev_sample_policy = sample_policy
      self._prev_rm_policy = rm_policy
      self._prev_legal_actions = legal_actions
      self._prev_info_state = info_state

    return rl_agent.StepOutput(action=sampled_action, probs=probs)


class BotAgent(rl_agent.AbstractAgent):
  """Agent class that wraps a bot.

  Note, the environment must include the OpenSpiel state.
  """

  def __init__(self, num_actions, bot, name="bot_agent"):
    assert num_actions > 0
    self._bot = bot
    self._num_actions = num_actions

  def restart(self):
    self._bot.restart()

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    _, state = pyspiel.deserialize_game_and_state(
        time_step.observations["serialized_state"]
    )

    action = self._bot.step(state)
    probs = np.zeros(self._num_actions)
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)


def create_training_agent(
    agent_type,
    num_actions,
    info_state_size,
    hidden_layers_sizes,
    max_abs_reward,
    rng_seed,
    player_id,
):
  """Create training agent."""
  if agent_type == "dqn":
    return dqn.DQN(
        player_id=player_id,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        hidden_layers_sizes=hidden_layers_sizes,
        learning_rate=FLAGS.learning_rate,
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        batch_size=FLAGS.batch_size,
    )
  elif agent_type == "bdqn":
    return boltzmann_dqn.BoltzmannDQN(
        player_id=player_id,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        hidden_layers_sizes=hidden_layers_sizes,
        learning_rate=FLAGS.learning_rate,
        replay_buffer_capacity=FLAGS.replay_buffer_capacity,
        batch_size=FLAGS.batch_size,
        eta=FLAGS.eta,
        seed=FLAGS.seed,
    )
  elif agent_type == "qlearning":
    return tabular_qlearner.QLearner(
        player_id=player_id,
        num_actions=num_actions,
        step_size=FLAGS.learning_rate,
        epsilon_schedule=rl_tools.LinearSchedule(0.5, 0.2, 1000000),
        discount_factor=0.99,
    )
  elif agent_type == "a2c":
    return policy_gradient.PolicyGradient(
        player_id,
        info_state_size,
        num_actions,
        loss_str="a2c",
        critic_learning_rate=FLAGS.critic_learning_rate,
        pi_learning_rate=FLAGS.pi_learning_rate,
        entropy_cost=FLAGS.entropy_cost,
        num_critic_before_pi=FLAGS.num_critic_before_pi,
        lambda_=FLAGS.lambda_,
        additional_discount_factor=0.99,
        hidden_layers_sizes=hidden_layers_sizes,
    )
  elif agent_type == "impala":
    return impala.IMPALA(  # pylint: disable=g-complex-comprehension
        player_id=player_id,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        num_players=2,
        unroll_len=FLAGS.unroll_length,
        net_factory=impala.BasicRNN,
        rng_key=jax.random.PRNGKey(rng_seed),
        max_abs_reward=max_abs_reward,
        learning_rate=FLAGS.pi_learning_rate,
        entropy=FLAGS.entropy_cost,
        hidden_layers_sizes=hidden_layers_sizes,
        num_predictions=pyspiel.ROSHAMBO_NUM_BOTS + 1,
        prediction_weight=FLAGS.prediction_weight,
        batch_size=FLAGS.batch_size,
    )
  elif agent_type == "rm":
    return RegretMatchingAgent(
        player_id=player_id, num_actions=num_actions, epsilon=FLAGS.rm_epsilon
    )
  elif agent_type == "rock":
    return ConstantActionAgent(player_id, num_actions, 0)
  elif agent_type == "paper":
    return ConstantActionAgent(player_id, num_actions, 1)
  elif agent_type == "scissors":
    return ConstantActionAgent(player_id, num_actions, 2)
  elif agent_type == "uniform":
    return random_agent.RandomAgent(player_id, num_actions)
  else:
    assert False


def sample_bot_agent(pid, bot_names, population_ids, num_actions):
  idx = np.random.randint(0, len(population_ids))
  bot_id = population_ids[idx]
  name = bot_names[bot_id]
  bot = pyspiel.make_roshambo_bot(pid, name)
  return BotAgent(num_actions, bot, name=name), bot_id


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


def train_test_split(roshambo_bot_ids):
  """Create a train/test split for the roshambo bots."""

  if FIXED_POPULATION is not None:
    training_ids = FIXED_POPULATION[:]
    testing_ids = FIXED_POPULATION[:]
  elif FLAGS.pop_only >= 0:
    # If the pop_only flag is set, make a population of just that member
    assert FLAGS.pop_only < len(roshambo_bot_ids)
    training_ids = [FLAGS.pop_only]
    testing_ids = [FLAGS.pop_only]
  else:
    # Otherwise, do the train/test split
    bot_ids_copy = roshambo_bot_ids.copy()
    training_ids = list(bot_ids_copy.values())
    testing_ids = []
    if FLAGS.leave_out_set_size == 0:
      testing_ids = training_ids[:]
    else:
      while len(testing_ids) < FLAGS.leave_out_set_size:
        idx = np.random.randint(0, len(training_ids))
        testing_ids.append(training_ids[idx])
        training_ids.pop(idx)
  return training_ids, testing_ids


def print_roshambo_bot_names_and_ids(roshambo_bot_names):
  for i, name in enumerate(roshambo_bot_names):
    print(f"{i}: {name}")


class AgentBot(pyspiel.Bot):
  """An agent that wraps a bot."""

  def __init__(self, agent):
    pyspiel.Bot.__init__(self)
    self._agent = agent
    self._env = rl_environment.Environment(
        "repeated_game(stage_game=matrix_rps(),num_repetitions="
        + f"{pyspiel.ROSHAMBO_NUM_THROWS},"
        + f"recall={FLAGS.env_recall})",
        include_full_state=True,
    )

  def step(self, state):
    self._env.set_state(state)
    time_step = self._env.get_time_step()
    agent_output = self._agent.step(time_step, is_evaluation=True)
    return agent_output.action


def eval_checkpoint(roshambo_bot_names, prediction_logger):
  """Evaluate a checkpoint."""

  print("Starting eval checkpoint")
  print("Loading checkpoint")
  checkpoint = Checkpoint(FLAGS.eval_checkpoint)
  checkpoint.restore_or_save()
  assert checkpoint.state.learning_agents is not None
  print("Checkpoint loaded")
  greenberg_bot = pyspiel.make_roshambo_bot(1, "greenberg")
  greenberg_agent = BotAgent(3, greenberg_bot, name="greenberg_agent")
  print("Starting eval for agent...")
  env = rl_environment.Environment(
      "repeated_game(stage_game=matrix_rps(),num_repetitions="
      + f"{pyspiel.ROSHAMBO_NUM_THROWS},"
      + f"recall={FLAGS.env_recall})",
      include_full_state=True,
  )
  sum_eval_returns = np.zeros(pyspiel.ROSHAMBO_NUM_BOTS)
  for j in range(50):
    print(f"Eval checkpoint, j={j}")
    _, pop_expl = eval_agent(
        env,
        2,
        3,
        roshambo_bot_names,
        # checkpoint.state.learning_agents[1],
        greenberg_agent,
        prediction_logger,
        0,
    )
    eval_returns = (-1) * pop_expl
    sum_eval_returns += eval_returns
    avg_eval_returns = sum_eval_returns / (j + 1)
    pop_return = avg_eval_returns.sum() / pyspiel.ROSHAMBO_NUM_BOTS
    wp_expl = avg_eval_returns.min() * (-1)
    print(f"Pop return: {pop_return}, WP expl: {wp_expl}")
    print(avg_eval_returns)


def main(_):
  np.random.seed(FLAGS.seed)

  envs = [None, None]
  envs[0] = rl_environment.Environment(
      "repeated_game(stage_game=matrix_rps(),num_repetitions="
      + f"{pyspiel.ROSHAMBO_NUM_THROWS},"
      + f"recall={FLAGS.env_recall})",
      include_full_state=True,
  )
  envs[1] = rl_environment.Environment(
      "repeated_game(stage_game=matrix_rps(),num_repetitions="
      + f"{pyspiel.ROSHAMBO_NUM_THROWS},"
      + f"recall={FLAGS.env_recall})",
      include_full_state=True,
  )
  num_players = 2
  max_abs_reward = max(
      abs(envs[0].game.min_utility()), abs(envs[0].game.max_utility())
  )

  info_state_size = envs[0].observation_spec()["info_state"][0]
  num_actions = envs[0].action_spec()["num_actions"]

  print("Loading population...")
  pop_size = pyspiel.ROSHAMBO_NUM_BOTS
  print(f"Population size: {pop_size}")
  roshambo_bot_names = pyspiel.roshambo_bot_names()
  roshambo_bot_names.sort()
  print_roshambo_bot_names_and_ids(roshambo_bot_names)

  bot_id = 0
  roshambo_bot_ids = {}
  for name in roshambo_bot_names:
    roshambo_bot_ids[name] = bot_id
    bot_id += 1

  print(f"Leave out set size: {FLAGS.leave_out_set_size}")
  train_pop_ids, test_pop_ids = train_test_split(roshambo_bot_ids)
  print(f"Training ids: {train_pop_ids}")
  print(f"Test pop ids: {test_pop_ids}")

  if FLAGS.eval_checkpoint is not None:
    prediction_logger = PredictionLogger(FLAGS.pred_logs_dir)
    eval_checkpoint(roshambo_bot_names, prediction_logger)
    return

  rolling_averager = RollingAverage(FLAGS.window_size)
  expl_rolling_averagers = []
  for _ in range(pyspiel.ROSHAMBO_NUM_BOTS):
    expl_rolling_averagers.append(RollingAverage(FLAGS.window_size))

  print("Looking for checkpoint.")
  if FLAGS.cp_dir is None:
    print("cp_dir is None, disabling checkpointing.")
    # checkpoint = phoenix.Checkpoint()
    checkpoint = None
  else:
    print(f"Looking for checkpoint in {FLAGS.cp_dir}")
    checkpoint = Checkpoint(FLAGS.cp_dir)
    checkpoint.restore_or_save()
    print(f"Checkpoint loaded. ep = {checkpoint.state.ep}")

  if FLAGS.interactive_mode is not None:
    # Must restore an agent from a checkpoint
    assert checkpoint.state.ep is not None
    assert checkpoint.state.learning_agents is not None
    interactive_episode(
        envs[0],
        num_players,
        num_actions,
        roshambo_bot_names,
        checkpoint.state.learning_agent,
    )

  ep = None
  if checkpoint is not None:
    ep = checkpoint.state.ep
    if checkpoint.state.rolling_averager is not None:
      rolling_averager = checkpoint.state.rolling_averager
    if checkpoint.state.expl_rolling_averagers is not None:
      expl_rolling_averagers = checkpoint.state.expl_rolling_averagers
    if checkpoint.state.np_rng_state is not None:
      print("Restoring numpy random state")
      np.random.set_state(checkpoint.state.np_rng_state)

  if ep is None:
    ep = 0
  prediction_logger = PredictionLogger(FLAGS.pred_logs_dir)
  if FLAGS.pred_logs_dir is not None:
    pass  # TODO(author5): Add back in (make full director)

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  # pylint: disable=g-complex-comprehension
  if checkpoint is None or checkpoint.state.learning_agents is None:
    learning_agents = [
        create_training_agent(
            FLAGS.learner,
            num_actions,
            info_state_size,
            hidden_layers_sizes,
            max_abs_reward,
            np.random.randint(100000000),
            player_id,
        )
        for player_id in [0, 1]
    ]
  else:
    learning_agents = checkpoint.state.learning_agents

  print(f"Starting at ep {ep}.")
  total_train_time = 0

  print("Starting training loop...")
  while ep < FLAGS.num_train_episodes:
    # Checkpoint save.
    if checkpoint is not None and ep > 0 and ep % FLAGS.cp_freq == 0:
      print("")
      print(f"Saving checkpoint at ep {ep}...")
      checkpoint.state.ep = ep
      checkpoint.state.np_rng_state = np.random.get_state()
      checkpoint.state.learning_agents = learning_agents
      checkpoint.state.rolling_averager = rolling_averager
      checkpoint.state.expl_rolling_averagers = expl_rolling_averagers
      checkpoint.save()
      print("Done saving checkpoint.")

    if (ep + 1) % FLAGS.eval_every == 0:
      print("")
      eps_per_sec = (ep + 1) / total_train_time
      print(f"Starting eval at ep {ep}. Avg train eps per sec: {eps_per_sec}")
      start_time_eval = time.time()
      eval_returns, pop_expl = eval_agent(
          envs[0],
          num_players,
          num_actions,
          roshambo_bot_names,
          learning_agents[1],
          prediction_logger,
          ep + 1,
      )
      value = eval_returns[1]
      rolling_averager.add(value)
      max_pop_exp = -1000
      for i in range(pyspiel.ROSHAMBO_NUM_BOTS):
        expl_rolling_averagers[i].add(pop_expl[i])
        max_pop_exp = max(max_pop_exp, expl_rolling_averagers[i].mean())
      r_mean = rolling_averager.mean()
      end_time_eval = time.time()
      print(f"Time for eval: {end_time_eval - start_time_eval}")
      data = {
          "episodes": ep + 1,
          "value": value,
          "swa_value": r_mean,
          "expl_swa_value": max_pop_exp,
          "agg_score_swa": r_mean - max_pop_exp,
          "eps_per_sec": eps_per_sec,
      }
      print(data)
      sys.stdout.flush()

    ep_start_time = time.time()
    for learner_pid in range(2):
      agents = [None, None]
      agents[learner_pid] = learning_agents[learner_pid]
      env = envs[learner_pid]
      assert env is not None
      # print(f"Learner pid: {learner_pid}")
      roll = np.random.uniform()

      if roll < FLAGS.prob_selfplay:
        agents[1 - learner_pid] = learning_agents[1 - learner_pid]
        env.set_prediction_label(pyspiel.ROSHAMBO_NUM_BOTS)
      else:
        pop_agent, pop_idx = sample_bot_agent(
            1 - learner_pid, roshambo_bot_names, train_pop_ids, num_actions
        )
        agents[1 - learner_pid] = pop_agent
        env.set_prediction_label(pop_idx)

      time_step = env.reset()
      while not time_step.last():
        time_step2 = copy.deepcopy(time_step)
        player_id = time_step.observations["current_player"]
        agents_output = [agents[0].step(time_step), agents[1].step(time_step2)]
        action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      time_step2 = copy.deepcopy(time_step)
      assert agents[0] is not None
      assert agents[1] is not None
      agents[0].step(time_step)
      agents[1].step(time_step2)

    ep_end_time = time.time()
    total_train_time += ep_end_time - ep_start_time
    ep += 1
    print(".", end="")
    sys.stdout.flush()


if __name__ == "__main__":
  app.run(main)
