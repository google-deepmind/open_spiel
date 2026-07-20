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

"""Example running PSRO on OpenSpiel Sequential games.

To reproduce results from (Muller et al., "A Generalized Training Approach for
Multiagent Learning", ICLR 2020; https://arxiv.org/abs/1909.12823), run this
script with:
  - `game_name` in ['kuhn_poker', 'leduc_poker']
  - `n_players` in [2, 3, 4, 5]
  - `meta_strategy_method` in ['alpharank', 'uniform', 'nash', 'prd', 'ssd']
  - `rectifier` in ['', 'rectified']

The other parameters keeping their default values.
"""

import threading
import time

from absl import app
from absl import flags
import numpy as np

# pylint: disable=g-bad-import-order
import pyspiel

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
import sys
sys.setrecursionlimit(3000)


def get_memory_usage_mb():
  try:
    import psutil

    rss = psutil.Process().memory_info().rss
    return rss / (1024.0 * 1024.0)
  except Exception:
    try:
      import resource

      rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      return rss_kb / 1024.0
    except Exception:
      return -1.0


class MemorySampler:
  """Background sampler that tracks peak memory usage."""

  def __init__(self, interval_s=0.05):
    self._interval_s = max(interval_s, 0.0)
    self._running = False
    self._thread = None
    self.max_mb = -1.0

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc, exc_tb):
    self.stop()

  def start(self):
    if self._running:
      return
    self._running = True
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def stop(self):
    if not self._running:
      return
    self._running = False
    if self._thread is not None:
      self._thread.join()
    # Capture one final sample after stopping to include post-loop usage.
    self._record_sample()

  def _run(self):
    while self._running:
      self._record_sample()
      if self._interval_s == 0.0:
        # Avoid busy spinning if configured interval is zero.
        time.sleep(0)
      else:
        time.sleep(self._interval_s)

  def _record_sample(self):
    sample = get_memory_usage_mb()
    if sample >= 0:
      if self.max_mb < 0:
        self.max_mb = sample
      else:
        self.max_mb = max(self.max_mb, sample)
    return sample


FLAGS = flags.FLAGS

# Game-related
flags.DEFINE_string("game_name", "leduc_poker", "Game name.")
flags.DEFINE_integer("n_players", 3, "The number of players.")
flags.DEFINE_string(
  "log_path", "",
  "Optional file path for logging exploitabilities. Leave empty to disable file logging.")

flags.DEFINE_bool("use_sparse", False, "whether to use scipy sparse matrices")


# PSRO related
flags.DEFINE_string(
  "meta_strategy_method", "ssd",
  "Name of meta strategy computation method")
flags.DEFINE_integer("number_policies_selected", 1,
                     "Number of new strategies trained at each PSRO iteration.")
flags.DEFINE_integer("sims_per_entry", 100,
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

flags.DEFINE_integer("gpsro_iterations", 70,
                     "Number of training steps for GPSRO.")
flags.DEFINE_bool("symmetric_game", False, "Whether to consider the current "
                  "game as a symmetric game.")

# Rectify options
flags.DEFINE_string("rectifier", "",
                    "Which rectifier to use. Choices are '' "
                    "(No filtering), 'rectified' for rectified.")
flags.DEFINE_string("training_strategy_selector", "top_k_probabilities",
                    "Which strategy selector to use. Choices are "
                    " - 'top_k_probabilities': select top "
                    "`number_policies_selected` strategies. "
                    " - 'probabilistic': Randomly samples "
                    "`number_policies_selected` strategies with probability "
                    "equal to their selection probabilities. "
                    " - 'uniform': Uniformly sample `number_policies_selected` "
                    "strategies. "
                    " - 'rectified': Select every non-zero-selection-"
                    "probability strategy available to each player.")

# General (RL) agent parameters
flags.DEFINE_string("oracle_type", "BR", "Choices are DQN, PG (Policy "
                    "Gradient) or BR (exact Best Response)")
flags.DEFINE_integer("number_training_episodes", int(1e4), "Number training "
                     "episodes per RL policy. Used for PG and DQN")
flags.DEFINE_float("self_play_proportion", 0.0, "Self play proportion")
flags.DEFINE_integer("hidden_layer_size", 256, "Hidden layer size")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_float("sigma", 0.0, "Policy copy noise (Gaussian Dropout term).")
flags.DEFINE_string("optimizer_str", "adam", "'adam' or 'sgd'")

# Policy Gradient Oracle related
flags.DEFINE_string("loss_str", "qpg", "Name of loss used for BR training.")
flags.DEFINE_integer("num_q_before_pi", 8, "# critic updates before Pi update")
flags.DEFINE_integer("n_hidden_layers", 4, "# of hidden layers")
flags.DEFINE_float("entropy_cost", 0.001, "Self play proportion")
flags.DEFINE_float("critic_learning_rate", 1e-2, "Critic learning rate")
flags.DEFINE_float("pi_learning_rate", 1e-3, "Policy learning rate.")

# DQN
flags.DEFINE_float("dqn_learning_rate", 1e-2, "DQN learning rate.")
flags.DEFINE_integer("update_target_network_every", 1000, "Update target "
                     "network every [X] steps")
flags.DEFINE_integer("learn_every", 10, "Learn every [X] steps.")

# General
flags.DEFINE_integer("seed", 1, "Seed.")
flags.DEFINE_bool("local_launch", False, "Launch locally or not.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")
flags.DEFINE_float(
  "memory_sample_interval", 0.05,
  "Seconds between asynchronous memory samples collected during each"
  " gpsro iteration. Set to 0 for best-effort continuous sampling.")


def init_pg_responder(env):
  """Initializes the Policy Gradient-based responder and agents."""
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  agent_class = rl_policy.PGPolicy

  agent_kwargs = {
      "info_state_size": info_state_size,
      "num_actions": num_actions,
      "loss_str": FLAGS.loss_str,
      "loss_class": False,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "entropy_cost": FLAGS.entropy_cost,
      "critic_learning_rate": FLAGS.critic_learning_rate,
      "pi_learning_rate": FLAGS.pi_learning_rate,
      "num_critic_before_pi": FLAGS.num_q_before_pi,
      "optimizer_str": FLAGS.optimizer_str
  }
  oracle = rl_oracle.RLOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=FLAGS.number_training_episodes,
      self_play_proportion=FLAGS.self_play_proportion,
      sigma=FLAGS.sigma)

  agents = [
      agent_class(  # pylint: disable=g-complex-comprehension
          env,
          player_id,
          **agent_kwargs)
      for player_id in range(FLAGS.n_players)
  ]
  for agent in agents:
    agent.freeze()
  return oracle, agents


def init_br_responder(env):
  """Initializes the tabular best-response based responder and agents."""
  random_policy = policy.TabularPolicy(env.game)
  oracle = best_response_oracle.BestResponseOracle(
      game=env.game, policy=random_policy)
  agents = [random_policy.__copy__() for _ in range(FLAGS.n_players)]
  return oracle, agents


def init_dqn_responder(env):
  """Initializes the Policy Gradient-based responder and agents."""
  state_representation_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  agent_class = rl_policy.DQNPolicy
  agent_kwargs = {
      "state_representation_size": state_representation_size,
      "num_actions": num_actions,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "batch_size": FLAGS.batch_size,
      "learning_rate": FLAGS.dqn_learning_rate,
      "update_target_network_every": FLAGS.update_target_network_every,
      "learn_every": FLAGS.learn_every,
      "optimizer_str": FLAGS.optimizer_str
  }
  oracle = rl_oracle.RLOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=FLAGS.number_training_episodes,
      self_play_proportion=FLAGS.self_play_proportion,
      sigma=FLAGS.sigma)

  agents = [
      agent_class(  # pylint: disable=g-complex-comprehension
          env,
          player_id,
          **agent_kwargs)
      for player_id in range(FLAGS.n_players)
  ]
  for agent in agents:
    agent.freeze()
  return oracle, agents


def print_policy_analysis(policies, game, verbose=False):
  """Function printing policy diversity within game's known policies.

  Warning : only works with deterministic policies.
  Args:
    policies: List of list of policies (One list per game player)
    game: OpenSpiel game object.
    verbose: Whether to print policy diversity information. (True : print)

  Returns:
    List of list of unique policies (One list per player)
  """
  states_dict = get_all_states.get_all_states(game, np.inf, False, False)
  unique_policies = []
  for player in range(len(policies)):
    cur_policies = policies[player]
    cur_set = set()
    for pol in cur_policies:
      cur_str = ""
      for state_str in states_dict:
        if states_dict[state_str].current_player() == player:
          pol_action_dict = pol(states_dict[state_str])
          max_prob = max(list(pol_action_dict.values()))
          max_prob_actions = [
              a for a in pol_action_dict if pol_action_dict[a] == max_prob
          ]
          cur_str += "__" + state_str
          for a in max_prob_actions:
            cur_str += "-" + str(a)
      cur_set.add(cur_str)
    unique_policies.append(cur_set)
  if verbose:
    print("\n=====================================\nPolicy Diversity :")
    for player, cur_set in enumerate(unique_policies):
      print("Player {} : {} unique policies.".format(player, len(cur_set)))
  print("")
  return unique_policies


def gpsro_looper(env, oracle, agents):
  """Initializes and executes the GPSRO training loop."""
  sample_from_marginals = True  # TODO(somidshafiei) set False for alpharank
  training_strategy_selector = (FLAGS.training_strategy_selector or
                                strategy_selectors.probabilistic)

  g_psro_solver = psro_v2.PSROSolver(
      env.game,
      oracle,
      initial_policies=agents,
      training_strategy_selector=training_strategy_selector,
      rectifier=FLAGS.rectifier,
      sims_per_entry=FLAGS.sims_per_entry,
      number_policies_selected=FLAGS.number_policies_selected,
      meta_strategy_method=FLAGS.meta_strategy_method,
      prd_iterations=50000,
      prd_gamma=1e-10,
      sample_from_marginals=sample_from_marginals,
      symmetric_game=FLAGS.symmetric_game,
      use_sparse=FLAGS.use_sparse)

  start_time = time.time()
  max_memory_mb = -1.0
  for gpsro_iteration in range(FLAGS.gpsro_iterations):
    iter_start = time.time()
    sampler = MemorySampler(interval_s=FLAGS.memory_sample_interval)
    sampler.start()
    try:
      g_psro_solver.iteration()
    finally:
      sampler.stop()
    iter_end = time.time()
    iter_time = iter_end - iter_start
    memory_mb = get_memory_usage_mb()
    iteration_peak_mb = sampler.max_mb if sampler.max_mb >= 0 else memory_mb
    if iteration_peak_mb >= 0:
      if max_memory_mb < 0:
        max_memory_mb = iteration_peak_mb
      else:
        max_memory_mb = max(max_memory_mb, iteration_peak_mb)

    meta_game = g_psro_solver.get_meta_game()
    meta_probabilities = g_psro_solver.get_meta_strategies()
    policies = g_psro_solver.get_policies()

    # The following lines only work for sequential games for the moment.
    if env.game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL:
      aggregator = policy_aggregator.PolicyAggregator(env.game)
      aggr_policies = aggregator.aggregate(
          range(FLAGS.n_players), policies, meta_probabilities)

      exploitabilities, expl_per_player = exploitability.nash_conv(
          env.game, aggr_policies, return_only_nash_conv=False)

      _ = print_policy_analysis(policies, env.game, FLAGS.verbose)
      if FLAGS.verbose:
        print("Iteration : {}".format(gpsro_iteration))
        print("Iteration time (s): {:.6f}".format(iter_time))
        if memory_mb >= 0:
          print("mb used: {:.2f}".format(memory_mb))
        else:
          print("mb used: n/a")
        if iteration_peak_mb >= 0:
          print("peak iteration mb: {:.2f}".format(iteration_peak_mb))
        else:
          print("peak iteration mb: n/a")
        if max_memory_mb >= 0:
          print("max mb so far: {:.2f}".format(max_memory_mb))
        else:
          print("max mb so far: n/a")
        print("Exploitabilities : {}".format(exploitabilities))
        if FLAGS.log_path:
          try:
            with open(FLAGS.log_path, "a") as f:
              f.write(
                  str(exploitabilities)
                  + "," + str(iter_time)
                  + "," + str(memory_mb)
                  + "," + str(iteration_peak_mb)
                  + "," + str(max_memory_mb)
                  + "\n")
          except Exception as e:
            if FLAGS.verbose:
              print("Failed to write exploitabilities to {}: {}".format(FLAGS.log_path, e))
        print("Exploitabilities per player : {}".format(expl_per_player))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  np.random.seed(FLAGS.seed)

  game = pyspiel.load_game_as_turn_based(FLAGS.game_name,
                                         {"players": FLAGS.n_players})
  env = rl_environment.Environment(game)

  # Initialize oracle and agents
  if FLAGS.oracle_type == "DQN":
    oracle, agents = init_dqn_responder(env)
  elif FLAGS.oracle_type == "PG":
    oracle, agents = init_pg_responder(env)
  elif FLAGS.oracle_type == "BR":
    oracle, agents = init_br_responder(env)
  gpsro_looper(env, oracle, agents)


if __name__ == "__main__":
  app.run(main)
