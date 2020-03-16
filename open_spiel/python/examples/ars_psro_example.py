"""
Example running PSRO with ARS.
"""

import time

from absl import app
from absl import flags
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_aggregator

from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
import pyspiel


FLAGS = flags.FLAGS

# Game-related
flags.DEFINE_string("game_name", "kuhn_poker", "Game name.")
flags.DEFINE_integer("n_players", 2, "The number of players.")

# PSRO related
flags.DEFINE_string("meta_strategy_method", "alpharank",
                    "Name of meta strategy computation method.")
flags.DEFINE_integer("number_policies_selected", 1,
                     "Number of new strategies trained at each PSRO iteration.")
flags.DEFINE_integer("sims_per_entry", 1000,
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

flags.DEFINE_integer("gpsro_iterations", 100,
                     "Number of training steps for GPSRO.")
flags.DEFINE_bool("symmetric_game", False, "Whether to consider the current "
                  "game as a symmetric game.")

# Rectify options
flags.DEFINE_string("rectifier", "",
                    "Which rectifier to use. Choices are '' "
                    "(No filtering), 'rectified' for rectified.")
flags.DEFINE_string("training_strategy_selector", "probabilistic",
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

# ARS
flags.DEFINE_integer("learning_rate", 0.02, "The learning rate of ARS.")
flags.DEFINE_integer("nb_directions", 16, "The number of directions sampled per iteration.")
flags.DEFINE_integer("nb_best_directions", 16, "The number of top best directions.")
flags.DEFINE_float("noise", 0.03, "Standard deviation of the exploration noise")
flags.DEFINE_integer("ars_seed", 123, "Seed for sampling.")
flags.DEFINE_float("additional_discount_factor", 1.0, "Additional discount factor for rewards.")
flags.DEFINE_bool("v2", False, "Whether to enable V2 ARS.")

# General
flags.DEFINE_integer("seed", 1, "Seed.")
flags.DEFINE_bool("local_launch", False, "Launch locally or not.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")


def init_ars_responder(sess, env):
    """Initializes the ARS responder and agents."""
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent_class = rl_policy.ARSPolicy

    agent_kwargs = {
        "learning_rate": FLAGS.learning_rate,
        "nb_directions": FLAGS.nb_directions,
        "nb_best_directions": FLAGS.nb_best_directions,
        "noise": FLAGS.noise,
        "seed": FLAGS.ars_seed,
        "additional_discount_factor": FLAGS.additional_discount_factor,
        "v2": FLAGS.v2
    }

    oracle = rl_oracle.RLOracle(
        env,
        agent_class,
        sess,
        info_state_size,
        num_actions,
        agent_kwargs,
        number_training_episodes=FLAGS.number_training_episodes,
        self_play_proportion=FLAGS.self_play_proportion,
        sigma=FLAGS.sigma)

    agents = [
        agent_class(  # pylint: disable=g-complex-comprehension
            env,
            sess,
            player_id,
            info_state_size,
            num_actions,
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
  states_dict = get_all_states.get_all_states(game, np.infty, False, False)
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
  training_strategy_selector = FLAGS.training_strategy_selector or strategy_selectors.probabilistic_strategy_selector

  if FLAGS.meta_strategy_method == "alpharank":
    # TODO(somidshafiei): Implement epsilon-sweep for Openspiel alpharank.
    print("\n")
    print("==================================================================\n"
          "============================ Warning =============================\n"
          "==================================================================\n"
         )
    print("Selected alpharank. Warning : Current alpharank version is unstable."
          " It can raise errors because of infinite / nans elements in arrays. "
          "A fix should be uploaded in upcoming openspiel iterations.")
    print("\n")
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
      symmetric_game=FLAGS.symmetric_game)

  start_time = time.time()
  for gpsro_iteration in range(FLAGS.gpsro_iterations):
    if FLAGS.verbose:
      print("Iteration : {}".format(gpsro_iteration))
      print("Time so far: {}".format(time.time() - start_time))
    g_psro_solver.iteration()
    meta_game = g_psro_solver.get_meta_game()
    meta_probabilities = g_psro_solver.get_meta_strategies()
    policies = g_psro_solver.get_policies()

    if FLAGS.verbose:
      print("Meta game : {}".format(meta_game))
      print("Probabilities : {}".format(meta_probabilities))

    aggregator = policy_aggregator.PolicyAggregator(env.game)
    aggr_policies = aggregator.aggregate(
        range(FLAGS.n_players), policies, meta_probabilities)

    exploitabilities, expl_per_player = exploitability.nash_conv(
        env.game, aggr_policies, return_only_nash_conv=False)

    _ = print_policy_analysis(policies, env.game, FLAGS.verbose)
    if FLAGS.verbose:
      print("Exploitabilities : {}".format(exploitabilities))
      print("Exploitabilities per player : {}".format(expl_per_player))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  np.random.seed(FLAGS.seed)

  game = pyspiel.load_game_as_turn_based(FLAGS.game_name,
                                         {"players": pyspiel.GameParameter(
                                             FLAGS.n_players)})
  env = rl_environment.Environment(game)

  # Initialize oracle and agents
  sess = None
  oracle, agents = init_ars_responder(sess, env)
  gpsro_looper(env, oracle, agents)

if __name__ == "__main__":
  app.run(main)



























