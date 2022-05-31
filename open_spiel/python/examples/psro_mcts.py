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


"""Example use of MCTS as a best response oracle in PSRO.

PSRO-MCTS demonstrates that online planning algorithms are compatiable with the population-based PSRO paradigm.
It potentially fits for large-scale perfect-information games.
On tic_tac_toe nashconv decreases from 2.0 to ~0.5 in 50 PSRO iterations.
"""


import time

from absl import app
from absl import flags

import pyspiel

from open_spiel.python.algorithms import best_response as pyspiel_best_response
from open_spiel.python.algorithms.mcts import MCTSBot
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator
from open_spiel.python.algorithms.mcts import SearchNode
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.psro_v2 import optimization_oracle
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.policy import Policy


import copy
import numpy as np


FLAGS = flags.FLAGS

# MCTS related
flags.DEFINE_float("uct_c", 2, "uct_c for MCTS")
flags.DEFINE_integer("rollout_count", 1, "rollout count for MCTS")
flags.DEFINE_integer("max_simulations", 100,
                     "max iterations for MCTS simulations phase")
flags.DEFINE_integer("seed", 17, "random seed")

# PSRO related
flags.DEFINE_integer("psro_sims_per_entry", 1,
                     "simulation numbers for meta-game entry")
flags.DEFINE_integer("psro_iterations", 100, "number of PSRO iterations")


class AgainstMixtureMCTSBot(MCTSBot):
  """An MCTS bot that conducts planning against a fixed mixture of opponents.

  Adapted from open_spiel.python.algorithms.mcts.MCTSBot
  Args:
    player_id: the player id that this bot corresponds to in a game
    player_policies: if is_joint, then a list of lists each containing num_players policies, 
    representing joint-policy profiles. Otherwise a list of lists each containing individual policies for each num_players
    weights: lists of arrays. it corresponds to the mixture of player_policies
    is_joint: a boolean indicating whether player_policies are represented as joint profile or not
    epsilon: a trembling-hand parameters dealing with zero-reaching cases
  """

  def __init__(self,
               player_id,
               player_policies,
               weights,
               is_joint=False,
               epsilon=1e-40,
               **mcts_base_params):

    MCTSBot.__init__(self, **mcts_base_params)

    self._player_id = player_id
    self._player_policies = player_policies
    self._weights = weights
    self._is_joint = is_joint
    self._epsilon = epsilon

  def compute_current_weights(self, state):
    """computes correct posterior opponent-sampling weights"""
    working_state = self._game.new_initial_state()
    history = state.full_history()
    self._cur_weights = copy.deepcopy(self._weights)
    for h in history:
      if h.player != self._player_id:
        if self._is_joint:
          for i in range(len(self._player_policies)):
            action_prob = self._player_policies[i][h.player].action_probabilities(
                working_state)
            if h.action not in action_prob:
              self._cur_weights[i] = 0
            else:
              self._cur_weights[i] *= action_prob[h.action]
        else:
          for i in range(len(self._player_policies[h.player])):
            action_prob = self._player_policies[h.player][i].action_probabilities(
                working_state)
            if h.action not in action_prob:
              self._cur_weights[h.player][i] = 0
            else:
              self._cur_weights[h.player][i] *= action_prob[h.action]
      working_state.apply_action(h.action)

    assert working_state.information_state_string() == state.information_state_string()

  def step_with_policy(self, state):
    """Returns bot's policy and action at given state."""
    self.compute_current_weights(state)
    return MCTSBot.step_with_policy(self, state)

  def _apply_tree_policy(self, root, state):
    visit_path = [root]
    working_state = state.clone()
    current_node = root
    while (not working_state.is_terminal() and
           current_node.explore_count > 0) or (
               working_state.is_chance_node() and self.dont_return_chance_node):
      if not current_node.children:
        # For a new node, initialize its state, then choose a child as normal.
        legal_actions = self.evaluator.prior(working_state)
        if current_node is root and self._dirichlet_noise:
          epsilon, alpha = self._dirichlet_noise
          noise = self._random_state.dirichlet([alpha] * len(legal_actions))
          legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                           for (a, p), n in zip(legal_actions, noise)]
        # Reduce bias from move generation order.
        self._random_state.shuffle(legal_actions)
        player = working_state.current_player()
        current_node.children = [
            SearchNode(action, player, prior) for action, prior in legal_actions
        ]

      # select actions of chance nodes or opponent policies according to the fixed probabilities
      if working_state.current_player() != self._player_id:
        if working_state.is_chance_node():
          outcomes = working_state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
        else:
          cur_player = working_state.current_player()
          action_list, prob_list = zip(
              *(self._cur_policies[cur_player].action_probabilities(working_state)).items())

        action = self._random_state.choice(action_list, p=prob_list)
        chosen_child = next(
            c for c in current_node.children if c.action == action)
      else:
        # Otherwise choose node with largest UCT value
        chosen_child = max(
            current_node.children,
            key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                c, current_node.explore_count, self.uct_c))

      working_state.apply_action(chosen_child.action)
      current_node = chosen_child
      visit_path.append(current_node)

    return visit_path, working_state

  def mcts_search(self, state):
    root_player = state.current_player()
    root = SearchNode(None, state.current_player(), 1)
    for _ in range(self.max_simulations):
      # sample an opponent profile according to the current mixture at current state
      if self._is_joint:
        norm_prob = np.array(self._cur_weights) + self._epsilon
        norm_prob = norm_prob/np.sum(norm_prob)
        self._cur_policies = self._random_state.choice(
            self._player_policies, p=norm_prob)
      else:
        norm_probs = [
            np.array(weights) + self._epsilon for weights in self._cur_weights]
        norm_probs = [weights/np.sum(weights) for weights in norm_probs]
        self._cur_policies = [self._random_state.choice(
            self._player_policies[n], p=norm_probs[n]) for n in range(len(norm_probs))]
      visit_path, working_state = self._apply_tree_policy(root, state)
      if working_state.is_terminal():
        returns = working_state.returns()
        visit_path[-1].outcome = returns
        solved = self.solve
      else:
        returns = self.evaluator.evaluate(working_state)
        solved = False

      for node in reversed(visit_path):
        node.total_reward += returns[root_player if node.player !=
                                     self._player_id else node.player]
        node.explore_count += 1

        if solved and node.children:
          player = node.children[0].player
          if player != self._player_id:
            # Only back up chance nodes if all have the same outcome.
            # An alternative would be to back up the weighted average of
            # outcomes if all children are solved, but that is less clear.
            outcome = node.children[0].outcome
            if (outcome is not None and
                    all(np.array_equal(c.outcome, outcome) for c in node.children)):
              node.outcome = outcome
            else:
              solved = False
          else:
            # If any have max utility (won?), or all children are solved,
            # choose the one best for the player choosing.
            best = None
            all_solved = True
            for child in node.children:
              if child.outcome is None:
                all_solved = False
              elif best is None or child.outcome[player] > best.outcome[player]:
                best = child
            if (best is not None and
                    (all_solved or best.outcome[player] == self.max_utility)):
              node.outcome = best.outcome
            else:
              solved = False
      if root.outcome is not None:
        break

    return root


class WrappedMCTSTabularPolicy(Policy):
  """A policy class wrapping and caching MCTS search results."""

  def __init__(self, game, mcts_agent):
    self._game = game
    self._mcts_agent = mcts_agent
    self._cache = {}

  def action_probabilities(self, state, player_id=None):
    if state.information_state_string() in self._cache:
      return self._cache[state.information_state_string()]
    action_and_probs = self._mcts_agent.step_with_policy(state)[0]
    self._cache[state.information_state_string()] = {
        action: prob for action, prob in action_and_probs}
    return self._cache[state.information_state_string()]


class MCTSBROracle(optimization_oracle.AbstractOracle):
  """A BR oracle class that generates MCTS-based BR policies"""

  def __init__(self, **oracle_specific_execution_kwargs):
    self._oracle_specific_execution_kwargs = oracle_specific_execution_kwargs

  def __call__(self, game, training_parameters, strategy_sampler, using_joint_strategies, **oracle_specific_execution_kwargs):
    new_policies = []
    for player_parameters in training_parameters:
      player_policies = []
      for params in player_parameters:
        current_player = params['current_player']
        total_policies = params['total_policies']
        probabilities_of_playing_policies = params['probabilities_of_playing_policies']
        if using_joint_strategies:
          mcts_agent = AgainstMixtureMCTSBot(game=game, player_id=current_player,
                                             player_policies=utils.marginal_to_joint(
                                                 utils.marginal_to_joint(total_policies)),
                                             weights=probabilities_of_playing_policies.reshape(
                                                 -1),
                                             is_joint=True, **self._oracle_specific_execution_kwargs)
        else:
          mcts_agent = AgainstMixtureMCTSBot(game=game, player_id=current_player,
                                             player_policies=total_policies,
                                             weights=probabilities_of_playing_policies,
                                             is_joint=False, **self._oracle_specific_execution_kwargs)
        player_policies.append(WrappedMCTSTabularPolicy(game, mcts_agent))
      new_policies.append(player_policies)
    return new_policies

# adapted from open_spiel.python.algorithms.exploitability


def _state_values(state, num_players, policy):
  """Value of a state for every player given a policy."""
  if state.is_terminal():
    return np.array(state.returns())
  else:
    p_action = (
        state.chance_outcomes() if state.is_chance_node() else
        policy.action_probabilities(state).items())
    return sum(prob * _state_values(state.child(action), num_players, policy)
               for action, prob in p_action)


def nash_conv(game, policy, use_cpp_br=False):
  root_state = game.new_initial_state()
  if use_cpp_br:
    best_response_values = np.array([
        pyspiel_best_response.CPPBestResponsePolicy(
            game, best_responder, policy).value(root_state)
        for best_responder in range(game.num_players())
    ])
  else:
    best_response_values = np.array([
        pyspiel_best_response.BestResponsePolicy(
            game, best_responder, policy).value(root_state)
        for best_responder in range(game.num_players())
    ])
  on_policy_values = _state_values(root_state, game.num_players(), policy)
  player_improvements = best_response_values - on_policy_values
  nash_conv_ = sum(player_improvements)
  return nash_conv_, best_response_values, on_policy_values, player_improvements


def main(_):
  rng = np.random.RandomState(FLAGS.seed)
  evaluator = RandomRolloutEvaluator(FLAGS.rollout_count, rng)
  psro_sims_per_entry = 1

  game = pyspiel.load_game("tic_tac_toe")

  mcts_br = MCTSBROracle(
      uct_c=FLAGS.uct_c, max_simulations=FLAGS.max_simulations, evaluator=evaluator)

  psro_solver = psro_v2.PSROSolver(game, mcts_br, sims_per_entry=FLAGS.psro_sims_per_entry,
                                   training_strategy_selector='probabilistic',
                                   meta_strategy_method='nash',
                                   sample_from_marginals=True)

  for it in range(FLAGS.psro_iterations):
    start_time = time.time()
    meta_game = psro_solver.get_meta_game()
    meta_probabilities = psro_solver.get_meta_strategies()
    print("------------------iter {}--------------------------------".format(it))
    print("meta game matrix for row player:")
    print(meta_game[0])
    print("meta probabilities:")
    print(meta_probabilities)
    policies = psro_solver.get_policies()
    aggregator = policy_aggregator.PolicyAggregator(game)
    aggr_policies = aggregator.aggregate(
        range(game.num_players()), policies, meta_probabilities)
    exploitabilities, br_values, on_policy_values, expl_per_player = nash_conv(
        game, aggr_policies)
    print("Nash Conv:")
    print(exploitabilities)
    print("exploitabilities")
    print(expl_per_player)
    print("original values")
    print(on_policy_values)
    print("BR values")
    print(br_values)
    tt = (time.time()-start_time)/60
    print("spent {} mins".format(tt))
    psro_solver.iteration()


if __name__ == "__main__":
  app.run(main)
