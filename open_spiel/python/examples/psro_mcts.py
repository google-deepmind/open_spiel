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
Notice that we implemented a tabular-version of AlphaZero wrapping MCTS as the BR method.
For games with medium or large size of state space (even for tic-tac-toe) 
it needs a huge number of training episodes to learn an accurate value estimation.
This suggests the motivation for using function approximator.
"""


from absl import app
from absl import flags

from open_spiel.python.algorithms.mcts import MCTSBot
from open_spiel.python.algorithms.mcts import SearchNode
from open_spiel.python.algorithms.mcts import Evaluator
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator
from open_spiel.python.algorithms.psro_v2 import optimization_oracle
from open_spiel.python.algorithms.psro_v2.utils import marginal_to_joint
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms import best_response as pyspiel_best_response
from open_spiel.python.policy import Policy
from open_spiel.python import rl_tools

import pyspiel
import copy
import time
import numpy as np


FLAGS = flags.FLAGS



# MCTS related
flags.DEFINE_float("uct_c", 2, "uct_c for MCTS")
flags.DEFINE_integer("max_simulations", 100,
                     "max iterations for MCTS simulations phase")

# PSRO related
flags.DEFINE_integer("psro_sims_per_entry", 100,
                     "simulation numbers for meta-game entry")
flags.DEFINE_integer("psro_iterations", 100, "number of PSRO iterations")

# AZ related
flags.DEFINE_integer("az_episodes", 100000,
                     "episodes for AZ training")
flags.DEFINE_float("v_lr", 0.2, "AZ value learning rate")
flags.DEFINE_float("p_lr", 0.2, "AZ prior learning rate")
flags.DEFINE_integer("az_delay", 10, "AZ delay")
flags.DEFINE_bool(
    "use_obs", True, "use obs string or infoset for storing values")


class AgainstMixtureMCTSBot(MCTSBot):
  """An MCTS Bot searching against a mixture of opponent.

    Args:
      player_id: the id of the searching player
      player_policies: if is_joint, then a list of joint policy each of size num_of_player;
      otherwise a list of num_of_player per-individual policy set
      weights: the corresponding un-normalized probablistic weights for player_policies
      is_joint: whether player_policies is represented as joint-policies or not
      epsilon: a trembling-hand parameter
      **mcts_base_params: mcts-related parameters
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
    """Computing correct posterior weights for player_policies."""
    working_state = self._game.new_initial_state()
    history = state.full_history()
    self._cur_weights = copy.deepcopy(self._weights)
    for h in history:
      if h.player != self._player_id and h.player != pyspiel.PlayerId.CHANCE:
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

  def get_evaluator(self):
    return self.evaluator

  def set_evaluator(self, evaluator):
    self._evaluator = evaluator

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
        assert len(legal_actions) == len(working_state.legal_actions())
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
      if self._is_joint:
        norm_prob = np.array(self._cur_weights) + self._epsilon
        norm_prob = norm_prob/np.sum(norm_prob)
        policy_idx = self._random_state.choice(
            len(self._player_policies), p=norm_prob)
        self._cur_policies = self._player_policies[policy_idx]
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
        node.total_reward += returns[self._player_id]
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


class LearnedTabularEvaluator(Evaluator):
  """A Tabular AlphaZero styled evaluator.

    Args:
      player_id: the player_id of the considered agent
      initial_evaluator: the default evaluator for the initial values
      use_obs: using observation_string() or information_state_string() for storing values
  """

  def __init__(self, player_id, initial_evaluator=None, use_obs=True):
    self._initial_evaluator = initial_evaluator or RandomRolloutEvaluator(1)
    self._state_values = {}
    self._priors = {}
    self._player_id = player_id
    self._use_obs = use_obs

  def get_state_key(self, state):
    if self._use_obs:
      return state.observation_string(self._player_id)
    else:
      return state.information_state_string(self._player_id)

  def evaluate(self, state):
    state_key = self.get_state_key(state)
    if state_key not in self._state_values:
      self._state_values[state_key] = np.array(
          self._initial_evaluator.evaluate(state))
    return self._state_values[state_key]

  def prior(self, state):
    state_key = self.get_state_key(state)
    if state_key not in self._priors:
      self._priors[state_key] = dict(self._initial_evaluator.prior(state))
    return list(self._priors[state_key].items())

  def learn_value_step(self, state, value_target, lr=0.001):
    state_key = self.get_state_key(state)
    state_value = self.evaluate(state)
    self._state_values[state_key] = (
        1-lr)*np.array(state_value) + lr*np.array(value_target)

  def learn_prior_step(self, state, prior_target, lr=0.001):
    state_key = self.get_state_key(state)
    state_prior = dict(self.prior(state))
    policy_map = {}
    for action, prob in state_prior.items():
      policy_map[action] = prob*(1-lr)
    for action, prob in prior_target.items():
      if action not in policy_map:
        policy_map[action] = prob*lr
      else:
        policy_map[action] += prob*lr
    self._priors[state_key] = policy_map


class AZMCTSTabularPolicy(Policy):
  """A Wrapper policy class for tabular-based AlphaZero policy.

    The wrapper class wraps an MCTS agent and provides a AlphaZero-like training method.
  """

  def __init__(self,
               game,
               player_id,
               player_policies,
               weights,
               is_joint=False,
               use_obs=True,
               random_state=None,
               epsilon_schedule=rl_tools.ConstantSchedule(0.1),
               **mcts_base_params):
    self._game = game
    self._player_id = player_id
    self._player_policies = player_policies
    self._weights = weights
    self._is_joint = is_joint
    self._evaluator = LearnedTabularEvaluator(
        player_id=player_id, use_obs=use_obs)
    self._mcts_bot = AgainstMixtureMCTSBot(game=game,
                                           player_id=player_id,
                                           player_policies=player_policies,
                                           weights=weights,
                                           is_joint=is_joint,
                                           evaluator=copy.deepcopy(
                                               self._evaluator),
                                           child_selection_fn=SearchNode.puct_value,
                                           solve=False,
                                           dont_return_chance_node=True,
                                           **mcts_base_params)
    self._random_state = random_state or np.random.RandomState()
    self._epsilon_schedule = epsilon_schedule

  def az_train(self, num_episodes=1000, v_lr=0.01, p_lr=0.01, delay_interval=10):
    for epi in range(num_episodes):
      if self._is_joint:
        norm_prob = np.array(self._weights)
        norm_prob = norm_prob/np.sum(norm_prob)
        policy_idx = self._random_state.choice(
            len(self._player_policies), p=norm_prob)
        cur_policies = self._player_policies[policy_idx]
      else:
        norm_probs = [np.array(weights) for weights in self._weights]
        norm_probs = [weights/np.sum(weights) for weights in norm_probs]
        cur_policies = [self._random_state.choice(
            self._player_policies[n], p=norm_probs[n]) for n in range(len(norm_probs))]

      state = self._game.new_initial_state()
      states = []
      prior_targets = []

      while not state.is_terminal():
        if state.current_player() != self._player_id:
          if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
          else:
            cur_player = state.current_player()
            action_list, prob_list = zip(
                *(cur_policies[cur_player].action_probabilities(state)).items())
          action = self._random_state.choice(action_list, p=prob_list)
        else:
          policy, _ = self._mcts_bot.step_with_policy(state)
          prior_targets.append((state.clone(), dict(policy)))
          epsilon = self._epsilon_schedule.step()
          policy = [(action, (1-epsilon)*prob + epsilon/len(policy))
                    for (action, prob) in policy]
          action_list, prob_list = zip(*policy)
          action = self._random_state.choice(action_list, p=prob_list)
        state.apply_action(action)
        states.append(state.clone())
      for i in range(len(states)):
        self._evaluator.learn_value_step(states[i], state.returns(), v_lr)
      for i in range(len(prior_targets)):
        self._evaluator.learn_prior_step(
            prior_targets[i][0], prior_targets[i][1], p_lr)
      if epi and epi % delay_interval == 0:
        self._mcts_bot.set_evaluator(copy.deepcopy(self._evaluator))

  def action_probabilities(self, state, player_id=None):
    """Return a greedy policy of the learned value estimation."""
    assert state.current_player() == self._player_id
    child_values = [(action, self._evaluator.evaluate(state.child(action)))
                    for action in state.legal_actions()]
    max_value = max(child_values, key=lambda x: x[1][self._player_id])[
        1][self._player_id]
    max_actions = [action for action,
                   value in child_values if value[self._player_id] == max_value]
    policy = [(action, 1./len(max_actions)) if value[self._player_id]
              == max_value else (action, 0.) for action, value in child_values]
    return dict(policy)


class AZMCTSBROracle(optimization_oracle.AbstractOracle):
  """Wrapping AZMCTSTabularPolicy as a BR oracle in PSRO."""

  def __init__(self, az_episodes=1000, v_lr=0.001, p_lr=0.001, az_delay=10, **mcts_base_params):
    self._az_episodes = az_episodes
    self._v_lr = v_lr
    self._p_lr = p_lr
    self._az_delay = az_delay
    self._mcts_base_params = mcts_base_params

  def __call__(self, game, training_parameters, strategy_sampler, using_joint_strategies, **oracle_specific_execution_kwargs):
    new_policies = []
    for player_parameters in training_parameters:
      player_policies = []
      for params in player_parameters:
        current_player = params['current_player']
        total_policies = params['total_policies']
        probabilities_of_playing_policies = params['probabilities_of_playing_policies']

        if using_joint_strategies:
          az_policy = AZMCTSTabularPolicy(game=game,
                                          player_id=current_player,
                                          player_policies=marginal_to_joint(
                                              total_policies),
                                          weights=probabilities_of_playing_policies.reshape(
                                              -1),
                                          is_joint=True,
                                          **self._mcts_base_params)
        else:
          az_policy = AZMCTSTabularPolicy(game=game,
                                          player_id=current_player,
                                          player_policies=total_policies,
                                          weights=probabilities_of_playing_policies,
                                          is_joint=False,
                                          **self._mcts_base_params)

        az_policy.az_train(num_episodes=self._az_episodes, v_lr=self._v_lr,
                           p_lr=self._p_lr, delay_interval=self._az_delay)
        player_policies.append(az_policy)
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

  game = pyspiel.load_game("tic_tac_toe")


  mcts_br = AZMCTSBROracle(az_episodes=FLAGS.az_episodes,
                           v_lr=FLAGS.v_lr,
                           p_lr=FLAGS.p_lr,
                           az_delay=FLAGS.az_delay,
                           uct_c=FLAGS.uct_c,
                           use_obs=FLAGS.use_obs,
                           max_simulations=FLAGS.max_simulations)

  psro_solver = psro_v2.PSROSolver(game, mcts_br, sims_per_entry=FLAGS.psro_sims_per_entry,
                                   training_strategy_selector='probabilistic',
                                   meta_strategy_method='nash',
                                   sample_from_marginals=True)

  start_time = time.time()
  for it in range(FLAGS.psro_iterations):
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
    start_time = time.time()
    psro_solver.iteration()


if __name__ == "__main__":
  app.run(main)
