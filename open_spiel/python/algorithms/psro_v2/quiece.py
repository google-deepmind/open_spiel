# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Modular implementations of the PSRO meta algorithm.

Allows the use of Restricted Nash Response, Nash Response, Uniform Response,
and other modular matchmaking selection components users can add.

This version works for N player, general sum games.

One iteration of the algorithm consists of:

1) Computing the selection probability vector (or meta-strategy) for current
strategies of each player, given their payoff.
2) [optional] Generating a mask over joint policies that restricts which policy
to train against, ie. rectify the set of policies trained against. (This
operation is designated by "rectify" in the code)
3) From every strategy used, generating a new best response strategy against the
meta-strategy-weighted, potentially rectified, mixture of strategies using an
oracle.
4) Updating meta game matrix with new game results.

"""

import itertools

import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms.psro_v2 import abstract_meta_trainer
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms.psro_v2 import PSROSolver

class PSROQuiesceSolver(PSROSolver):
  """
  quiesce class, incomplete information nash finding
  """
  def __init__(self,
               game,
               oracle,
               sims_per_entry,
               initial_policies=None,
               rectifier="",
               training_strategy_selector=None,
               meta_strategy_method="alpharank",
               sample_from_marginals=False,
               number_policies_selected=1,
               n_noisy_copies=0,
               alpha_noise=0.0,
               beta_noise=0.0,
               **kwargs):
    self._meta_strategy_str = meta_strategy_method
    super(PSROQuiesceSolver, self).__init__(**locals())

  def _initialize_policy(self, initial_policies):
    self._policies = [[] for k in range(self._num_players)]
    self._new_policies = [([initial_policies[k]] if initial_policies else
                           [policy.UniformRandomPolicy(self._game)])
                          for k in range(self._num_players)]
    self._complete_ind = [[] for _ in range(self._num_players)]

  def _initialize_game_state(self):
    effective_payoff_size = self._game_num_players
    self._meta_games = [
        np.array(utils.empty_list_generator(effective_payoff_size))
        for _ in range(effective_payoff_size)
    ]
    super(PSROQuiesceSolver,self).update_empirical_gamestate(seed=None)
  
  def update_meta_strategies(self):
    """Recomputes the current meta strategy of each player.
    Given new payoff tables, we call self._meta_strategy_method to update the
    meta-probabilities.
    """
    if self._meta_strategy_str == 'nash':
        self._meta_strategy_probabilities,self._non_marginalized_probabilities = self.inner_loop()
    else:
        super(PSROQuiesceSolver,self).update_meta_strategies()

  def update_empirical_gamestate(self, seed=None):
    """Given new agents in _new_policies, update meta_games through simulations. For quiesce, only update the meta game grid, but does not need to fill in values. If filling in value required, use parent class method

    Args:
      seed: Seed for environment generation.

    Returns:
      Meta game payoff matrix.
    """
    if seed is not None:
      np.random.seed(seed=seed)
    assert self._oracle is not None

    if self.symmetric_game:
      # Switch to considering the game as a symmetric game where players have
      # the same policies & new policies. This allows the empirical gamestate
      # update to function normally.
      self._policies = self._game_num_players * self._policies
      self._new_policies = self._game_num_players * self._new_policies
      self._num_players = self._game_num_players

    # Concatenate both lists.
    updated_policies = [
        self._policies[k] + self._new_policies[k]
        for k in range(self._num_players)
    ]

    # Each metagame will be (num_strategies)^self._num_players.
    # There are self._num_player metagames, one per player.
    total_number_policies = [
        len(updated_policies[k]) for k in range(self._num_players)
    ]
    number_older_policies = [
        len(self._policies[k]) for k in range(self._num_players)
    ]
    number_new_policies = [
        len(self._new_policies[k]) for k in range(self._num_players)
    ]

    # Initializing the matrix with nans to recognize unestimated states.
    meta_games = [
        np.full(tuple(total_number_policies), np.nan)
        for k in range(self._num_players)
    ]

    # Filling the matrix with already-known values.
    older_policies_slice = tuple(
        [slice(len(self._policies[k])) for k in range(self._num_players)])
    for k in range(self._num_players):
      meta_games[k][older_policies_slice] = self._meta_games[k]

    if self.symmetric_game:
      # Make PSRO consider that we only have one population again, as we
      # consider that we are in a symmetric game (No difference between players)
      self._policies = [self._policies[0]]
      self._new_policies = [self._new_policies[0]]
      updated_policies = [updated_policies[0]]
      self._num_players = 1

    self._meta_games = meta_games
    self._policies = updated_policies
    self.update_complete_ind(number_older_policies,add_sample=add_sample)
    return meta_games

  @property
  def get_complete_meta_game(self):
    """
    Returns the maximum complete game matrix
    in the same form as empirical game
    """
    selector = []
    for i in range(self._num_players):
      selector.append(list(np.where(np.array(self._complete_ind[i])==1)[0]))
    complete_subgame = [self._meta_games[i][np.ix_(*selector)] for i in range(self._num_players)] 
    return complete_subgame

  def get_and_update_non_marginalized_meta_strategies(self, update=True):
    """Returns the Nash Equilibrium distribution on meta game matrix."""
    if update:
      self.update_meta_strategies()
    return self._non_marginalized_probabilities

  def inner_loop(self):
      found_confirmed_eq = False
      while not found_confirmed_eq:
          maximum_subgame = self.get_complete_meta_game
          ne_subgame,non_marginalized_probabilities = self._meta_strategy_method(solver=self,return_joint=True,maximum_subgame)
          # ne_support_num: list of list, index of where equilibrium is [[0,1],[2]]
          # cumsum: index ne_subgame with self._complete_ind
          cum_sum = [np.cumsum(ele) for ele in self._complete_ind]
          ne_support_num = []
          for i in range(self._num_players):
            ne_support_num_p = []
            for j in range(len(self._complete_ind[i])):
              if self._complete_ind[i][j]==1 and ne_subgame[i][cum_sum[i][j]-1]!=0:
                ne_support_num_p.append(j)
            ne_support_num.append(ne_support_num_p)
          # ne_subgame: non-zero equilibrium support, [[0.1,0.5,0.4],[0.2,0.4,0.4]]
          ne_subgame_nonzero = [np.array(ele) for ele in ne_subgame]
          ne_subgame_nonzero = [list(ele[ele!=0]) for ele in ne_subgame_nonzero]
          # get players' payoffs in nash equilibrium
          ne_payoffs = self.get_mixed_payoff(ne_support_num,ne_subgame_nonzero)
          # all possible deviation payoffs
          dev_pol, dev_payoffs = self.schedule_deviation(ne_support_num,ne_subgame_nonzero)
          # check max deviations and sample full subgame where beneficial deviation included
          dev = []
          maximum_subgame_index = [list(np.where(np.array(ele)==1)[0]) for ele in self._complete_ind]
          for i in range(self._num_players):
              if not len(dev_payoffs[i])==0 and max(dev_payoffs[i]) > ne_payoffs[i]:
                  pol = dev_pol[i][np.argmax(dev_payoffs[i])]
                  new_subgame_sample_ind = copy.deepcopy(maximum_subgame_index)
                  maximum_subgame_index[i].append(pol)
                  new_subgame_sample_ind[i] = [pol]
                  # add best deviation into subgame and sample it
                  for pol in itertools.product(*new_subgame_sample_ind):
                      self.sample_pure_policy_to_empirical_game(pol) 
                  dev.append(i)
                  # all other player's policies have to sample previous players' best deviation
          found_confirmed_eq = (len(dev)==0)
          # debug: check maximum subgame remains the same
          # debug: check maximum game reached

      # return confirmed nash equilibrium
      eq = []
      for p in range(self._num_players):
          eq_p = np.zeros([len(self._policies[p])],dtype=float)
          np.put(eq_p,ne_support_num[p],ne_subgame_nonzero[p])
          eq.append(eq_p)
      return eq 

  def schedule_deviation(self,eq,eq_sup):
      """
      Sample all possible deviation from eq
      Return a list of best deviation for each player
      if none for a player, return None for that player
      Params:
        eq     :  list of list, where equilibrium is.[[1],[0]]: an example of 2x2 game
        eq_sup :  Exact same shape with eq. Documents equilibrium support
      """
      devs = []
      dev_pol = []
      for p in range(self._num_players):
          # check all possible deviations
          dev = []
          possible_dev = list(np.where(np.array(self._complete_ind[p])==0)[0])
          iter_eq = copy.deepcopy(eq)
          iter_eq[p] = possible_dev
          for pol in itertools.product(*iter_eq):
              self.sample_pure_policy_to_empirical_game(pol)
          for pol in possible_dev:
              stra_li,stra_sup = copy.deepcopy(eq),copy.deepcopy(eq_sup)
              stra_li[p] = [pol]
              stra_sup[p] = [1.0]
              dev.append(self.get_mixed_payoff(stra_li,stra_sup)[p])
          devs.append(dev)
          dev_pol.append(possible_dev)
      return dev_pol, devs

  def get_mixed_payoff(self,strategy_list,strategy_support):
      """
      Check if the payoff exists. If not, return False
      Params:
        strategy_list    : [[],[]] 2 dimensional list, nash support policy index
        strategy_support : [[],[]] 2 dimensional list, nash support probability for corresponding strategies
      """
      if np.any(np.isnan(self._meta_games[0][np.ix_(*strategy_list)])):
        return False
      meta_game = [ele[np.ix_(*strategy_list)] for ele in self._meta_games]
      # multiple player prob_matrix tensor
      prob_matrix = np.outer(strategy_support[0],strategy_support[1])
      for i in range(self._num_players-2):
          ind = tuple([Ellipsis for _ in range(len(prob_matrix.shape))]+[None])
          prob_matrix = prob_matrix[ind]*strategy_support[i+2]
      payoffs=[]
      for i in range(self._num_players):
          payoffs.append(np.sum(meta_game[i]*prob_matrix))
      return payoffs

  def update_complete_ind(self, policy_indicator, add_sample=True):
    """
    Find the maximum completed subgame with newly added policy(one policy for each player)
    Params:
    policy_indicator: policy to check, one number for each player
    add_sample      : whether there are sample data added after last update
    """
    for i in range(self._num_players):
      for _ in range(len(self._policies[i])-len(self._complete_ind[i])):
        self._complete_ind[i].append(0)

      if not add_sample or self._complete_ind[i][policy_indicator[i]]==1:
        continue
      selector = [list(np.where(np.array(ele)==1)[0]) for ele in self._complete_ind]
      selector[i].append(policy_indicator[i])
      if not np.any(np.isnan(self._meta_games[i][np.ix_(*selector)])):
        self._complete_ind[i][policy_indicator[i]]=1

  def sample_pure_policy_to_empirical_game(self, policy_indicator):
      """
      add data samples to data grid
      Params:
        policy_indicator: 1 dim list, containing poicy to sample for each player
      """
      if not np.isnan(self._meta_games[0][tuple(policy_indicator)]):
        return False
      estimated_policies = [self._policies[i][policy_indicator[i]] for i in range(self._num_players)]
      utility_estimates = self.rl_sample_episodes(estimated_policies,self._sims_per_entry)
      for k in range(self._num_players):
        self._meta_games[k][tuple(policy_indicator)] = utility_estimates[k]
      self.update_complete_ind(policy_indicator,add_sample=True)    
      return True

