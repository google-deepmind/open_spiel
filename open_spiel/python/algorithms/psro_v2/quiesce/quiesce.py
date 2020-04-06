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
import itertools
import copy
import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import meta_strategies

# TODO: test symmetric game, as self.symmetric flags changes self.policies and self.num_players
# TODO: incomplete meta_game may be called in other part of strategy exploration. Please check
class PSROQuiesceSolver(psro_v2.PSROSolver):
  """
  quiesce class, incomplete information nash finding
  """
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
    self.update_complete_ind([0 for _ in range(self._game_num_players)],add_sample=True)
  
  def update_meta_strategies(self):
    """Recomputes the current meta strategy of each player.
    Given new payoff tables, we call self._meta_strategy_method to update the
    meta-probabilities.
    """
    if self._meta_strategy_str == 'nash' or 'general_nash':
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
      updated_policies = updated_policies[0]
      self._new_policies = [self._new_policies[0]]
      self._num_players = 1

    self._meta_games = meta_games
    self._policies = updated_policies
    self.update_complete_ind(number_older_policies,add_sample=False)
    return meta_games

  @property
  def get_complete_meta_game(self):
    """
    Returns the maximum complete game matrix
    in the same form as empirical game
    """
    selector = []
    for i in range(self._game_num_players):
      selector.append(list(np.where(np.array(self._complete_ind[i])==1)[0]))
    complete_subgame = [self._meta_games[i][np.ix_(*selector)] for i in range(self._game_num_players)] 
    return complete_subgame

  def inner_loop(self):
    """
    Find equilibrium in the incomplete self._meta_games through iteratively augment the maximum complete subgame by sampling. Symmetric game could have insymmetric nash equilibrium, so uses self._game_num_players instead of self._num_players
    Returns:
        Equilibrium support, non_margianlized profile probability
    """
    found_confirmed_eq = False
    while not found_confirmed_eq:
      maximum_subgame = self.get_complete_meta_game
      ne_subgame = meta_strategies.general_nash_strategy(solver=self, return_joint=False, game=maximum_subgame)
      # ne_support_num: list of list, index of where equilibrium is [[0,1],[2]]
      # cumsum: index ne_subgame with self._complete_ind
      cum_sum = [np.cumsum(ele) for ele in self._complete_ind]
      ne_support_num = []
      for i in range(self._game_num_players):
        ne_support_num_p = []
        for j in range(len(self._complete_ind[i])):
          if self._complete_ind[i][j]==1 and ne_subgame[i][cum_sum[i][j]-1]!=0:
            ne_support_num_p.append(j)
        ne_support_num.append(ne_support_num_p)
      # ne_subgame: non-zero equilibrium support, [[0.1,0.5,0.4],[0.2,0.4,0.4]]
      ne_subgame_nonzero = [np.array(ele) for ele in ne_subgame]
      ne_subgame_nonzero = [ele[ele!=0] for ele in ne_subgame_nonzero]
      # get players' payoffs in nash equilibrium
      ne_payoffs = self.get_mixed_payoff(ne_support_num,ne_subgame_nonzero)
      # all possible deviation payoffs
      dev_pol, dev_payoffs = self.schedule_deviation(ne_support_num,ne_subgame_nonzero)
      # check max deviations and sample full subgame where beneficial deviation included
      dev = []
      maximum_subgame_index = [list(np.where(np.array(ele)==1)[0]) for ele in self._complete_ind]
      for i in range(self._game_num_players):
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
    policy_len = [len(self._policies) for _ in range(self._game_num_players)] if self.symmetric_game else [len(ele) for ele in self._policies]
    for p in range(self._game_num_players):
      eq_p = np.zeros([policy_len[p]],dtype=float)
      np.put(eq_p,ne_support_num[p],ne_subgame_nonzero[p])
      eq.append(eq_p)
    non_marginalized_probabilities = meta_strategies.get_joint_strategy_from_marginals(eq) 
    return eq,non_marginalized_probabilities

  def schedule_deviation(self,eq,eq_sup):
    """
    Sample all possible deviation from eq
    Return a list of best deviation for each player
    if none for a player, return None for that player
    Params:
      eq     : list of list, where equilibrium is.[[1],[0]]: an example of 2x2 game
      eq_sup : list of list, contains equilibrium support for each player
    Returns:
      dev_pol: list of list, deviation payoff of policy sampled
      devs   : list of list, position of policy sampled for each player
    """
    devs = []
    dev_pol = []
    for p in range(self._game_num_players):
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
        stra_sup[p] = np.array([1.0])
        dev.append(self.get_mixed_payoff(stra_li,stra_sup)[p])
      devs.append(dev)
      dev_pol.append(possible_dev)
    return dev_pol, devs

  def get_mixed_payoff(self,strategy_list,strategy_support):
    """
    Check if the payoff exists for the profile given. If not, return False
    Params:
      strategy_list    : list of list, policy index for each player
      strategy_support : list of list, policy support probability for each player
    Returns:
      payoffs          : payoff for each player in the profile
    """
    if np.any(np.isnan(self._meta_games[0][np.ix_(*strategy_list)])):
      return False
    meta_game = [ele[np.ix_(*strategy_list)] for ele in self._meta_games]
    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(strategy_support)
    payoffs=[]
    for i in range(self._num_players):
      try:
        payoffs.append(np.sum(meta_game[i]*prob_matrix))
      except:
        import pdb
        pdb.set_trace()
    return payoffs

  def update_complete_ind(self, policy_indicator, add_sample=True):
    """
    Update the maximum completed subgame with newly added policy(one policy for each player)
    Params:
      policy_indicator: one dimensional list, policy to check, one number for each player
      add_sample      : whether there are sample added after last update
    """
    policy_len = [len(self._policies) for _ in range(self._game_num_players)] if self.symmetric_game else [len(ele) for ele in self._policies]
    for i in range(self._game_num_players):
      for _ in range(policy_len[i]-len(self._complete_ind[i])):
        self._complete_ind[i].append(0)

      if not add_sample or self._complete_ind[i][policy_indicator[i]]==1:
        continue
      selector = [list(np.where(np.array(ele)==1)[0]) for ele in self._complete_ind]
      selector[i].append(policy_indicator[i])
      if not np.any(np.isnan(self._meta_games[i][np.ix_(*selector)])):
        self._complete_ind[i][policy_indicator[i]]=1

  def sample_pure_policy_to_empirical_game(self, policy_indicator):
    """
    sample data to data grid(self.meta_game)
    Params:
      policy_indicator: 1 dim list, containing poicy to sample for each player
    Returns:
      Bool            : True if data successfully added, False is data already there
    """
    if not np.isnan(self._meta_games[0][tuple(policy_indicator)]):
      return False
    if self.symmetric_game:
      estimated_policies = [self._policies[policy_indicator[i]] for i in range(self._game_num_players)]
    else:
      estimated_policies = [self._policies[i][policy_indicator[i]] for i in range(self._game_num_players)]
    utility_estimates = self.sample_episodes(estimated_policies,self._sims_per_entry)
    for k in range(self._game_num_players):
      self._meta_games[k][tuple(policy_indicator)] = utility_estimates[k]
    self.update_complete_ind(policy_indicator,add_sample=True)    
    return True
