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


TRAIN_TARGET_SELECTORS = {
    "": None,
    "rectified": strategy_selectors.rectified_selector,
}


class PSROSolver(abstract_meta_trainer.AbstractMetaTrainer):
  """A general implementation PSRO.

  PSRO is the algorithm described in (Lanctot et Al., 2017,
  https://arxiv.org/pdf/1711.00832.pdf ).

  Subsequent work regarding PSRO's matchmaking and training has been performed
  by David Balduzzi, who introduced Restricted Nash Response (RNR), Nash
  Response (NR) and Uniform Response (UR).
  RNR is Algorithm 4 in (Balduzzi, 2019, "Open-ended Learning in Symmetric
  Zero-sum Games"). NR, Nash response, is algorithm 3.
  Balduzzi et Al., 2019, https://arxiv.org/pdf/1901.08106.pdf

  This implementation allows one to modularly choose different meta strategy
  computation methods, or other user-written ones.
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
    """Initialize the PSRO solver.

    Arguments:
      game: The open_spiel game object.
      oracle: Callable that takes as input: - game - policy - policies played -
        array representing the probability of playing policy i - other kwargs
        and returns a new best response.
      sims_per_entry: Number of simulations to run to estimate each element of
        the game outcome matrix.
      initial_policies: A list of initial policies for each player, from which
        the optimization process will start.
      rectifier: A string indicating the rectifying method. Can be :
              - "" or None: Train against potentially all strategies.
              - "rectified": Train only against strategies beaten by current
                strategy.
      training_strategy_selector: Callable taking (PSROSolver,
        'number_policies_selected') and returning a list of list of selected
        strategies to train from - this usually means copying weights and
        rectifying with respect to the selected strategy's performance (One list
        entry per player), or string selecting pre-implemented methods.
        String value can be:
              - "top_k_probabilites": selects the first
              'number_policies_selected' policies with highest selection
              probabilities.
              - "probabilistic": randomly selects 'number_policies_selected'
                with probabilities determined by the meta strategies.
              - "exhaustive": selects every policy of every player.
              - "rectified": only selects strategies that have nonzero chance of
                being selected.
              - "uniform": randomly selects 'number_policies_selected'
                policies with uniform probabilities.
      meta_strategy_method: String or callable taking a GenPSROSolver object and
        returning two lists ; one list of meta strategies (One list entry per
        player), and one list of joint strategies.
        String value can be:
              - alpharank: AlphaRank distribution on policies.
              - "uniform": Uniform distribution on policies.
              - "nash": Taking nash distribution. Only works for 2 player, 0-sum
                games.
              - "prd": Projected Replicator Dynamics, as described in Lanctot et
                Al.
      sample_from_marginals: A boolean, specifying whether to sample from
        marginal (True) or joint (False) meta-strategy distributions.
      number_policies_selected: Number of policies to return for each player.

      n_noisy_copies: Number of noisy copies of each agent after training. 0 to
        ignore this.
      alpha_noise: lower bound on alpha noise value (Mixture amplitude.)
      beta_noise: lower bound on beta noise value (Softmax temperature.)
      **kwargs: kwargs for meta strategy computation and training strategy
        selection.
    """
    self._sims_per_entry = sims_per_entry
    print("Using {} sims per entry.".format(sims_per_entry))

    self._rectifier = TRAIN_TARGET_SELECTORS.get(
        rectifier, None)
    self._rectify_training = self._rectifier
    print("Rectifier : {}".format(rectifier))

    self._meta_strategy_probabilities = np.array([])
    self._non_marginalized_probabilities = np.array([])

    print("Perturbating oracle outputs : {}".format(n_noisy_copies > 0))
    self._n_noisy_copies = n_noisy_copies
    self._alpha_noise = alpha_noise
    self._beta_noise = beta_noise

    self._policies = []  # A list of size `num_players` of lists containing the
    # strategies of each player.
    self._new_policies = []

    # Alpharank is a special case here, as it's not supported by the abstract
    # meta trainer api, so has to be passed as a function instead of a string.
    if not meta_strategy_method or meta_strategy_method == "alpharank":
      meta_strategy_method = utils.alpharank_strategy

    print("Sampling from marginals : {}".format(sample_from_marginals))
    self.sample_from_marginals = sample_from_marginals

    super(PSROSolver, self).__init__(
        game,
        oracle,
        initial_policies,
        meta_strategy_method,
        training_strategy_selector,
        number_policies_selected=number_policies_selected,
        **kwargs)

  def _initialize_policy(self, initial_policies):
    self._policies = [[] for k in range(self._num_players)]
    self._new_policies = [([initial_policies[k]] if initial_policies else
                           [policy.UniformRandomPolicy(self._game)])
                          for k in range(self._num_players)]

  def _initialize_game_state(self):
    effective_payoff_size = self._game_num_players
    self._meta_games = [
        np.array(utils.empty_list_generator(effective_payoff_size))
        for _ in range(effective_payoff_size)
    ]
    self.update_empirical_gamestate(seed=None)

  def get_joint_policy_ids(self):
    """Returns a list of integers enumerating all joint meta strategies."""
    return utils.get_strategy_profile_ids(self._meta_games)

  def get_joint_policies_from_id_list(self, selected_policy_ids):
    """Returns a list of joint policies from a list of integer IDs.

    Args:
      selected_policy_ids: A list of integer IDs corresponding to the
        meta-strategies, with duplicate entries allowed.

    Returns:
      selected_joint_policies: A list, with each element being a joint policy
        instance (i.e., a list of policies, one per player).
    """
    policies = self.get_policies()

    selected_joint_policies = utils.get_joint_policies_from_id_list(
        self._meta_games, policies, selected_policy_ids)
    return selected_joint_policies

  def update_meta_strategies(self):
    """Recomputes the current meta strategy of each player.

    Given new payoff tables, we call self._meta_strategy_method to update the
    meta-probabilities.
    """
    if self.symmetric_game:
      self._policies = self._policies * self._game_num_players

    self._meta_strategy_probabilities, self._non_marginalized_probabilities = (
        self._meta_strategy_method(solver=self, return_joint=True))

    if self.symmetric_game:
      self._policies = [self._policies[0]]
      self._meta_strategy_probabilities = [self._meta_strategy_probabilities[0]]

  def get_policies_and_strategies(self):
    """Returns current policy sampler, policies and meta-strategies of the game.

    If strategies are rectified, we automatically switch to returning joint
    strategies.

    Returns:
      sample_strategy: A strategy sampling function
      total_policies: A list of list of policies, one list per player.
      probabilities_of_playing_policies: the meta strategies, either joint or
        marginalized.
    """
    sample_strategy = utils.sample_strategy_marginal
    probabilities_of_playing_policies = self.get_meta_strategies()
    if self._rectify_training or not self.sample_from_marginals:
      sample_strategy = utils.sample_strategy_joint
      probabilities_of_playing_policies = self._non_marginalized_probabilities

    total_policies = self.get_policies()
    return sample_strategy, total_policies, probabilities_of_playing_policies

  def _restrict_target_training(self,
                                current_player,
                                ind,
                                total_policies,
                                probabilities_of_playing_policies,
                                restrict_target_training_bool,
                                epsilon=1e-12):
    """Rectifies training.

    Args:
      current_player: the current player.
      ind: Current strategy index of the player.
      total_policies: all policies available to all players.
      probabilities_of_playing_policies: meta strategies.
      restrict_target_training_bool: Boolean specifying whether to restrict
        training. If False, standard meta strategies are returned. Otherwise,
        restricted joint strategies are returned.
      epsilon: threshold below which we consider 0 sum of probabilities.

    Returns:
      Probabilities of playing each joint strategy (If rectifying) / probability
      of each player playing each strategy (Otherwise - marginal probabilities)
    """
    true_shape = tuple([len(a) for a in total_policies])
    if not restrict_target_training_bool:
      return probabilities_of_playing_policies
    else:
      kept_probas = self._rectifier(
          self, current_player, ind)
      # Ensure probabilities_of_playing_policies has same shape as kept_probas.
      probability = probabilities_of_playing_policies.reshape(true_shape)
      probability = probability * kept_probas
      prob_sum = np.sum(probability)

      # If the rectified probabilities are too low / 0, we play against the
      # non-rectified probabilities.
      if prob_sum <= epsilon:
        probability = probabilities_of_playing_policies
      else:
        probability /= prob_sum

      return probability

  def update_agents(self):
    """Updates policies for each player at the same time by calling the oracle.

    The resulting policies are appended to self._new_policies.
    """

    used_policies, used_indexes = self._training_strategy_selector(
        self, self._number_policies_selected)

    (sample_strategy,
     total_policies,
     probabilities_of_playing_policies) = self.get_policies_and_strategies()

    # Contains the training parameters of all trained oracles.
    # This is a list (Size num_players) of list (Size num_new_policies[player]),
    # each dict containing the needed information to train a new best response.
    training_parameters = [[] for _ in range(self._num_players)]

    for current_player in range(self._num_players):
      if self.sample_from_marginals:
        currently_used_policies = used_policies[current_player]
        current_indexes = used_indexes[current_player]
      else:
        currently_used_policies = [
            joint_policy[current_player] for joint_policy in used_policies
        ]
        current_indexes = used_indexes[current_player]

      for i in range(len(currently_used_policies)):
        pol = currently_used_policies[i]
        ind = current_indexes[i]

        new_probabilities = self._restrict_target_training(
            current_player, ind, total_policies,
            probabilities_of_playing_policies,
            self._rectify_training)

        new_parameter = {
            "policy": pol,
            "total_policies": total_policies,
            "current_player": current_player,
            "probabilities_of_playing_policies": new_probabilities
        }
        training_parameters[current_player].append(new_parameter)

    if self.symmetric_game:
      self._policies = self._game_num_players * self._policies
      self._num_players = self._game_num_players
      training_parameters = [training_parameters[0]]

    # List of List of new policies (One list per player)
    self._new_policies = self._oracle(
        self._game,
        training_parameters,
        strategy_sampler=sample_strategy,
        using_joint_strategies=self._rectify_training or
        not self.sample_from_marginals)

    if self.symmetric_game:
      # In a symmetric game, only one population is kept. The below lines
      # therefore make PSRO consider only the first player during training,
      # since both players are identical.
      self._policies = [self._policies[0]]
      self._num_players = 1

  def update_empirical_gamestate(self, seed=None):
    """Given new agents in _new_policies, update meta_games through simulations.

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

    # Filling the matrix for newly added policies.
    for current_player in range(self._num_players):
      # Only iterate over new policies for current player ; compute on every
      # policy for the other players.
      range_iterators = [
          range(total_number_policies[k]) for k in range(current_player)
      ] + [range(number_new_policies[current_player])] + [
          range(total_number_policies[k])
          for k in range(current_player + 1, self._num_players)
      ]
      for current_index in itertools.product(*range_iterators):
        used_index = list(current_index)
        used_index[current_player] += number_older_policies[current_player]
        if np.isnan(meta_games[current_player][tuple(used_index)]):
          estimated_policies = [
              updated_policies[k][current_index[k]]
              for k in range(current_player)
          ] + [
              self._new_policies[current_player][current_index[current_player]]
          ] + [
              updated_policies[k][current_index[k]]
              for k in range(current_player + 1, self._num_players)
          ]

          if self.symmetric_game:
            # TODO(author4): This update uses ~2**(n_players-1) * sims_per_entry
            # samples to estimate each payoff table entry. This should be
            # brought to sims_per_entry to coincide with expected behavior.
            utility_estimates = self.sample_episodes(estimated_policies,
                                                     self._sims_per_entry)

            player_permutations = list(itertools.permutations(list(range(
                self._num_players))))
            for permutation in player_permutations:
              used_tuple = tuple([used_index[i] for i in permutation])
              for player in range(self._num_players):
                if np.isnan(meta_games[player][used_tuple]):
                  meta_games[player][used_tuple] = 0.0
                meta_games[player][used_tuple] += utility_estimates[
                    permutation[player]] / len(player_permutations)
          else:
            utility_estimates = self.sample_episodes(estimated_policies,
                                                     self._sims_per_entry)
            for k in range(self._num_players):
              meta_games[k][tuple(used_index)] = utility_estimates[k]

    if self.symmetric_game:
      # Make PSRO consider that we only have one population again, as we
      # consider that we are in a symmetric game (No difference between players)
      self._policies = [self._policies[0]]
      self._new_policies = [self._new_policies[0]]
      updated_policies = [updated_policies[0]]
      self._num_players = 1

    self._meta_games = meta_games
    self._policies = updated_policies
    return meta_games

  def get_meta_game(self):
    """Returns the meta game matrix."""
    return self._meta_games

  @property
  def meta_games(self):
    return self._meta_games

  def get_policies(self):
    """Returns a list, each element being a list of each player's policies."""
    policies = self._policies
    if self.symmetric_game:
      # For compatibility reasons, return list of expected length.
      policies = self._game_num_players * self._policies
    return policies

  def get_and_update_non_marginalized_meta_strategies(self, update=True):
    """Returns the Nash Equilibrium distribution on meta game matrix."""
    if update:
      self.update_meta_strategies()
    return self._non_marginalized_probabilities

  def get_strategy_computation_and_selection_kwargs(self):
    return self._strategy_computation_and_selection_kwargs
