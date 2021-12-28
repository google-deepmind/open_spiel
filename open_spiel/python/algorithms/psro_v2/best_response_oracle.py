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
"""An Oracle for Exact Best Responses.

This class computes the best responses against sets of policies.
"""

from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import policy_utils
from open_spiel.python.algorithms.psro_v2 import optimization_oracle
from open_spiel.python.algorithms.psro_v2 import utils
import pyspiel


class BestResponseOracle(optimization_oracle.AbstractOracle):
  """Oracle using exact best responses to compute BR to policies."""

  def __init__(self,
               best_response_backend='cpp',
               game=None,
               all_states=None,
               state_to_information_state=None,
               **kwargs):
    """Init function for the RLOracle.

    Args:
      best_response_backend: A string (either 'cpp' or 'py'), specifying the
        best response backend to use (C++ or python, respectively). The cpp
        backend should be preferred, generally, as it is significantly faster.
      game: The game on which the optimization process takes place.
      all_states: The result of calling get_all_states.get_all_states. Cached
        for improved performance.
      state_to_information_state: A dict mapping str(state) to
        state.information_state for every state in the game. Cached for improved
        performance.
      **kwargs: kwargs
    """
    super(BestResponseOracle, self).__init__(**kwargs)
    self.best_response_backend = best_response_backend
    if self.best_response_backend == 'cpp':
      # Should compute all_states and state_to_information_state only once in
      # the program, as caching them speeds up TabularBestResponse tremendously.
      self.all_states, self.state_to_information_state = (
          utils.compute_states_and_info_states_if_none(
              game, all_states, state_to_information_state))

      policy = openspiel_policy.UniformRandomPolicy(game)

      policy_to_dict = policy_utils.policy_to_dict(
          policy, game, self.all_states, self.state_to_information_state)

      # pylint: disable=g-complex-comprehension
      # Cache TabularBestResponse for players, due to their costly construction
      # TODO(b/140426861): Use a single best-responder once the code supports
      # multiple player ids.
      self.best_response_processors = [
          pyspiel.TabularBestResponse(game, best_responder_id, policy_to_dict)
          for best_responder_id in range(game.num_players())
      ]
      self.best_responders = [
          best_response.CPPBestResponsePolicy(
              game, i_player, policy, self.all_states,
              self.state_to_information_state,
              self.best_response_processors[i_player]
          )
          for i_player in range(game.num_players())
      ]
      # pylint: enable=g-complex-comprehension

  def __call__(self,
               game,
               training_parameters,
               strategy_sampler=utils.sample_strategy,
               using_joint_strategies=False,
               **oracle_specific_execution_kwargs):
    """Call method for oracle, returns best responses for training_parameters.

    Args:
      game: The game on which the optimization process takes place.
      training_parameters: List of list of dicts: one list per player, one dict
        per selected agent in the pool for each player,
        each dictionary containing the following fields:
        - policy: the policy from which to start training.
        - total_policies: A list of all policy.Policy strategies used for
          training, including the one for the current player. Either
          marginalized or joint strategies are accepted.
        - current_player: Integer representing the current player.
        - probabilities_of_playing_policies: A list of arrays representing, per
          player, the probabilities of playing each policy in total_policies for
          the same player.
      strategy_sampler: Callable that samples strategies from `total_policies`
        using `probabilities_of_playing_policies`. It only samples one joint
        "action" for all players. Implemented to be able to take into account
        joint probabilities of action.
      using_joint_strategies: Whether the meta-strategies sent are joint (True)
        or marginalized.
      **oracle_specific_execution_kwargs: Other set of arguments, for
        compatibility purposes. Can for example represent whether to Rectify
        Training or not.

    Returns:
      A list of list of OpenSpiel Policy objects representing the expected
      best response, following the same structure as training_parameters.
    """
    new_policies = []
    for player_parameters in training_parameters:
      player_policies = []
      for params in player_parameters:
        current_player = params['current_player']
        total_policies = params['total_policies']
        probabilities_of_playing_policies = params[
            'probabilities_of_playing_policies']
        if using_joint_strategies:
          aggr_policy = utils.aggregate_joint_policies(
              game, utils.marginal_to_joint(total_policies),
              probabilities_of_playing_policies.reshape(-1))
        else:
          aggr_policy = utils.aggregate_policies(
              game, total_policies, probabilities_of_playing_policies)

        # This takes as input an aggregate policy, and computes a best response
        # for current_player at the applicable information states by recursing
        # through the game tree. At information states involving other players
        # or chance, the aggr_policy is used to compute the expected value, such
        # that a best response for current_player can be computed.
        if self.best_response_backend == 'py':
          best_resp = best_response.BestResponsePolicy(game, current_player,
                                                       aggr_policy)
        else:
          self.best_response_processors[current_player].set_policy(
              policy_utils.policy_to_dict(aggr_policy, game, self.all_states,
                                          self.state_to_information_state))

          self.best_responders[current_player] = (
              best_response.CPPBestResponsePolicy(
                  game, current_player, aggr_policy, self.all_states,
                  self.state_to_information_state,
                  self.best_response_processors[current_player]))
          best_resp = self.best_responders[current_player]
        player_policies.append(best_resp)
      new_policies.append(player_policies)
    return new_policies
