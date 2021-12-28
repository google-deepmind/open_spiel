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
"""Class of Optimization Oracles generating best response against opponents.

Oracles are as defined in (Lanctot et Al., 2017,
https://arxiv.org/pdf/1711.00832.pdf ), functions generating a best response
against a probabilistic mixture of opponents. This class implements the abstract
class of oracles, and a simple oracle using Evolutionary Strategy as
optimization method.
"""

import numpy as np


def strategy_sampler_fun(total_policies, probabilities_of_playing_policies):
  """Samples strategies according to distribution over them.

  Args:
    total_policies: List of lists of policies for each player.
    probabilities_of_playing_policies: List of numpy arrays representing the
      probability of playing a strategy.

  Returns:
    One sampled joint strategy.
  """
  policies_selected = []
  for k in range(len(total_policies)):
    selected_opponent = np.random.choice(
        total_policies[k],
        1,
        p=probabilities_of_playing_policies[k]).reshape(-1)[0]
    policies_selected.append(selected_opponent)
  return policies_selected


class AbstractOracle(object):
  """The abstract class representing oracles, a hidden optimization process."""

  def __init__(self,
               number_policies_sampled=100,
               **oracle_specific_kwargs):
    """Initialization method for oracle.

    Args:
      number_policies_sampled: Number of different opponent policies sampled
        during evaluation of policy.
      **oracle_specific_kwargs: Oracle specific args, compatibility
        purpose. Since oracles can vary so much in their implementation, no
        specific argument constraint is put on this function.
    """
    self._number_policies_sampled = number_policies_sampled
    self._kwargs = oracle_specific_kwargs

  def set_iteration_numbers(self, number_policies_sampled):
    """Changes the number of iterations used for computing episode returns.

    Args:
      number_policies_sampled: Number of different opponent policies sampled
        during evaluation of policy.
    """
    self._number_policies_sampled = number_policies_sampled

  def __call__(self, game, policy, total_policies, current_player,
               probabilities_of_playing_policies,
               **oracle_specific_execution_kwargs):
    """Call method for oracle, returns best response against a set of policies.

    Args:
      game: The game on which the optimization process takes place.
      policy: The current policy, in policy.Policy, from which we wish to start
        optimizing.
      total_policies: A list of all policy.Policy strategies used for training,
        including the one for the current player.
      current_player: Integer representing the current player.
      probabilities_of_playing_policies: A list of arrays representing, per
        player, the probabilities of playing each policy in total_policies for
        the same player.
      **oracle_specific_execution_kwargs: Other set of arguments, for
        compatibility purposes. Can for example represent whether to Rectify
        Training or not.
    """
    raise NotImplementedError("Calling Abstract class method.")

  def sample_episode(self, game, policies_selected):
    raise NotImplementedError("Calling Abstract class method.")

  def evaluate_policy(self, game, pol, total_policies, current_player,
                      probabilities_of_playing_policies,
                      strategy_sampler=strategy_sampler_fun,
                      **oracle_specific_execution_kwargs):
    """Evaluates a specific policy against a nash mixture of policies.

    Args:
      game: The game on which the optimization process takes place.
      pol: The current policy, in policy.Policy, from which we wish to start
        optimizing.
      total_policies: A list of all policy.Policy strategies used for training,
        including the one for the current player.
      current_player: Integer representing the current player.
      probabilities_of_playing_policies: A list of arrays representing, per
        player, the probabilities of playing each policy in total_policies for
        the same player.
      strategy_sampler: callable sampling strategy.
      **oracle_specific_execution_kwargs: Other set of arguments, for
        compatibility purposes. Can for example represent whether to Rectify
        Training or not.

    Returns:
      Average return for policy when played against policies_played_against.
    """
    del oracle_specific_execution_kwargs  # Unused.

    totals = 0
    count = 0
    for _ in range(self._number_policies_sampled):
      policies_selected = strategy_sampler(total_policies,
                                           probabilities_of_playing_policies)
      policies_selected[current_player] = pol

      new_return = self.sample_episode(
          game,
          policies_selected)[current_player]

      totals += new_return
      count += 1

    # Avoid the 0 / 0 case.
    return totals / max(1, count)

