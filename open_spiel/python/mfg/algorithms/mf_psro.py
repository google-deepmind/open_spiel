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

"""Mean-Field PSRO.

As implemented in Muller et al., 2021, https://arxiv.org/abs/2111.08350
"""

from open_spiel.python import policy as policy_std
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.mfg.algorithms import correlated_equilibrium
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy


def dict_equal(dic1, dic2):
  return all([dic1[a] == dic2[a] for a in dic1]) and all(
      [dic1[a] == dic2[a] for a in dic2]
  )


def equal_policies(pol1, pol2, all_states):
  assert isinstance(pol1, greedy_policy.GreedyPolicy)
  equal = True
  for state_key in all_states:
    state = all_states[state_key]
    try:
      equal = equal and dict_equal(pol1(state), pol2(state))
    except KeyError:
      equal = False
    except ValueError:
      continue
  return equal


def filter_policies(policies, new_policies, all_states):
  all_policies = policies
  no_novelty = True
  for new_policy in new_policies:
    if all([
        not equal_policies(new_policy, policy, all_states)
        for policy in all_policies
    ]):
      all_policies.append(new_policy)
      no_novelty = False
  return all_policies, no_novelty


class MeanFieldPSRO:
  """Mean-Field PSRO."""

  def __init__(
      self,
      game,
      regret_minimizer,
      regret_steps_per_step,
      best_responder=correlated_equilibrium.cce_br,
      filter_new_policies=False,
      increase_precision_when_done_early=False,
  ):
    self._game = game
    self._regret_minimizer = regret_minimizer
    self._regret_steps_per_step = regret_steps_per_step

    self._filter_new_policies = filter_new_policies
    self._increase_precision_when_done_early = (
        increase_precision_when_done_early
    )

    self._best_responder = best_responder

    self._nus = [[1.0]]
    self._policies = [policy_std.UniformRandomPolicy(self._game)]
    self._mus = [distribution.DistributionPolicy(game, self._policies[0])]
    self._weights = [1.0]

    self._all_states = None
    if self._filter_new_policies:
      self._all_states = get_all_states.get_all_states(game)

  def step(self):
    """Does a best-response step."""
    rewards = self._regret_minimizer.get_rewards()

    print("Computing best response.")
    new_policies, gap_value = self._best_responder(
        self._game, self._policies, self._weights, self._mus, self._nus, rewards
    )

    no_novelty = False
    if self._filter_new_policies:
      print("Filtering best responses")
      self._policies, no_novelty = filter_policies(
          self._policies, new_policies, self._all_states
      )
    else:
      self._policies = self._policies + new_policies

    if no_novelty:
      print("No new policy added, PSRO has terminated.")
      if self._increase_precision_when_done_early:
        print("Increasing precision")
        self._regret_minimizer.increase_precision_x_fold(2.0)
        self._regret_steps_per_step *= 2
        self._regret_minimizer.restart()
        self._regret_minimizer.step_for(self._regret_steps_per_step)
    else:
      print("Minimizing regret")
      self._regret_minimizer.reset(self._policies)
      self._regret_minimizer.step_for(self._regret_steps_per_step)

    average_regret = self._regret_minimizer.compute_average_regret()
    print("Average Regret : {}".format(average_regret))

    self._mus, self._weights = self._regret_minimizer.get_mus_and_weights()
    self._nus = self._regret_minimizer.get_nus()
    return average_regret, gap_value

  def get_equilibrium(self):
    return self._policies, self._nus, self._mus, self._weights
