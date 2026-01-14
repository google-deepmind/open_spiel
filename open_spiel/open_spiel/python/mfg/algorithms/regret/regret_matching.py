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

"""Regret Matching algorithm for MFGs."""

from typing import Optional

import numpy as np

from open_spiel.python.mfg.algorithms import utils
from open_spiel.python.mfg.algorithms.regret import regret_minimizer


def regret_matching(regrets):
  regrets = np.array(regrets)
  regret_plus = regrets * (regrets > 0.0)
  regrets_sum = np.sum(regret_plus, axis=-1)
  regret_plus[regrets_sum > 0.0, :] = regret_plus[
      regrets_sum > 0.0, :
  ] / regrets_sum[regrets_sum > 0.0].reshape(-1, 1)
  regret_plus[regrets_sum <= 0.0, :] = (
      np.ones_like(regret_plus[regrets_sum <= 0.0, :]) / regret_plus.shape[-1]
  )
  return regret_plus


class RegretMatching(regret_minimizer.RegretMinimizer):
  """Base class for Regret Minimizers.

  Implements base functions for regret minimizers to implement.

  Attributes:
    _game: Pyspiel game.
    _regret_steps_per_step: Number of regret steps per `step` call (Maximum
      number in case `stop_early` is true)
    _rho_tol: If `_compress_nus` is true, minimum probability threshold (
      Probabilities below `rho_tol` will be filtered out).
    _compress_nus: Whether to compress nus (Remove nus with low selection
      probability) or not.
    _compress_lbd: Penalty term in L1 minimization when compressing nus.
    _stop_early: Whether to stop regret computation when average regret is lower
      than `_stop_regret_threshold` or to keep going until
      `_regret_steps_per_step` steps have been accomplished.
    _stop_regret_threshold: If `stop_early` is true, average regret threshold
      under which the algorithm will stop.
    _policies: List of Policies
    _value_estimator: Value estimation function.
    _value_estimation_n: Number of runs to average _value_estimator's result on.
  """

  def __init__(
      self,
      game,
      policies,
      eta: Optional[float] = None,
      regret_steps_per_step: int = 1,
      rho_tol: float = 1e-4,
      compress_nus: bool = True,
      compress_lbd: float = 0.0,
      compress_every: int = 1,
      stop_early: bool = True,
      stop_regret_threshold: float = 1e-3,
      value_estimator=utils.sample_value,
      value_estimation_n: int = 1,
      compute_internal_regret: bool = False,
  ):
    super().__init__(
        game,
        policies,
        regret_steps_per_step=regret_steps_per_step,
        rho_tol=rho_tol,
        compress_nus=compress_nus,
        compress_every=compress_every,
        compress_lbd=compress_lbd,
        stop_early=stop_early,
        stop_regret_threshold=stop_regret_threshold,
        value_estimator=value_estimator,
        value_estimation_n=value_estimation_n,
        compute_internal_regret=compute_internal_regret,
    )

    if self._compute_internal_regret:
      self._regrets = np.zeros((len(policies), len(policies)))
    else:
      self._regrets = np.zeros(len(policies))
    self._p = np.ones(len(policies)) / (1.0 * len(policies))

  def get_all_action_regrets(self):
    assert self._compute_internal_regret
    return [
        regret_matching(np.sum(action_regret, axis=0))
        for action_regret in self._regrets
    ]

  def compute_last_regret(self, nu, reward):
    reward = np.array(reward)
    if self._compute_internal_regret:
      weighted_rewards = nu.reshape(-1, 1) * reward.reshape(1, -1)
      on_policy_values = np.sum(
          regret_matching(self._regrets) * weighted_rewards,
          axis=-1,
          keepdims=True,
      )
      return weighted_rewards - on_policy_values
    else:
      on_policy_value = np.sum(np.array(nu) * np.array(reward))
      return reward - on_policy_value

  def update_regret(self, nu, reward):
    self._regrets += self.compute_last_regret(nu, reward)

  def get_all_w_nus(self):
    assert self._compute_internal_regret
    return regret_matching(self._regrets)

  def get_nu(self):
    if self._compute_internal_regret:
      return np.sum(
          self._p.reshape(-1, 1) * regret_matching(self._regrets), axis=0
      )
    else:
      return regret_matching(self._regrets)

  def step(self, welfare_bonus=0.0):
    rewards = np.zeros(len(self._policies))
    nu = self.get_nu()
    assert np.all(nu >= 0.0) and (np.abs(np.sum(nu) - 1) < 1e-8)
    self._nus.append(nu)
    self._nu_weights.append(1.0)

    mu = utils.MixedDistribution(self._policy_mus, nu)
    for _ in range(self._value_estimation_n):
      for index, policy in enumerate(self._policies):
        rewards[index] += self._value_estimator(policy, mu, self._game)
    rewards /= self._value_estimation_n

    welfare = np.sum(np.array(rewards) * np.array(nu))

    self._rewards.append(rewards + welfare_bonus * welfare * nu)
    self._true_rewards.append(rewards)

    self.update_regret(nu, rewards + welfare_bonus * welfare * nu)
    if self._compute_internal_regret:
      self.compute_p()

  def reset(self, policies):
    """Restart the bandit with new policies."""
    self._p = np.ones(len(policies)) / (1.0 * len(policies))
    self._policies = policies
    self._nus = []
    self._rewards = []
    self._true_rewards = []
    if self._compute_internal_regret:
      self._regrets = np.zeros((len(policies), len(policies)))
    else:
      self._regrets = np.zeros(len(policies))
    self._policy_mus = []
    self._nu_weights = []
    self.update_policy_mus()
