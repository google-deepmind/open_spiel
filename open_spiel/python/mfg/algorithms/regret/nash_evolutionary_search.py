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

"""Randomly searches for a Restricted Nash Equilibrium.

"""

from typing import Optional

import cma
import numpy as np
from open_spiel.python.mfg.algorithms import utils
from open_spiel.python.mfg.algorithms.regret import regret_minimizer


def softmax(x):
  e = np.exp(x - np.max(x))
  return e / np.sum(e, axis=-1, keepdims=True)


class NashCMAES(regret_minimizer.RegretMinimizer):
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

  def __init__(self,
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
               value_estimation_n: int = 1):
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
        value_estimation_n=value_estimation_n)
    self._nu = np.ones(len(policies)) / len(policies)
    self._exploitability = None

  def compute_exploitability(self, nu):
    mu = utils.MixedDistribution(self._policy_mus, nu)
    per_policy_reward = 0.0
    for _ in range(self._value_estimation_n):
      per_policy_reward += np.array(
          [self._value_estimator(pi, mu, self._game) for pi in self._policies])
    per_policy_reward /= self._value_estimation_n
    on_policy_reward = np.sum(per_policy_reward * nu)
    return np.max(per_policy_reward - on_policy_reward)

  def step_for(self, T):  # pylint: disable=invalid-name
    self.step(T)

  def get_exploitabilities(self, nus):
    return np.array([self.compute_exploitability(nu) for nu in nus])

  def step(self, T):  # pylint: disable=invalid-name
    best_nu = np.ones(len(self._policies)) / len(self._policies)
    nu = best_nu
    n = 0
    best_exploitability = self.compute_exploitability(nu)
    exploitability = best_exploitability

    optimizer = cma.CMAEvolutionStrategy(x0=nu, sigma0=1.0)

    while best_exploitability > self._rho_tol and n < max(
        T, self._regret_steps_per_step):
      n += 1

      logit_nus = optimizer.ask()
      nus = softmax(logit_nus)
      exploitabilities = self.get_exploitabilities(nus)
      optimizer.tell(logit_nus, exploitabilities)

      best_new_exploitability = np.min(exploitabilities[0])
      if best_new_exploitability < best_exploitability:
        best_exploitability = best_new_exploitability
        best_nu = nus[np.argmin(exploitabilities)]
        print(best_exploitability)

    self._nus = [best_nu]
    self._nu_weights = [1.0]
    self._exploitability = exploitability

  def compute_average_regret(self):
    return self._exploitability

  def reset(self, policies):
    """Restart the bandit with new policies."""
    self._policies = policies
    self._policy_mus = []
    self._nu_weights = []
    self._exploitability = None
    self.update_policy_mus()
