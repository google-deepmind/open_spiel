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

"""Polynomial Weights algorithm for MFGs."""

from typing import Optional
import numpy as np
from open_spiel.python.mfg.algorithms import utils
from open_spiel.python.mfg.algorithms.regret import regret_minimizer


def polynomial_weight_update(weights, rewards, eta):
  return weights * (1 + eta * rewards)


class PolynomialWeightAlgorithm(regret_minimizer.RegretMinimizer):
  """Implements the Polynomial Weight Algorithm Regret minimizer.

  This is an external-regret minimizer, adapted here to the Mean-Field,
  Partially-Observable case.

  References: Muller et al, https://arxiv.org/abs/2111.08350, and
    Blum et al, https://www.cs.cmu.edu/~avrim/ML10/regret-chapter.pdf
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
      self._ws = [np.ones(len(policies)) for _ in range(len(policies))]
      self._p = np.ones(len(policies)) / (1.0 * len(policies))
    else:
      self._w = np.ones(len(policies))

    if eta is None:
      assert regret_steps_per_step is not None, (
          "Both `eta` and "
          "`regret_steps_per_step` were "
          "None, whereas our algorithm "
          "requires either value to be "
          "set."
      )
      self.compute_optimal_eta()
    else:
      self._eta = eta

    self._compress_every = compress_every

  def get_all_w_nus(self):
    assert self._compute_internal_regret
    return [w / np.sum(w) for w in self._ws]

  def get_nu(self):
    if self._compute_internal_regret:
      return np.sum(
          self._p.reshape(-1, 1) * np.array(self.get_all_w_nus()), axis=0
      )
    else:
      return self._w / np.sum(self._w)

  def _update_weights(self, rewards):
    if self._compute_internal_regret:
      self._ws = [
          w * (1 + self._eta * rewards * p) for w, p in zip(self._ws, self._p)
      ]
      self.compute_p()
    else:
      self._w = self._w * (1 + self._eta * rewards)

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

    self._update_weights(rewards)

    welfare = np.sum(np.array(rewards) * np.array(nu))

    self._rewards.append(rewards + welfare_bonus * welfare * nu)
    self._true_rewards.append(rewards)

  def compute_optimal_eta(self):
    if self._regret_steps_per_step is not None:
      self._eta = min(
          np.sqrt(np.log(len(self._policies)) / self._regret_steps_per_step),
          0.5,
      )

  def reset(self, policies):
    if self._compute_internal_regret:
      self._ws = [np.ones(len(policies)) for _ in range(len(policies))]
      self._p = np.ones(len(policies)) / (1.0 * len(policies))
    else:
      self._w = np.ones(len(policies))
    self._policies = policies
    self._nus = []
    self._rewards = []
    self._true_rewards = []
    self._policy_mus = []
    self._nu_weights = []
    self.update_policy_mus()
    self.compute_optimal_eta()
