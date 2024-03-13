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

"""Hedge algorithm for MFGs."""

from typing import Optional

import numpy as np

from open_spiel.python.mfg.algorithms import utils
from open_spiel.python.mfg.algorithms.regret import polynomial_weights


class Hedge(polynomial_weights.PolynomialWeightAlgorithm):
  """Hedge algorithm."""

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
        eta=eta,
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
      self._constant_eta = 1.0
    else:
      self._eta = eta
      self._constant_eta = eta

  def _update_weights(self, rewards):
    if self._compute_internal_regret:
      self._ws = [
          w * np.exp(self._eta * rewards * p) for w, p in zip(self._ws, self._p)
      ]
      self.compute_p()
    else:
      self._w = self._w * np.exp(self._eta * rewards)
