# Copyright 2022 DeepMind Technologies Limited
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
"""Boltzmann Policy Iteration."""

from open_spiel.python import policy as policy_lib
from open_spiel.python.mfg.algorithms import mirror_descent


class BoltzmannPolicyIteration(mirror_descent.MirrorDescent):
  """Boltzmann Policy Iteration algorithm.

  In this algorithm, at each iteration, we update the policy by first computing
  the Q-function that evaluates the current policy, and then take a softmax.
  This corresponds to using Online Mirror Descent algorithm without summing
  Q-functions but simply taking the latest Q-function.
  """

  def get_projected_policy(self) -> policy_lib.Policy:
    """Returns the projected policy."""
    return mirror_descent.ProjectedPolicy(
        self._game,
        list(range(self._game.num_players())),
        self._state_value,
        coeff=self._lr)
