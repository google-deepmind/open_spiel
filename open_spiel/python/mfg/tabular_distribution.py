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
"""A tabular representation of a distribution for a game."""

from typing import Dict, Optional

from open_spiel.python.mfg import distribution
import pyspiel

DistributionDict = Dict[str, float]


class TabularDistribution(distribution.ParametricDistribution):
  """Distribution that uses a dictionary to store the values of the states."""

  def __init__(self, game: pyspiel.Game):
    self._distribution: DistributionDict = {}
    super().__init__(game)

  def value(self, state: pyspiel.State) -> float:
    return self.value_str(self.state_to_str(state))

  def value_str(self,
                state_str: str,
                default_value: Optional[float] = None) -> float:
    """Returns the probability of the distribution on the state string given.

    Args:
      state_str: A string.
      default_value: If not None, return this value if the state is not in the
        support of the distribution.

    Returns:
      A `float`.

    Raises:
      ValueError: If the state has not been seen by the distribution and no
        default value has been passed to the method.
    """
    if default_value is None:
      try:
        return self._distribution[state_str]
      except KeyError as e:
        raise ValueError(
            f"Distribution not computed for state {state_str}") from e
    return self._distribution.get(state_str, default_value)

  def get_params(self) -> DistributionDict:
    return self._distribution

  def set_params(self, params: DistributionDict):
    self._distribution = params

  def state_to_str(self, state: pyspiel.State) -> str:
    # TODO(author15): Consider switching to
    # state.mean_field_population(). For now, this does not matter in
    # practice since games don't have different observation strings for
    # different player IDs.
    return state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)

  @property
  def distribution(self) -> DistributionDict:
    return self._distribution
