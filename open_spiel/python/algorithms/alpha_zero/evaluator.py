# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An MCTS Evaluator for an AlphaZero model."""

import functools

import numpy as np

from open_spiel.python.algorithms import mcts
import pyspiel


class AlphaZeroEvaluator(mcts.Evaluator):
  """An AlphaZero MCTS Evaluator."""

  def __init__(self, game, model):
    """An AlphaZero MCTS Evaluator."""
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
      raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("Game must have sequential turns.")
    if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
      raise ValueError("Game must be deterministic.")

    self._model = model

  def cache_info(self):
    return self.value_and_prior.cache_info()  # pylint: disable=no-value-for-parameter

  def clear_cache(self):
    self.value_and_prior.cache_clear()

  @functools.lru_cache(maxsize=2**12)
  def value_and_prior(self, state):
    # Make a singleton batch
    obs = np.expand_dims(state.observation_tensor(), 0)
    mask = np.expand_dims(state.legal_actions_mask(), 0)
    value, policy = self._model.inference(obs, mask)
    return value[0, 0], policy[0]  # Unpack batch

  def evaluate(self, state):
    """Returns a value for the given state."""
    value, _ = self.value_and_prior(state)
    return np.array([value, -value])

  def prior(self, state):
    """Returns the probabilities for all actions."""
    _, policy = self.value_and_prior(state)
    return [(action, policy[action]) for action in state.legal_actions()]
