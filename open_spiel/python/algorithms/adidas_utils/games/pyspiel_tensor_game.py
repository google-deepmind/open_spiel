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
"""Wrapper for loading pyspiel games as payoff tensors."""

from absl import logging  # pylint:disable=unused-import

import numpy as np

from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel


class PyspielTensorGame(object):
  """Matrix Game."""

  def __init__(self, string_specifier='blotto(coins=10,fields=3,players=3)',
               tensor_game=False, seed=None):
    """Ctor. Inits payoff tensor (players x actions x ... np.array)."""
    self.pt = None
    self.string_specifier = string_specifier
    self.tensor_game = tensor_game

    if tensor_game:
      self.game = pyspiel.load_tensor_game(string_specifier)
    else:
      self.game = pyspiel.load_game(string_specifier)

    self.seed = seed  # currently unused

  def num_players(self):
    return self.game.num_players()

  def num_strategies(self):
    return [self.game.num_distinct_actions()] * self.num_players()

  def payoff_tensor(self):
    if self.pt is None:
      if not self.tensor_game:
        logging.info('reloading pyspiel game as tensor_game')
        self.game = pyspiel.load_tensor_game(self.string_specifier)
        self.tensor_game = True
      pt = np.asarray(game_payoffs_array(self.game))
      self.pt = pt - self.game.min_utility()
    return self.pt

  def get_payoffs_for_strategies(self, policies):
    """Return vector of payoffs for all players given list of strategies.

    Args:
      policies: list of integers indexing strategies for each player
    Returns:
      np.array (length num players) of payoffs
    """
    state = self.game.new_initial_state()
    state.apply_actions(policies)
    return np.asarray(state.returns()) - self.game.min_utility()
