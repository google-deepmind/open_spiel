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
"""GAMUT games.

See https://github.com/deepmind/open_spiel/tree/master/open_spiel/games/gamut
for details on how to build OpenSpiel with support for GAMUT.
"""

from absl import logging  # pylint:disable=unused-import

import numpy as np

from open_spiel.python.egt.utils import game_payoffs_array
import pyspiel


class GAMUT(object):
  """GAMUT Games."""

  def __init__(self, config_list, java_path='', seed=None):
    """Ctor. Inits payoff tensor (players x actions x ... np.array).

    Args:
      config_list: a list or strings alternating between gamut flags and values
        see http://gamut.stanford.edu/userdoc.pdf for more information
        e.g., config_list = ['-g', 'CovariantGame', '-players', '6',
                             '-normalize', '-min_payoff', '0',
                             '-max_payoff', '1', '-actions', '5', '-r', '0']
      java_path: string, java path
      seed: random seed, some GAMUT games are randomly generated
    """
    self.pt = None
    self.config_list = config_list

    self.seed = seed
    self.random = np.random.RandomState(seed)

    # parse interval for rho if supplied, e.g., '[-.2,1]'
    if '-r' in config_list:
      idx = next(i for i, s in enumerate(config_list) if s == '-r')
      val = config_list[idx + 1]
      if not val.isnumeric() and val[0] in '([' and val[-1] in ')]':
        a, b = val.strip('[]()').split(',')
        a = float(a)
        b = float(b)
        rho = self.random.rand() * (b - a) + a
        config_list[idx + 1] = str(rho)

    if isinstance(seed, int):
      self.config_list += ['-random_seed', str(seed)]
    self.java_path = java_path

    if java_path:
      generator = pyspiel.GamutGenerator(
          java_path,
          'gamut/gamut_main_deploy.jar')
    else:  # use default java path as specified by pyspiel
      generator = pyspiel.GamutGenerator(
          'gamut.jar')
    self.game = generator.generate_game(config_list)

  def num_players(self):
    return self.game.num_players()

  def num_strategies(self):
    return [self.game.num_distinct_actions()] * self.num_players()

  def payoff_tensor(self):
    if self.pt is None:
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
