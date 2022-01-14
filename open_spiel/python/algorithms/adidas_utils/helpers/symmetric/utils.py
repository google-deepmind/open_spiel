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
"""Symmetric payoff tensor utils."""

import itertools
import math

from absl import logging  # pylint:disable=unused-import

import numpy as np


def sym(pt):
  """Symmetrize stack of payoff tensors (stacked along first dimension).

  A payoff tensor can be `symmetrized' by averaging over all possible
  permutations of the players. This means permuting the axes corresponding to
  the player strategies as well as the payoffs assigned to the players. E.g.,
  player A playing strategy 1 and player B playing strategy 3 is no different
  from player A playing strategy 3 and player B playing strategy 1 in a
  symmetric game. Note we permuted the strategies, but we must also permute the
  payoffs.

  Args:
    pt: tensor of shape: (num_players,) + (num_strategies,) * num_players
  Returns:
    pt_sym: symmetrized payoff tensor of same shape
  """
  num_players = len(pt.shape[1:])
  num_perms = math.factorial(num_players)
  pt_sym = np.zeros_like(pt)
  logging.info('Symmetrizing over {:d} permutations...'.format(num_perms))
  for i, perm_players in enumerate(itertools.permutations(range(num_players))):
    if (i % (num_perms // 5)) == 0:
      logging.info('\t{:d} / {:d}'.format(i, num_perms))
    perm_axes = tuple([pi + 1 for pi in perm_players])
    permuted_tensor = np.transpose(pt, (0,) + perm_axes)[list(perm_players)]
    pt_sym += permuted_tensor / float(num_perms)
  logging.info('\t{total:d} / {total:d}'.format(total=num_perms))
  return pt_sym
