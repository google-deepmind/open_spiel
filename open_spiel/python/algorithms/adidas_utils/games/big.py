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
"""Big tensor games."""

from absl import logging  # pylint:disable=unused-import

import numpy as np

from open_spiel.python.algorithms.adidas_utils.helpers import misc


class TensorGame(object):
  """Tensor Game."""

  def __init__(self, pt, seed=None):
    """Ctor. Inits payoff tensor (players x actions x ... np.array).

    Args:
      pt: payoff tensor, np.array
      seed: seed for random number generator, used if computing best responses
    """
    if np.any(pt < 0.):
      raise ValueError("Payoff tensor must contain non-negative values")
    self.pt = pt

    self.seed = seed
    self.random = np.random.RandomState(seed)

  def num_players(self):
    return self.pt.shape[0]

  def num_strategies(self):
    return self.pt.shape[1:]

  def payoff_tensor(self):
    return self.pt

  def get_payoffs_for_strategies(self, policies):
    """Return vector of payoffs for all players given list of strategies.

    Args:
      policies: list of integers indexing strategies for each player
    Returns:
      np.array (length num players) of payoffs
    """
    return self.pt[tuple([slice(None)] + policies)]

  def best_response(self, mixed_strategy, return_exp=False):
    """Return best response and its superiority over the current strategy.

    Args:
      mixed_strategy: np.ndarray (distribution over strategies)
      return_exp: bool, whether to return how much best response exploits the
        given mixed strategy (default is False)
    Returns:
      br: int, index of strategy (ties split randomly)
      exp: u(br) - u(mixed_strategy)
    """
    logging.warn("Assumes symmetric game! Returns br for player 0.")
    gradient = misc.pt_reduce(self.pt[0],
                              [mixed_strategy] * self.num_players(),
                              [0])
    br = misc.argmax(self.random, gradient)
    exp = gradient.max() - gradient.dot(mixed_strategy)
    if return_exp:
      return br, exp
    else:
      return br

  def best_population_response(self, dist, policies):
    """Returns the best response to the current population of policies.

    Args:
      dist: np.ndarray, distribution over policies
      policies: list of integers indexing strategies for each player
    Returns:
      best response, exploitability tuple (see best_response)
    """
    ns = self.num_strategies()
    mixed_strat = np.zeros(ns)
    for pure_strat, prob in zip(policies, dist):
      mixed_strat[pure_strat] += prob
    return self.best_response(mixed_strat)


class ElFarol(TensorGame):
  """N-Player, 2-Action symmetric game with unique symmetric Nash."""

  def __init__(self, n=2, c=0.5, B=0, S=1, G=2, seed=None):
    """Ctor. Initializes payoff tensor (N x (2,) * N np.array).

    See Section 3.1, The El Farol Stage Game in
    http://www.econ.ed.ac.uk/papers/id186_esedps.pdf

    action 0: go to bar
    action 1: avoid bar

    Args:
      n: int, number of players
      c: float, threshold for `crowded' as a fraction of number of players
      B: float, payoff for going to a crowded bar
      S: float, payoff for staying at home
      G: float, payoff for going to an uncrowded bar
      seed: seed for random number generator, used if computing best responses
    """
    assert G > S > B, "Game parameters must satisfy G > S > B."
    pt = np.zeros((n,) + (2,) * n)
    for idx in np.ndindex(pt.shape):
      p = idx[0]
      a = idx[1:]
      a_i = a[p]
      go_to_bar = (a_i < 1)
      crowded = (n - 1 - sum(a) + a_i) >= (c * n)
      if go_to_bar and not crowded:
        pt[idx] = G
      elif go_to_bar and crowded:
        pt[idx] = B
      else:
        pt[idx] = S
    super().__init__(pt, seed)
