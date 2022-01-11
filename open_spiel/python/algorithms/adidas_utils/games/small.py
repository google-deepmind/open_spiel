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
"""Small matrix games."""

from absl import logging  # pylint:disable=unused-import

import numpy as np

from open_spiel.python.algorithms.adidas_utils.helpers import misc


class MatrixGame(object):
  """Matrix Game."""

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
    return self.pt[:, policies[0], policies[1]]

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
    gradient = self.pt[0].dot(mixed_strategy)
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


class BiasedGame(MatrixGame):
  """2-Player, 3-Action symmetric game with biased stochastic best responses."""

  def __init__(self, seed=None):
    """Ctor. Initializes payoff tensor (2 x 3 x 3 np.array).

    Args:
      seed: seed for random number generator, used if computing best responses
    """
    # pylint:disable=bad-whitespace
    pt_r = np.array([[0,  0,  0 ],
                     [1, -2,  .5],
                     [-2, 1,  -1]]) + 2.
    # pylint:enable=bad-whitespace
    pt_c = pt_r.T  # symmetric game
    pt = np.stack((pt_r, pt_c), axis=0).astype(float)
    pt /= pt.max()  # arbitrary design choice to upper bound entries to 1
    super().__init__(pt, seed)


class PrisonersDilemma(MatrixGame):
  """2-Player, 2-Action symmetric prisoner's dilemma."""

  def __init__(self, seed=None):
    """Ctor. Initializes payoff tensor (2 x 2 x 2 np.array).

    Args:
      seed: seed for random number generator, used if computing best responses
    """
    # pylint:disable=bad-whitespace
    pt_r = np.array([[-1, -3],
                     [0,  -2]])
    # pylint:enable=bad-whitespace
    # shift tensor to ensure positivity required for ATE
    pt_r -= pt_r.min()
    pt_c = pt_r.T  # symmetric game
    pt = np.stack((pt_r, pt_c), axis=0).astype(float)
    pt /= pt.max()  # arbitrary design choice to upper bound entries to 1
    super().__init__(pt, seed)


class RockPaperScissors(MatrixGame):
  """2-Player, 3-Action symmetric RPS."""

  def __init__(self, weights=None, seed=None):
    """Ctor. Initializes payoff tensor (2 x 3 x 3 np.array).

    Args:
      weights: list of weights (floats) for [rock, paper, scissors]
      seed: seed for random number generator, used if computing best responses
    """
    if weights is None:
      weights = np.ones(3)
    r, p, s = weights
    # pylint:disable=bad-whitespace
    pt_r = np.array([[0, -p,  r],
                     [p,  0, -s],
                     [-r, s,  0]])
    # pylint:enable=bad-whitespace
    # shift tensor to ensure positivity required for ATE
    pt_r -= pt_r.min()
    pt_c = pt_r.T  # symmetric game
    pt = np.stack((pt_r, pt_c), axis=0).astype(float)
    super().__init__(pt, seed)


class SpiralGame(MatrixGame):
  """2-Player, 3-Action symmetric game with spiral dynamics on simplex."""

  def __init__(self, center=None, seed=None):
    """Ctor. Initializes payoff tensor (2 x 3 x 3 np.array).

    Args:
      center: center of cycle given in [x, y, z] Euclidean coordinates
      seed: seed for random number generator, used if computing best responses
    """
    if center is None:
      center = np.ones(3) / 3.
    else:
      if not ((np.sum(center) <= 1 + 1e-8) and np.all(center >= -1e-8)):
        raise ValueError("center must lie on simplex")
    self.center = center
    center = center.reshape((3, 1))

    # define coordinate frame for simplex; basis vectors on columns of transform
    transform = np.array([[.5, -.5, 0], [-.5, -.5, 1], [1, 1, 1]]).T
    transform /= np.linalg.norm(transform, axis=0)
    transform_inv = np.linalg.inv(transform)

    # canonical cycle matrix in 2-d
    cycle = 0.1 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

    # payoff tensor maps euclidean to simplex frame, applies cycle, maps back
    pt_r = transform.dot(cycle.dot(transform_inv))
    # subtracting off a column vector effectively offsets the vector field
    # because [[c c c], ...] [[x], [y], [z]] = [c * (x + y + z), ...] = [c, ...]
    pt_r -= pt_r.dot(center)
    # shift tensor to ensure positivity required for ATE
    if pt_r.min() < 0:
      pt_r -= pt_r.min()

    pt_c = pt_r.T  # symmetric game
    pt = np.stack((pt_r, pt_c), axis=0).astype(float)
    super().__init__(pt, seed)


class MatchingPennies(MatrixGame):
  """2-Player, 2-Action non-symmetric matching pennies."""

  def __init__(self, bias=1., seed=None):
    """Ctor. Initializes payoff tensor (2 x 2 x 2 np.array).

    Args:
      bias: float, rewards one action (bias) more than the other (1)
      seed: seed for random number generator, used if computing best responses
    """
    # pylint:disable=bad-whitespace
    pt_r = np.array([[1,  -1],
                     [-1, bias]])
    # pylint:enable=bad-whitespace
    pt_c = (-pt_r).T  # zero-sum game
    pt = np.stack((pt_r, pt_c), axis=0).astype(float)
    # shift tensor to ensure positivity required for ATE
    pt -= pt.min()
    pt /= pt.max()  # arbitrary design choice to upper bound entries to 1
    super().__init__(pt, seed)


class Shapleys(MatrixGame):
  """2-Player, 3-Action non-symmetric Shapleys game."""

  def __init__(self, beta=1., seed=None):
    """Ctor. Initializes payoff tensor (2 x 2 x 2 np.array).

    See Eqn 4 in https://arxiv.org/pdf/1308.4049.pdf.

    Args:
      beta: float, modifies the game so that the utilities @ Nash are now
        u_1(Nash) = (1 + beta) / 3 and u_2(Nash) = (1 - beta) / 3
        where Nash is the joint uniform distribution
      seed: seed for random number generator, used if computing best responses
    """
    # pylint:disable=bad-whitespace
    pt_r = np.array([[1,    0,    beta],
                     [beta, 1,    0],
                     [0,    beta, 1]])
    pt_c = np.array([[-beta,  1,    0],
                     [0,     -beta, 1],
                     [1,      0,   -beta]])
    # pylint:enable=bad-whitespace
    pt = np.stack((pt_r, pt_c), axis=0).astype(float)
    # shift tensor to ensure positivity required for ATE
    pt -= pt.min()
    pt /= pt.max()  # arbitrary design choice to upper bound entries to 1
    super().__init__(pt, seed)
