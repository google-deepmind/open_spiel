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

"""Utility functions for ResponseGraphUCB."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.egt import utils as egt_utils
import pyspiel
from open_spiel.python.utils import file_utils


def get_method_tuple_acronym(method_tuple):
  """Returns pretty acronym for specified ResponseGraphUCB method tuple."""
  if isinstance(method_tuple, tuple):
    acronyms = [get_method_acronym(m) for m in method_tuple]
    return ', '.join(acronyms)
  else:
    return get_method_acronym(method_tuple)


def get_method_tuple_linespecs(method):
  """Gets plot linespecs for the specified ResponseGraphUCB method."""
  sampling_strats = [
      'uniform-exhaustive', 'uniform', 'valence-weighted', 'count-weighted'
  ]
  conf_methods = ['ucb-standard', 'clopper-pearson-ucb']
  method_to_id_map = dict(
      (m, i)
      for i, m in enumerate(itertools.product(sampling_strats, conf_methods)))

  # Create palette
  num_colors = len(method_to_id_map.keys())
  colors = plt.get_cmap('Set1', num_colors).colors

  # Spec out the linestyle
  base_method = (method[0], method[1].replace('-relaxed', '')
                )  # Method name without -relaxed suffix
  linespecs = {
      'color': colors[method_to_id_map[base_method]]
  }  # Use base method for color (ignoring relaxed vs non-relaxed)
  if 'relaxed' in method[1]:  # Use actual method for linestyle
    linespecs['linestyle'] = 'dashed'
  else:
    linespecs['linestyle'] = 'solid'

  return linespecs


def get_method_acronym(method):
  """Gets pretty acronym for specified ResponseGraphUCB method."""
  if method == 'uniform-exhaustive':
    return r'$\mathcal{S}$: UE'
  elif method == 'uniform':
    return r'$\mathcal{S}$: U'
  elif method == 'valence-weighted':
    return r'$\mathcal{S}$: VW'
  elif method == 'count-weighted':
    return r'$\mathcal{S}$: CW'
  elif method == 'ucb-standard':
    return r'$\mathcal{C}(\delta)$: UCB'
  elif method == 'ucb-standard-relaxed':
    return r'$\mathcal{C}(\delta)$: R-UCB'
  elif method == 'clopper-pearson-ucb':
    return r'$\mathcal{C}(\delta)$: CP-UCB'
  elif method == 'clopper-pearson-ucb-relaxed':
    return r'$\mathcal{C}(\delta)$: R-CP-UCB'
  elif method == 'fixedbudget-uniform':
    return r'$\mathcal{S}$: U, $\mathcal{C}(\delta)$: FB'
  else:
    raise ValueError('Unknown sampler method: {}!'.format(method))


def digraph_edge_hamming_dist(g1, g2):
  """Returns number of directed edge mismatches between digraphs g1 and g2."""
  dist = 0
  for e1 in g1.edges:
    if e1 not in g2.edges:
      dist += 1
  return dist


class BernoulliGameSampler(object):
  """A sampler for a game with Bernoulli-distributed payoffs."""

  def __init__(self, strategy_spaces, means, payoff_bounds):
    """Initializes the Bernoulli game sampler.

    Payoffs are automatically scaled to lie between 0 and 1.

    Args:
      strategy_spaces: a list of sizes of player strategy spaces.
      means: 1+num_players dimensional array of mean payoffs.
      payoff_bounds: min/max observable value of payoffs, necessary since one
        may seek Bernoulli-sampling for games with different payoff ranges.
    """
    self.strategy_spaces = strategy_spaces
    self.n_players = len(strategy_spaces)
    self.raw_means = means
    self.payoff_bounds = payoff_bounds
    self.means = self.rescale_payoff(means)

    # Specific to the Bernoulli case. Probas in [0,1], proportional to payoffs
    self.p_max = self.means

  def rescale_payoff(self, payoff):
    """Rescales payoffs to be in [0,1]."""
    # Assumes payoffs are bounded between [-payoff_bound, payoff_bound]
    return (payoff - self.payoff_bounds[0]) / (
        self.payoff_bounds[1] - self.payoff_bounds[0])

  def observe_result(self, strat_profile):
    """Returns empirical payoffs for each agent."""
    outcomes = np.zeros(self.n_players)
    for k in range(self.n_players):
      # compute Bernoulli probabilities
      outcomes[k] = np.random.choice(
          [1, 0],
          p=[self.p_max[k][strat_profile], 1. - self.p_max[k][strat_profile]])
    return outcomes


class ZeroSumBernoulliGameSampler(BernoulliGameSampler):
  """A sampler for a zero-sum game with Bernoulli-distributed payoffs."""

  def __init__(self, strategy_spaces, means, payoff_bounds):
    super(ZeroSumBernoulliGameSampler, self).__init__(strategy_spaces, means,
                                                      payoff_bounds)
    # Verify the game is zero-sum
    assert np.allclose(np.sum(self.means, axis=0), 1.)

  def observe_result(self, strat_profile):
    outcomes = np.zeros(self.n_players)
    win_ix = np.random.choice(
        self.n_players, p=self.means[(slice(None),) + strat_profile])
    outcomes[win_ix] = 1.
    return outcomes


def get_payoffs_bernoulli_game(size=(2, 2, 2)):
  """Gets randomly-generated zero-sum symmetric two-player game."""
  too_close = True
  while too_close:
    M = np.random.uniform(-1, 1, size=size)  # pylint: disable=invalid-name
    M[0, :, :] = 0.5 * (M[0, :, :] - M[0, :, :].T)
    M[1, :, :] = -M[0, :, :]
    if np.abs(M[0, 0, 1]) < 0.1:
      too_close = True
    else:
      too_close = False
  return M


def get_soccer_data():
  """Returns the payoffs and strategy labels for MuJoCo soccer experiments."""
  payoff_file = file_utils.find_file(
      'open_spiel/data/paper_data/response_graph_ucb/soccer.txt', 2)
  payoffs = np.loadtxt(payoff_file)
  return payoffs


def get_kuhn_poker_data(num_players=4, iterations=3):
  """Returns the kuhn poker data for the number of players specified."""
  game = pyspiel.load_game('kuhn_poker', {'players': num_players})
  xfp_solver = fictitious_play.XFPSolver(game, save_oracles=True)
  for _ in range(iterations):
    xfp_solver.iteration()

  # Results are seed-dependent, so show some interesting cases
  if num_players == 2:
    meta_games = xfp_solver.get_empirical_metagame(100, seed=1)
  elif num_players == 3:
    meta_games = xfp_solver.get_empirical_metagame(100, seed=5)
  elif num_players == 4:
    meta_games = xfp_solver.get_empirical_metagame(100, seed=2)

  # Metagame utility matrices for each player
  payoff_tables = []
  for i in range(num_players):
    payoff_tables.append(meta_games[i])
  return payoff_tables


def get_game_for_sampler(game_name):
  """Returns pre-processed game data for ResponseGraphUCB examples."""
  # pylint: disable=invalid-name
  if game_name == 'bernoulli':
    M = get_payoffs_bernoulli_game()
    strategy_spaces = [2, 2]
    G = ZeroSumBernoulliGameSampler(
        strategy_spaces, means=M, payoff_bounds=[-1., 1.])
  elif game_name == 'soccer':
    M = get_soccer_data()
    M = M * 2. - 1  # Convert to zero-sum
    strategy_spaces = np.shape(M)
    M = np.asarray([M, M.T])
    G = ZeroSumBernoulliGameSampler(strategy_spaces, means=M,
                                    payoff_bounds=[np.min(M), np.max(M)])
  elif game_name in ['kuhn_poker_2p', 'kuhn_poker_3p', 'kuhn_poker_4p']:
    if '2p' in game_name:
      num_players = 2
    elif '3p' in game_name:
      num_players = 3
    elif '4p' in game_name:
      num_players = 4
    M = get_kuhn_poker_data(num_players, iterations=2)  # pylint: disable=invalid-name
    strategy_spaces = egt_utils.get_num_strats_per_population(M, False)
    G = BernoulliGameSampler(
        strategy_spaces, means=M, payoff_bounds=[np.min(M), np.max(M)])
  else:
    raise ValueError('Game', game_name, 'not implemented!')
  # pylint: enable=invalid-name
  return G


def plot_timeseries(ax, id_ax, data, xticks, xlabel='', ylabel='',
                    label='', logx=False, logy=False, zorder=10,
                    linespecs=None):
  """Plots timeseries data with error bars."""
  if logx:
    ax[id_ax].set_xscale('log')
  if logy:
    ax[id_ax].set_yscale('log')
  if linespecs:
    kwargs = {'color': linespecs['color']}
  else:
    kwargs = {}

  # Seaborn's bootstrapped confidence intervals were used in the original paper
  se = scipy.stats.sem(data, axis=0)
  ax[id_ax].fill_between(xticks, data.mean(0)-se, data.mean(0)+se,
                         zorder=zorder, alpha=0.2, **kwargs)
  ax[id_ax].plot(xticks, data.mean(0), label=label, zorder=zorder, **kwargs)

  # There may be multiple lines on the current axis, some from previous calls to
  # plot_timeseries, so reference just the latest
  if linespecs:
    ax[id_ax].get_lines()[-1].set_dashes([5, 5])
    ax[id_ax].get_lines()[-1].set_linestyle(linespecs['linestyle'])

  ax[id_ax].set(xlabel=xlabel, ylabel=ylabel)
  ax[id_ax].set_axisbelow(True)
  ax[id_ax].grid(True)
  for _, spine in ax[id_ax].spines.items():
    spine.set_zorder(-1)
