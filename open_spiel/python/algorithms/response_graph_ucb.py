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

"""Implements ResponseGraphUCB algorithm from the below paper.

  "Multiagent Evaluation under Incomplete Information" (Rowland et al., 2019)
  See https://arxiv.org/abs/1909.09849 for details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import itertools
import operator
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats


class ResponseGraphUCB(object):
  """ResponseGraphUCB sampler class."""

  def __init__(self,
               game,
               exploration_strategy='uniform-exhaustive',
               confidence_method='ucb-standard',
               delta=0.01,
               ucb_eps=0,
               per_payoff_confidence=True,
               time_dependent_delta=False):
    """Initializes ResponseGraphUCB instance.

    Assumes that all payoffs fall in the interval [0,1].

    Args:
      game: an instance of the BernoulliGameSampler class.
      exploration_strategy: string specifying the exploration strategy.
      confidence_method: string specifying the confidence method.
      delta: float specifying the UCB delta parameter.
      ucb_eps: float specifying the UCB epsilon parameter.
      per_payoff_confidence: bool specifying whether confidence level applies
        on a per-payoff basis, or to all payoffs simultaneously.
      time_dependent_delta: bool specifying whether the confidence parameter
        varies with the number of interactions so that a union bound holds.
    """
    self.exploration_strategy = exploration_strategy
    self.confidence_method = confidence_method
    self.ucb_eps = ucb_eps
    self.G = game  # pylint: disable=invalid-name
    self.per_payoff_confidence = per_payoff_confidence
    self.time_dependent_delta = time_dependent_delta
    if self.per_payoff_confidence:
      self._delta = delta
    else:
      self._delta = delta / (
          self.G.n_players *
          functools.reduce(operator.mul, self.G.strategy_spaces, 1))

    # Compute the graph
    self.V = list(  # pylint: disable=invalid-name
        itertools.product(*[range(smax) for smax in self.G.strategy_spaces]))
    self.E = []  # pylint: disable=invalid-name
    for v in self.V:
      adj_strats = [
          list(range(v[k] + 1, self.G.strategy_spaces[k]))
          for k in range(self.G.n_players)
      ]
      for k in range(self.G.n_players):
        for new_s in adj_strats[k]:
          second_vertex = list(v)
          second_vertex[k] = new_s
          second_vertex = tuple(second_vertex)
          self.E.append((v, second_vertex))
    self.count_history = {v: [] for v in self.V}
    self.total_interactions = 0

  def delta(self, k, s):
    """Returns the confidence parameter for a given player and profile."""
    if not self.time_dependent_delta:
      return self._delta
    else:
      return self._delta * (6 / (np.pi**2 * self.count[k][s] **2))

  def initialise_mean_and_count(self):
    """Initializes means and counts for all response graph profiles."""
    self.mu = [
        np.zeros(tuple(self.G.strategy_spaces)) for _ in range(self.G.n_players)
    ]
    self.count = [
        np.zeros(tuple(self.G.strategy_spaces)) for _ in range(self.G.n_players)
    ]

  def update_mean_and_count(self, strat_profile, game_outcome):
    """Updates means and counts for strat_profile given game_outcome."""
    self.total_interactions += 1
    for k in range(self.G.n_players):
      self.mu[k][strat_profile] *= self.count[k][strat_profile]
      self.mu[k][strat_profile] += game_outcome[k]
      self.count[k][strat_profile] += 1
      self.mu[k][strat_profile] /= self.count[k][strat_profile]

    for s in self.V:
      self.count_history[s].append(self.count[0][s] /
                                   float(self.total_interactions))

  def _find_focal_coord(self, s1, s2):
    num_deviations = tuple(s1[l] != s2[l] for l in range(len(s1)))
    assert np.sum(num_deviations) == 1, ('Invalid profile pair s1, s2: ({},{}).'
                                         'Exactly one player should'
                                         'deviate!'.format(s1, s2))
    return np.argmax(num_deviations)

  def _initialise_queue_uniform(self):
    self.remaining_edges = copy.deepcopy(self.E)

  def _add_to_queue_uniform(self, edges_removed):
    """Adds edge to sampling queue using uniform sampling."""
    for e in edges_removed:
      self.remaining_edges.remove(e)
    self.profile_queue.append(
        random.choice(random.choice(self.remaining_edges)))

  def _initialise_queue_uniform_exhaustive(self):
    self.edge_order = copy.deepcopy(self.E)
    random.shuffle(self.edge_order)

  def _add_to_queue_uniform_exhaustive(self, edges_removed):
    """Adds edge to sampling queue using uniform-exhausitive sampling."""
    for e in edges_removed:
      self.edge_order.remove(e)
    self.profile_queue.append(random.choice(self.edge_order[0]))

  def _initialise_queue_valence_weighted(self):
    self.vertex_valences = {
        v: np.sum(self.G.strategy_spaces) - self.G.n_players for v in self.V
    }
    self.sum_valences = sum(self.vertex_valences.values())

  def _add_to_queue_valence_weighted(self, edges_removed):
    """Adds edge to sampling queue using valence-weighted sampling."""
    # Deal with removed edges
    for e in edges_removed:
      for s in e:
        self.vertex_valences[s] -= 1
        self.sum_valences -= 1

    # Calculate probabilities
    probs = np.array([self.vertex_valences[v]**2 for v in self.V])
    probs = probs / np.sum(probs)
    s_ix = np.random.choice(np.arange(len(self.V)), p=probs)
    self.profile_queue.append(self.V[s_ix])

  def _initialise_queue_count_weighted(self):
    # Keep track of which vertices have non-zero valence in graph
    self.vertex_valences = {
        v: np.sum(self.G.strategy_spaces) - self.G.n_players for v in self.V
    }
    self.sum_valences = sum(self.vertex_valences.values())

  def _add_to_queue_count_weighted(self, edges_removed):
    """Adds edge to sampling queue using count-weighted sampling."""
    # Update vertex valences
    for e in edges_removed:
      for s in e:
        self.vertex_valences[s] -= 1
        self.sum_valences -= 1
    # Check counts
    eligible_vertices = {
        v: self.count[0][v] for v in self.V if self.vertex_valences[v] != 0
    }
    strat = min(eligible_vertices, key=eligible_vertices.get)
    self.profile_queue.append(strat)

  def initialise_queue(self):
    """Initializes sampling queue."""
    self.edges_remaining = copy.deepcopy(self.E)
    if self.exploration_strategy == 'uniform':
      self._initialise_queue_uniform()
    elif self.exploration_strategy == 'uniform-exhaustive':
      self._initialise_queue_uniform_exhaustive()
    elif self.exploration_strategy == 'valence-weighted':
      self._initialise_queue_valence_weighted()
    elif self.exploration_strategy == 'count-weighted':
      self._initialise_queue_count_weighted()
    else:
      raise ValueError('Did not recognise exploration strategy: {}'.format(
          self.exploration_strategy))

    self.profile_queue = []

  def add_to_queue(self, removed):
    """Update the sampling queue and the list of resolved edges.

    Args:
      removed: the list of edges resolved in the previous round, which should be
        removed from the sampling list in subsequent rounds.
    """
    if self.exploration_strategy == 'uniform':
      self._add_to_queue_uniform(removed)
    elif self.exploration_strategy == 'uniform-exhaustive':
      self._add_to_queue_uniform_exhaustive(removed)
    elif self.exploration_strategy == 'valence-weighted':
      self._add_to_queue_valence_weighted(removed)
    elif self.exploration_strategy == 'count-weighted':
      self._add_to_queue_count_weighted(removed)
    else:
      raise ValueError('Did not recognise exploration strategy: {}'.format(
          self.exploration_strategy))

  def evaluate_strategy_profile(self, yield_outcomes=False):
    """Evaluates a strategy profile on the sampling queue.

    Specifically, this:
      1. Removes a strategy profile from the queue.
      2. Evaluates it.
      3. Updates internal statistics.
      4. Adjusts list of strategy profiles whose statistics have been updated
         since last confidence bound check.

    Args:
      yield_outcomes: set True to yield the outcomes as well.

    Yields:
      s: profile evaluated.
      game_outcome: outcomes (player payoffs) for profile s.
    """
    if self.profile_queue:
      s = self.profile_queue.pop(0)
      if s not in self.active_strategy_profiles:
        self.active_strategy_profiles.append(s)
      game_outcome = self.G.observe_result(s)
      if yield_outcomes:
        yield s, game_outcome
      self.update_mean_and_count(s, game_outcome)

  def _ucb_standard_factor(self, s, k):
    return np.sqrt(np.log(2 / self.delta(k, s)) / (2 * self.count[k][s]))

  def _bernoulli_upper(self, p, n, delta):
    """Returns upper confidence bound for proportion p successes of n trials.

    Uses exact Clopper-Pearson interval.

    Args:
      p: proportion of successes.
      n: number of trials.
      delta: confidence parameter.
    """
    if p > 1 - 1e-6:
      return 1.
    else:
      upper = scipy.stats.beta.ppf(1. - delta / 2, p * n + 1, n - p * n)
      return upper

  def _bernoulli_lower(self, p, n, delta):
    """Returns lower confidence bound for proportion p successes of n trials.

    Uses exact Clopper-Pearson interval.

    Args:
      p: proportion of successes.
      n: number of trials.
      delta: confidence parameter.
    """
    if p < 1e-6:
      return 0.
    else:
      lower = scipy.stats.beta.ppf(delta / 2, p * n, n - p * n + 1)
      return lower

  def _ucb(self, s, k):
    """Returns k-th player's payoff upper-confidence-bound given profile s."""
    if self.confidence_method == 'ucb-standard':
      ucb_factor = self._ucb_standard_factor(s, k)
      return self.mu[k][s] + ucb_factor
    elif self.confidence_method == 'ucb-standard-relaxed':
      ucb_factor = self._ucb_standard_factor(s, k) - self.ucb_eps
      return self.mu[k][s] + ucb_factor
    elif self.confidence_method == 'clopper-pearson-ucb':
      return self._bernoulli_upper(self.mu[k][s], self.count[k][s],
                                   self.delta(k, s))
    elif self.confidence_method == 'clopper-pearson-ucb-relaxed':
      return self._bernoulli_upper(self.mu[k][s], self.count[k][s],
                                   self.delta(k, s)) - self.ucb_eps
    else:
      raise ValueError('Did not recognise confidence method {}'.format(
          self.confidence_method))

  def _lcb(self, s, k):
    """Returns k-th player's payoff lower-confidence-bound given profile s."""
    if self.confidence_method == 'ucb-standard':
      ucb_factor = self._ucb_standard_factor(s, k)
      return self.mu[k][s] - ucb_factor
    elif self.confidence_method == 'ucb-standard-relaxed':
      ucb_factor = self._ucb_standard_factor(s, k) + self.ucb_eps
      return self.mu[k][s] - ucb_factor
    elif self.confidence_method == 'clopper-pearson-ucb':
      return self._bernoulli_lower(self.mu[k][s], self.count[k][s],
                                   self.delta(k, s))
    elif self.confidence_method == 'clopper-pearson-ucb-relaxed':
      return self._bernoulli_lower(self.mu[k][s], self.count[k][s],
                                   self.delta(k, s)) + self.ucb_eps
    else:
      raise ValueError('Did not recognise confidence method {}'.format(
          self.confidence_method))

  def ucb_check(self, e):
    """Conducts a UCB check on response graph edge e.

    Specifically, given edge e connecting two strategy profiles s1 and s2, this:
      1. Determines the dominating strategy.
      2. Checks whether the payoff_UCB(worse_strategy) is less than
        the payoff_LCB of the better strategy; if this is true, the confidence
        intervals are disjoint, and the edge e is considered 'resolved'.

    Args:
      e: response graph edge.

    Returns:
      A bool indicating whether the edge is resolved,
      and also a tuple specifying the worse and better strategies.
    """

    s1, s2 = e
    k = self._find_focal_coord(s1, s2)
    if self.mu[k][s1] > self.mu[k][s2]:
      better_strat = s1
      worse_strat = s2
    else:
      better_strat = s2
      worse_strat = s1

    ucb = self._ucb(worse_strat, k)
    lcb = self._lcb(better_strat, k)

    return (ucb < lcb), (worse_strat, better_strat)

  def check_confidence(self):
    """Returns the edges that are 'resolved' given a confidence bound check."""
    edges_to_check = []

    for e in self.edges_remaining:
      for s in self.active_strategy_profiles:
        if s in e:
          if e not in edges_to_check:
            edges_to_check.append(e)

    edges_removed = []
    for e in edges_to_check:
      removed, ordered_edge = self.ucb_check(e)
      if removed:
        edges_removed.append(e)
        self.edges_remaining.remove(e)
        self.directed_edges.append(ordered_edge)

    self.active_strategy_profiles = []

    return edges_removed

  def real_edge_direction(self, e):
    s1, s2 = e
    k = self._find_focal_coord(s1, s2)
    if self.G.means[k][s1] > self.G.means[k][s2]:
      return (s2, s1)
    else:
      return (s1, s2)

  def construct_real_graph(self):
    directed_edges = []
    for e in self.E:
      ordered_edge = self.real_edge_direction(e)
      directed_edges.append(ordered_edge)

    return self._construct_digraph(directed_edges)

  def compute_graph(self):
    for e in self.E:
      s1, s2 = e[0], e[1]
      k = self._find_focal_coord(s1, s2)
      if self.mu[k][s1] > self.mu[k][s2]:
        directed_edge = (s2, s1)
      else:
        directed_edge = (s1, s2)
      if directed_edge not in self.directed_edges:
        self.directed_edges.append(directed_edge)

  def forced_exploration(self):
    for v in self.V:
      game_outcome = self.G.observe_result(v)
      self.update_mean_and_count(v, game_outcome)

  def run(self, verbose=True, max_total_iterations=50000):
    """Runs the ResponseGraphUCB algorithm."""
    self.verbose = verbose

    # Upper bounds on number of evaluations
    self.max_total_iterations = max_total_iterations

    self.initialise_mean_and_count()
    self.directed_edges = []
    self.active_strategy_profiles = []
    self.initialise_queue()

    # Forced initial exploration
    self.forced_exploration()

    # Keep evaluating nodes until check method declares that we're finished
    iterations = 0
    edges_resolved_this_round = []
    while self.total_interactions < max_total_iterations:
      # Add nodes to queue
      self.add_to_queue(removed=edges_resolved_this_round)

      # Evaluate the nodes and log results
      for v, _ in self.evaluate_strategy_profile():
        if verbose:
          print(v)

      # Recompute confidence bounds, eliminate, stop etc.
      edges_resolved_this_round = self.check_confidence()

      if not self.edges_remaining:
        break
      iterations += 1

    # Fill in missing edges if max iters reached without resolving all edges
    self.compute_graph()

    # Compute objects to be returned
    if verbose:
      total_steps = self.compute_total_steps()
      print('\nTotal steps taken = {}'.format(total_steps))
    results = {}
    results['interactions'] = int(np.sum(self.count[0]))
    graph = self._construct_digraph(self.directed_edges)
    results['graph'] = graph
    return results

  def compute_total_steps(self):
    return int(np.sum(self.count[0]))

  def _construct_digraph(self, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(self.V)
    for e in edges:
      graph.add_edge(e[0], e[1])
    return graph

  def _plot_errorbars_2x2x2(self, x, y, xerr, yerr, fmt):
    """Plots ResponseGraph with error bars for a 2-player 2x2 game."""

    # plt.errorbar does not accept list of colors, so plot twice
    for i_strat in [0, 1]:
      if xerr[i_strat] is None:
        plt.errorbar(
            x=x[i_strat],
            y=y[i_strat],
            yerr=np.reshape(yerr[:, i_strat], (2, 1)),
            markerfacecolor='b',
            ecolor='b',
            fmt=fmt,
            zorder=0)
      elif yerr[i_strat] is None:
        plt.errorbar(
            x=x[i_strat],
            y=y[i_strat],
            xerr=np.reshape(xerr[:, i_strat], (2, 1)),
            markerfacecolor='b',
            ecolor='b',
            fmt=fmt,
            zorder=0)
      else:
        raise ValueError()

  def visualise_2x2x2(self, real_values, graph):
    """Plots summary of ResponseGraphUCB for a 2-player 2x2 game."""
    _, axes = plt.subplots(3, 3, figsize=(10, 10),
                           gridspec_kw={'width_ratios': [1, 2, 1],
                                        'height_ratios': [1, 2, 1]})
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')
    axes[2, 0].axis('off')
    axes[2, 2].axis('off')

    # (0,0) vs. (0,1)
    plt.sca(axes[0, 1])
    s1 = (0, 0)
    s2 = (0, 1)
    self._plot_errorbars_2x2x2(
        x=[0, 1],
        y=[self.mu[1][s1], self.mu[1][s2]],
        xerr=[None, None],
        yerr=np.array([[self.mu[1][s1] - self._lcb(s1, 1),
                        self.mu[1][s2] - self._lcb(s2, 1)],
                       [self._ucb(s1, 1) - self.mu[1][s1],
                        self._ucb(s2, 1) - self.mu[1][s2]]]),
        fmt='o')
    plt.scatter([0, 1], [real_values[1, 0, 0], real_values[1, 0, 1]],
                color='red',
                zorder=1)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.xticks([])
    plt.yticks([0, 0.5, 1])
    plt.gca().set_yticklabels(['0', '', '1'])
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().grid(True)
    plt.ylim(0, 1)

    # (0,0) vs. (1,0)
    plt.sca(axes[1, 0])
    s1 = (1, 0)
    s2 = (0, 0)
    self._plot_errorbars_2x2x2(
        x=[self.mu[0][s1], self.mu[0][s2]],
        y=[0, 1],
        xerr=np.array([[self.mu[0][s1] - self._lcb(s1, 0),
                        self.mu[0][s2] - self._lcb(s2, 0)],
                       [self._ucb(s1, 0) - self.mu[0][s1],
                        self._ucb(s2, 0) - self.mu[0][s2]]]),
        yerr=[None, None],
        fmt='o')
    plt.scatter([real_values[0, 1, 0], real_values[0, 0, 0]], [0, 1],
                color='red',
                zorder=1)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.xticks([0, 0.5, 1])
    plt.gca().set_xticklabels(['0', '', '1'])
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().grid(True)
    plt.yticks([])
    plt.xlim(0, 1)

    # (0,1) vs. (1,1)
    plt.sca(axes[1, 2])
    s1 = (1, 1)
    s2 = (0, 1)
    self._plot_errorbars_2x2x2(
        x=[self.mu[0][s1], self.mu[0][s2]],
        y=[0, 1],
        xerr=np.array([[self.mu[0][s1] - self._lcb(s1, 0),
                        self.mu[0][s2] - self._lcb(s2, 0)],
                       [self._ucb(s1, 0) - self.mu[0][s1],
                        self._ucb(s2, 0) - self.mu[0][s2]]]),
        yerr=[None, None],
        fmt='o')
    plt.scatter([real_values[0, 1, 1], real_values[0, 0, 1]], [0, 1],
                color='red',
                zorder=1)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.xticks([0, 0.5, 1])
    plt.gca().set_xticklabels(['0', '', '1'])
    plt.gca().xaxis.set_ticks_position('top')
    plt.yticks([])
    plt.gca().grid(True)
    plt.xlim(0, 1)

    # (1,0) vs. (1,1)
    plt.sca(axes[2, 1])
    s1 = (1, 0)
    s2 = (1, 1)
    self._plot_errorbars_2x2x2(
        x=[0, 1],
        y=[self.mu[1][s1], self.mu[1][s2]],
        xerr=[None, None],
        yerr=np.array([[self.mu[1][s1] - self._lcb(s1, 1),
                        self.mu[1][s2] - self._lcb(s2, 1)],
                       [self._ucb(s1, 1) - self.mu[1][s1],
                        self._ucb(s2, 1) - self.mu[1][s2]]]),
        fmt='o')
    plt.scatter([0, 1], [real_values[1, 1, 0], real_values[1, 1, 1]],
                color='red',
                zorder=1)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.xticks([])
    plt.yticks([0, 0.5, 1])
    plt.gca().set_yticklabels(['0', '', '1'])
    plt.gca().yaxis.set_ticks_position('right')
    plt.gca().grid(True)
    plt.ylim(0, 1)
    self.plot_graph(graph, subplot=True, axes=axes)  # Chart in the middle

  def plot_graph(self, graph, subplot=False, axes=None):
    """Plots the response graph."""
    if subplot:
      plt.sca(axes[1, 1])
      axes[1, 1].axis('off')
    else:
      plt.figure(figsize=(5, 5))
    if len(graph.nodes) == 4:
      pos = {(0, 0): [0, 1], (0, 1): [1, 1], (1, 0): [0, 0], (1, 1): [1, 0]}
    else:
      pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(
        graph, pos, node_size=1800, node_color='w', edgecolors='k')
    nx.draw_networkx_edges(
        graph,
        pos,
        node_size=1800,
        edge_color='k',
        arrowstyle='->',
        arrowsize=10,
        width=3)
    nx.draw_networkx_labels(self.G, pos, {x: x for x in self.V}, font_size=14)

  def visualise_count_history(self, figsize=(5, 2)):
    """Plots the sampling count history for each strategy profile."""
    plt.figure(figsize=figsize)
    data = []
    labels = []
    for v in self.V:
      print(v)
      labels.append(v)
      data.append(self.count_history[v])
    pal = plt.get_cmap('Dark2').colors
    plt.stackplot(
        np.arange(1, self.total_interactions + 1),
        np.array(data),
        labels=labels,
        colors=pal)
    plt.ylim(top=1, bottom=0)
    plt.xlabel('Interactions')
    plt.ylabel('Proportions')

    # Shrink current axis
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.67, box.height])
    plt.xlim(1, self.total_interactions)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
