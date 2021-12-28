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

"""Various visualization tools for Alpha-Rank.

All equations and variable names correspond to the following paper:
  https://arxiv.org/abs/1903.01373

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

try:
  import matplotlib.patches as patches  # pylint: disable=g-import-not-at-top
  import matplotlib.patheffects as PathEffects  # pylint: disable=g-import-not-at-top
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
except ImportError as e:
  logging.info("If your tests failed with the error 'ImportError: No module "
               "named functools_lru_cache', this is a known bug in matplotlib "
               "and there is a workaround (run sudo apt install "
               "python-backports.functools-lru-cache. See: "
               "https://github.com/matplotlib/matplotlib/issues/9344.")
  raise ImportError(str(e))

import networkx as nx  # pylint: disable=g-import-not-at-top
import numpy as np

from open_spiel.python.egt import utils


class NetworkPlot(object):
  """A class for visualizing the Alpha-Rank interaction network."""

  def __init__(self,
               payoff_tables,
               rhos,
               rho_m,
               pi,
               state_labels,
               num_top_profiles=None):
    """Initializes a network plotting object.

    Args:
      payoff_tables: List of game payoff tables, one for each agent identity.
        Each payoff_table may be either a 2D numpy array, or a
        _PayoffTableInterface object.
      rhos: Fixation probabilities.
      rho_m: Neutral fixation probability.
      pi: Stationary distribution of fixation Markov chain defined by rhos.
      state_labels: Labels corresponding to Markov states. For the
        single-population case, state_labels should be a list of pure strategy
        names. For the multi-population case, it
                    should be a dict with (key,value) pairs: (population
                      index,list of strategy names)
      num_top_profiles: Set to (int) to show only the graph nodes corresponding
        to the top k elements of stationary distribution, or None to show all.
    """
    self.fig = plt.figure(figsize=(10, 10))
    self.num_populations = len(payoff_tables)
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
    self.num_strats_per_population = (
        utils.get_num_strats_per_population(payoff_tables,
                                            payoffs_are_hpt_format))
    self.rhos = rhos
    self.rho_m = rho_m
    self.pi = pi
    self.num_profiles = len(pi)
    self.state_labels = state_labels
    self.first_run = True
    self.num_top_profiles = num_top_profiles

    if self.num_top_profiles:
      # More than total number of strats requested for plotting
      if self.num_top_profiles > self.num_profiles:
        self.num_top_profiles = self.num_profiles
      # Skip the bottom num_profiles-k stationary strategies.
      self.nodes_to_skip = list(self.pi.argsort()[:self.num_profiles -
                                                  self.num_top_profiles])
    else:
      self.nodes_to_skip = []

    self._reset_cycle_counter()

  def _reset_cycle_counter(self):
    self.i_cycle_to_show = -1

  def _draw_network(self):
    """Draws the NetworkX object representing the underlying graph."""
    plt.clf()

    if self.num_populations == 1:
      node_sizes = 5000
      node_border_width = 1.
    else:
      node_sizes = 15000
      node_border_width = 3.

    vmin, vmax = 0, np.max(self.pi) + 0.1

    nx.draw_networkx_nodes(
        self.g,
        self.pos,
        node_size=node_sizes,
        node_color=self.node_colors,
        edgecolors="k",
        cmap=plt.cm.Blues,
        vmin=vmin,
        vmax=vmax,
        linewidths=node_border_width)

    nx.draw_networkx_edges(
        self.g,
        self.pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        edge_color=self.edge_colors,
        edge_cmap=plt.cm.Blues,
        width=5)

    nx.draw_networkx_edge_labels(self.g, self.pos, edge_labels=self.edge_labels)

    if self.num_populations > 1:
      subnode_separation = 0.1
      subgraph = nx.Graph()
      for i_population in range(self.num_populations):
        subgraph.add_node(i_population)

    for i_strat_profile in self.g:
      x, y = self.pos[i_strat_profile]
      if self.num_populations == 1:
        node_text = "$\\pi_{" + self.state_labels[i_strat_profile] + "}=$"
        node_text += str(np.round(self.pi[i_strat_profile], decimals=2))
      else:
        node_text = ""  # No text for multi-population case as plot gets messy
      txt = plt.text(
          x,
          y,
          node_text,
          horizontalalignment="center",
          verticalalignment="center",
          fontsize=12)
      txt.set_path_effects(
          [PathEffects.withStroke(linewidth=3, foreground="w")])

      if self.num_populations > 1:
        sub_pos = nx.circular_layout(subgraph)
        subnode_labels = dict()
        strat_profile = utils.get_strat_profile_from_id(
            self.num_strats_per_population, i_strat_profile)
        for i_population in subgraph.nodes():
          i_strat = strat_profile[i_population]
          subnode_labels[i_population] = "$s^{" + str(i_population + 1) + "}="
          subnode_labels[i_population] += (
              self.state_labels[i_population][i_strat] + "$")
          # Adjust the node positions generated by NetworkX's circular_layout(),
          # such that the node for the 1st strategy starts on the left.
          sub_pos[i_population] = (-sub_pos[i_population] * subnode_separation +
                                   self.pos[i_strat_profile])
        nx.draw(
            subgraph,
            pos=sub_pos,
            with_labels=True,
            width=0.,
            node_color="w",
            labels=subnode_labels,
            node_size=2500)

  def compute_and_draw_network(self):
    """Computes the various node/edge connections of the graph and draws it."""

    if np.max(self.rhos) < self.rho_m:
      print("All node-to-node fixation probabilities (not including self-cycles"
            " are lower than neutral. Thus, no graph will be drawn.")
      return

    self.g = nx.MultiDiGraph()
    self.edge_labels = {}
    self.edge_alphas = []
    rho_max = np.max(self.rhos / self.rho_m)
    rho_m_alpha = 0.1  # Transparency of neutral selection edges

    for i in range(self.num_profiles):
      for j in range(self.num_profiles):
        # Do not draw edge if any node involved is skipped
        if j not in self.nodes_to_skip and i not in self.nodes_to_skip:
          rate = self.rhos[i][j] / self.rho_m
          # Draws edges when fixation from one strategy to another occurs (i.e.,
          # rate > 1), or with fixation equal to neutral selection probability
          # (i.e., rate == 1). This is consistent with visualizations used in
          # finite-population literature.
          if rate > 1:
            # Compute alphas. Clip needed due to numerical precision.
            alpha = np.clip(rho_m_alpha + (1 - rho_m_alpha) * rate / rho_max,
                            None, 1.)
            self.g.add_edge(i, j, weight=alpha, label="{:.01f}".format(rate))
            self.edge_alphas.append(alpha)
          elif np.isclose(rate, 1):
            alpha = rho_m_alpha
            self.g.add_edge(i, j, weight=alpha, label="{:.01f}".format(rate))
            self.edge_alphas.append(alpha)
          # Label edges for non-self-loops with sufficient flowrate
          if i != j and rate > 1:
            edge_string = "$" + str(np.round(rate, decimals=2)) + "\\rho_m$"
          else:
            edge_string = ""
          self.edge_labels[(i, j)] = edge_string

    # MultiDiGraph nodes are not ordered, so order the node colors accordingly
    self.node_colors = [self.pi[node] for node in self.g.nodes()]

    self.cycles = list(nx.simple_cycles(self.g))
    self.num_cycles = len(self.cycles)

    # Color the edges of cycles if user requested it
    if self.i_cycle_to_show >= 0:
      all_cycle_edges = [
          zip(nodes, (nodes[1:] + nodes[:1])) for nodes in self.cycles
      ]
      cur_cycle_edges = all_cycle_edges[self.i_cycle_to_show]
      self.edge_colors = []
      for u, v in self.g.edges():
        if (u, v) in cur_cycle_edges:
          self.edge_colors.append([1., 0., 0.])
        else:
          self.edge_colors.append([1. - self.g[u][v][0]["weight"]] * 3)
    else:
      self.edge_colors = [
          [1. - self.g[u][v][0]["weight"]] * 3 for u, v in self.g.edges()
      ]
      self.edge_alphas = [self.g[u][v][0]["weight"] for u, v in self.g.edges()]

    ax = plt.gca()

    # Centered circular pose
    self.pos = nx.layout.circular_layout(self.g)
    all_x = [node_pos[0] for node, node_pos in self.pos.items()]
    all_y = [node_pos[1] for node, node_pos in self.pos.items()]
    min_x = np.min(all_x)
    max_x = np.max(all_x)
    min_y = np.min(all_y)
    max_y = np.max(all_y)
    for _, node_pos in self.pos.items():
      node_pos[0] -= (max_x + min_x) / 2
      node_pos[1] -= (max_y + min_y) / 2

    # Rendering
    self._draw_network()
    if self.first_run:
      ax.autoscale_view()
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.ylim(-1.3, 1.3)
    plt.xlim(-1.3, 1.3)
    if self.first_run:
      self.first_run = False
      plt.axis("off")
      plt.show()


def _draw_pie(ax,
              ratios,
              colors,
              x_center=0,
              y_center=0,
              size=100,
              clip_on=True,
              zorder=0):
  """Plots a pie chart.

  Args:
    ax: plot axis.
    ratios: list indicating size of each pie slice, with elements summing to 1.
    colors: list indicating color of each pie slice.
    x_center: x coordinate of pie center.
    y_center: y coordinate of pie center.
    size: pie size.
    clip_on: control clipping of pie (e.g., to show it when it's out of axis).
    zorder: plot z order (e.g., to show pie on top of other plot elements).
  """
  xy = []
  start = 0.
  for ratio in ratios:
    x = [0] + np.cos(
        np.linspace(2 * np.pi * start, 2 * np.pi *
                    (start + ratio), 30)).tolist()
    y = [0] + np.sin(
        np.linspace(2 * np.pi * start, 2 * np.pi *
                    (start + ratio), 30)).tolist()
    xy.append(list(zip(x, y)))
    start += ratio

  for i, xyi in enumerate(xy):
    ax.scatter([x_center], [y_center],
               marker=xyi,
               s=size,
               facecolor=colors[i],
               edgecolors="none",
               clip_on=clip_on,
               zorder=zorder)


def generate_sorted_masses_strats(pi_list, curr_alpha_idx, strats_to_go):
  """Generates a sorted list of (mass, strats) tuples.

  Args:
    pi_list: List of stationary distributions, pi
    curr_alpha_idx: Index in alpha_list for which to start clustering
    strats_to_go: List of strategies that still need to be ordered

  Returns:
    Sorted list of (mass, strats) tuples.
  """
  if curr_alpha_idx > 0:
    sorted_masses_strats = list()
    masses_to_strats = utils.cluster_strats(pi_list[curr_alpha_idx,
                                                    strats_to_go])

    for mass, strats in sorted(masses_to_strats.items(), reverse=True):
      if len(strats) > 1:
        to_append = generate_sorted_masses_strats(pi_list, curr_alpha_idx - 1,
                                                  strats)

        to_append = [(mass, [strats_to_go[s]
                             for s in strats_list])
                     for (mass, strats_list) in to_append]

        sorted_masses_strats.extend(to_append)
      else:
        sorted_masses_strats.append((mass, [
            strats_to_go[strats[0]],
        ]))

    return sorted_masses_strats
  else:
    to_return = sorted(
        utils.cluster_strats(pi_list[curr_alpha_idx, strats_to_go]).items(),
        reverse=True)
    to_return = [(mass, [strats_to_go[s]
                         for s in strats_list])
                 for (mass, strats_list) in to_return]
    return to_return


def plot_pi_vs_alpha(pi_list,
                     alpha_list,
                     num_populations,
                     num_strats_per_population,
                     strat_labels,
                     num_strats_to_label,
                     plot_semilogx=True,
                     xlabel=r"Ranking-intensity $\alpha$",
                     ylabel=r"Strategy mass in stationary distribution $\pi$",
                     legend_sort_clusters=False):
  """Plots stationary distributions, pi, against selection intensities, alpha.

  Args:
    pi_list: List of stationary distributions, pi.
    alpha_list: List of selection intensities, alpha.
    num_populations: The number of populations.
    num_strats_per_population: List of the number of strategies per population.
    strat_labels: Human-readable strategy labels.
    num_strats_to_label: The number of top strategies to label in the legend.
    plot_semilogx: Boolean set to enable/disable semilogx plot.
    xlabel: Plot xlabel.
    ylabel: Plot ylabel.
    legend_sort_clusters: If true, strategies in the same cluster are sorted in
      the legend according to orderings for earlier alpha values. Primarily for
      visualization purposes! Rankings for lower alpha values should be
      interpreted carefully.
  """

  # Cluster strategies for which the stationary distribution has similar masses
  masses_to_strats = utils.cluster_strats(pi_list[-1, :])

  # Set colors
  num_strat_profiles = np.shape(pi_list)[1]
  num_strats_to_label = min(num_strats_to_label, num_strat_profiles)
  cmap = plt.get_cmap("Paired")
  colors = [cmap(i) for i in np.linspace(0, 1, num_strat_profiles)]

  # Plots stationary distribution vs. alpha series
  plt.figure(facecolor="w")
  axes = plt.gca()

  legend_line_objects = []
  legend_labels = []

  rank = 1
  num_strats_printed = 0
  add_legend_entries = True

  if legend_sort_clusters:
    sorted_masses_strats = generate_sorted_masses_strats(
        pi_list, pi_list.shape[0] - 1, range(pi_list.shape[1]))
  else:
    sorted_masses_strats = sorted(masses_to_strats.items(), reverse=True)

  for mass, strats in sorted_masses_strats:
    for profile_id in strats:
      if num_populations == 1:
        strat_profile = profile_id
      else:
        strat_profile = utils.get_strat_profile_from_id(
            num_strats_per_population, profile_id)

      if plot_semilogx:
        series = plt.semilogx(
            alpha_list,
            pi_list[:, profile_id],
            color=colors[profile_id],
            linewidth=2)
      else:
        series = plt.plot(
            alpha_list,
            pi_list[:, profile_id],
            color=colors[profile_id],
            linewidth=2)

      if add_legend_entries:
        if num_strats_printed >= num_strats_to_label:
          # Placeholder blank series for remaining entries
          series = plt.semilogx(np.NaN, np.NaN, "-", color="none")
          label = "..."
          add_legend_entries = False
        else:
          label = utils.get_label_from_strat_profile(num_populations,
                                                     strat_profile,
                                                     strat_labels)
        legend_labels.append(label)
        legend_line_objects.append(series[0])
      num_strats_printed += 1
    rank += 1

  # Plots pie charts on far right of figure to indicate clusters of strategies
  # with identical rank
  for mass, strats in iter(masses_to_strats.items()):
    _draw_pie(
        axes,
        ratios=[1 / len(strats)] * len(strats),
        colors=[colors[i] for i in strats],
        x_center=alpha_list[-1],
        y_center=mass,
        size=200,
        clip_on=False,
        zorder=10)

  # Axes ymax set slightly above highest stationary distribution mass
  max_mass = np.amax(pi_list)
  axes_y_max = np.ceil(
      10. * max_mass) / 10  # Round upward to nearest first decimal
  axes_y_max = np.clip(axes_y_max, 0., 1.)

  # Plots a rectangle highlighting the rankings on the far right of the figure
  box_x_min = alpha_list[-1] * 0.7
  box_y_min = np.min(pi_list[-1, :]) - 0.05 * axes_y_max
  width = 0.7 * alpha_list[-1]
  height = np.max(pi_list[-1, :]) - np.min(
      pi_list[-1, :]) + 0.05 * axes_y_max * 2
  axes.add_patch(
      patches.Rectangle((box_x_min, box_y_min),
                        width,
                        height,
                        edgecolor="b",
                        facecolor=(1, 0, 0, 0),
                        clip_on=False,
                        linewidth=5,
                        zorder=20))

  # Plot formatting
  axes.set_xlim(np.min(alpha_list), np.max(alpha_list))
  axes.set_ylim([0.0, axes_y_max])
  axes.set_xlabel(xlabel)
  axes.set_ylabel(ylabel)
  axes.set_axisbelow(True)  # Axes appear below data series in terms of zorder

  # Legend on the right side of the current axis
  box = axes.get_position()
  axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  axes.legend(
      legend_line_objects,
      legend_labels,
      loc="center left",
      bbox_to_anchor=(1.05, 0.5))
  plt.grid()
  plt.show()
