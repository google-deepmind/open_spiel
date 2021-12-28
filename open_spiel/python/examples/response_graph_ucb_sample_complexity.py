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

"""Example computing ResponseGraphUCB sample complexity results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.algorithms import response_graph_ucb
from open_spiel.python.algorithms import response_graph_ucb_utils as utils

FLAGS = flags.FLAGS

flags.DEFINE_string('game_name', 'soccer', 'Name of the game.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Parameters to run
  deltas = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
  sampling_methods = [
      'uniform-exhaustive', 'uniform', 'valence-weighted', 'count-weighted'
  ]
  conf_methods = [
      'ucb-standard', 'ucb-standard-relaxed', 'clopper-pearson-ucb',
      'clopper-pearson-ucb-relaxed'
  ]

  methods = list(itertools.product(sampling_methods, conf_methods))
  mean_counts = {m: [[] for _ in range(len(deltas))] for m in methods}
  edge_errs = {m: [[] for _ in range(len(deltas))] for m in methods}

  if FLAGS.game_name == 'bernoulli':
    max_total_interactions = 50000
    repetitions = 20
  elif FLAGS.game_name == 'soccer':
    max_total_interactions = 100000
    repetitions = 5
  elif FLAGS.game_name == 'kuhn_poker_3p':
    max_total_interactions = 100000
    repetitions = 5
  else:
    raise ValueError(
        'game_name must be "bernoulli", "soccer", or "kuhn_poker_3p".')

  for r in range(repetitions):
    print('Iteration {}'.format(r + 1))
    G = utils.get_game_for_sampler(FLAGS.game_name)  # pylint: disable=invalid-name

    for m in methods:
      print('  Method: {}'.format(m))
      for ix, d in enumerate(deltas):
        print('    Delta: {}'.format(d))
        r_ucb = response_graph_ucb.ResponseGraphUCB(
            G,
            exploration_strategy=m[0],
            confidence_method=m[1],
            delta=d,
            ucb_eps=1e-1)
        results = r_ucb.run(max_total_iterations=max_total_interactions)

        # Updated
        mean_counts[m][ix].append(results['interactions'])
        real_graph = r_ucb.construct_real_graph()
        edge_errs[m][ix].append(
            utils.digraph_edge_hamming_dist(real_graph, results['graph']))

  # Plotting
  _, axes = plt.subplots(1, 2, figsize=(10, 4))
  max_mean_count = 0
  for m in methods:
    utils.plot_timeseries(
        axes,
        id_ax=0,
        data=np.asarray(mean_counts[m]).T,
        xticks=deltas,
        xlabel=r'$\delta$',
        ylabel='Interactions required',
        label=utils.get_method_tuple_acronym(m),
        logx=True,
        logy=True,
        linespecs=utils.get_method_tuple_linespecs(m))
    if np.max(mean_counts[m]) > max_mean_count:
      max_mean_count = np.max(mean_counts[m])
  plt.xlim(left=np.min(deltas), right=np.max(deltas))
  plt.ylim(top=max_mean_count * 1.05)

  max_error = 0
  for m in methods:
    utils.plot_timeseries(
        axes,
        id_ax=1,
        data=np.asarray(edge_errs[m]).T,
        xticks=deltas,
        xlabel=r'$\delta$',
        ylabel='Response graph errors',
        label=utils.get_method_tuple_acronym(m),
        logx=True,
        logy=False,
        linespecs=utils.get_method_tuple_linespecs(m))
    if np.max(edge_errs[m]) > max_error:
      max_error = np.max(edge_errs[m])
  plt.xlim(left=np.min(deltas), right=np.max(deltas))
  plt.ylim(bottom=0, top=max_error*1.05)

  # Shared legend
  plt.figure(figsize=(1, 6))
  plt.figlegend(
      *axes[0].get_legend_handles_labels(),
      loc='center right',
      bbox_to_anchor=(0.8, 0.5),
      bbox_transform=plt.gcf().transFigure,
      ncol=1,
      handlelength=1.7)
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  app.run(main)
