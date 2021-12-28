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

"""Example of ResponseGraphUCB run on a 2x2 game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import matplotlib.pyplot as plt
import numpy as np

from open_spiel.python.algorithms import response_graph_ucb
from open_spiel.python.algorithms import response_graph_ucb_utils


def get_example_2x2_payoffs():
  mean_payoffs = np.random.uniform(-1, 1, size=(2, 2, 2))
  mean_payoffs[0, :, :] = np.asarray([[0.5, 0.85], [0.15, 0.5]])
  mean_payoffs[1, :, :] = 1 - mean_payoffs[0, :, :]
  return mean_payoffs


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  mean_payoffs = get_example_2x2_payoffs()
  game = response_graph_ucb_utils.BernoulliGameSampler(
      [2, 2], mean_payoffs, payoff_bounds=[-1., 1.])
  game.p_max = mean_payoffs
  game.means = mean_payoffs
  print('Game means:\n', game.means)

  exploration_strategy = 'uniform-exhaustive'
  confidence_method = 'ucb-standard'
  r_ucb = response_graph_ucb.ResponseGraphUCB(
      game,
      exploration_strategy=exploration_strategy,
      confidence_method=confidence_method,
      delta=0.1)
  results = r_ucb.run()

  # Plotting
  print('Number of total samples: {}'.format(np.sum(r_ucb.count[0])))
  r_ucb.visualise_2x2x2(real_values=game.means, graph=results['graph'])
  r_ucb.visualise_count_history(figsize=(5, 3))
  plt.gca().xaxis.label.set_fontsize(15)
  plt.gca().yaxis.label.set_fontsize(15)

  # Compare to ground truth graph
  real_graph = r_ucb.construct_real_graph()
  r_ucb.plot_graph(real_graph)
  plt.show()

if __name__ == '__main__':
  app.run(main)
