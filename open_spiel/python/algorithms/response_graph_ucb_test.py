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
"""Tests for open_spiel.python.algorithms.response_graph_ucb."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import absltest

# pylint: disable=g-import-not-at-top
import matplotlib
matplotlib.use('agg')  # switch backend for testing

import numpy as np

from open_spiel.python.algorithms import response_graph_ucb
from open_spiel.python.algorithms import response_graph_ucb_utils


class ResponseGraphUcbTest(absltest.TestCase):

  def get_example_2x2_payoffs(self):
    mean_payoffs = np.random.uniform(-1, 1, size=(2, 2, 2))
    mean_payoffs[0, :, :] = np.asarray([[0.5, 0.85], [0.15, 0.5]])
    mean_payoffs[1, :, :] = 1 - mean_payoffs[0, :, :]
    return mean_payoffs

  def test_sampler(self):
    mean_payoffs = self.get_example_2x2_payoffs()
    game = response_graph_ucb_utils.BernoulliGameSampler(
        [2, 2], mean_payoffs, payoff_bounds=[-1., 1.])
    game.p_max = mean_payoffs
    game.means = mean_payoffs

    # Parameters to run
    sampling_methods = [
        'uniform-exhaustive', 'uniform', 'valence-weighted', 'count-weighted'
    ]
    conf_methods = [
        'ucb-standard', 'ucb-standard-relaxed', 'clopper-pearson-ucb',
        'clopper-pearson-ucb-relaxed'
    ]
    per_payoff_confidence = [True, False]
    time_dependent_delta = [True, False]

    methods = list(itertools.product(sampling_methods,
                                     conf_methods,
                                     per_payoff_confidence,
                                     time_dependent_delta))
    max_total_interactions = 50

    for m in methods:
      r_ucb = response_graph_ucb.ResponseGraphUCB(
          game,
          exploration_strategy=m[0],
          confidence_method=m[1],
          delta=0.1,
          ucb_eps=1e-1,
          per_payoff_confidence=m[2],
          time_dependent_delta=m[3])
      _ = r_ucb.run(max_total_iterations=max_total_interactions)

  def test_soccer_data_import(self):
    response_graph_ucb_utils.get_soccer_data()

if __name__ == '__main__':
  absltest.main()
