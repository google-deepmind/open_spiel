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

"""Tests for open_spiel.python.egt.alpharank_visualizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

# pylint: disable=g-import-not-at-top
import matplotlib
matplotlib.use("agg")  # switch backend for testing

import mock
import numpy as np

from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils
import pyspiel


class AlpharankVisualizerTest(absltest.TestCase):

  @mock.patch("%s.alpharank_visualizer.plt" % __name__)
  def test_plot_pi_vs_alpha(self, mock_plt):
    # Construct game
    game = pyspiel.load_matrix_game("matrix_rps")
    payoff_tables = utils.game_payoffs_array(game)
    _, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)

    # Compute alpharank
    alpha = 1e2
    _, _, pi, num_profiles, num_strats_per_population = (
        alpharank.compute(payoff_tables, alpha=alpha))
    strat_labels = utils.get_strat_profile_labels(payoff_tables,
                                                  payoffs_are_hpt_format)
    num_populations = len(payoff_tables)

    # Construct synthetic pi-vs-alpha history
    pi_list = np.empty((num_profiles, 0))
    alpha_list = []
    for _ in range(2):
      pi_list = np.append(pi_list, np.reshape(pi, (-1, 1)), axis=1)
      alpha_list.append(alpha)

    # Test plotting code (via pyplot mocking to prevent plot pop-up)
    alpharank_visualizer.plot_pi_vs_alpha(
        pi_list.T,
        alpha_list,
        num_populations,
        num_strats_per_population,
        strat_labels,
        num_strats_to_label=0)
    self.assertTrue(mock_plt.show.called)


if __name__ == "__main__":
  absltest.main()
