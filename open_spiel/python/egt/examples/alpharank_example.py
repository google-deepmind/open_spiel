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

"""Example running AlphaRank on OpenSpiel games.

  AlphaRank output variable names corresponds to the following paper:
    https://arxiv.org/abs/1903.01373
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

from open_spiel.python.algorithms import fictitious_play
from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils
import pyspiel


def get_kuhn_poker_data(num_players=3):
  """Returns the kuhn poker data for the number of players specified."""
  game = pyspiel.load_game('kuhn_poker', {'players': num_players})
  xfp_solver = fictitious_play.XFPSolver(game, save_oracles=True)
  for _ in range(3):
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


def main(unused_arg):
  # Construct meta-game payoff tables
  payoff_tables = get_kuhn_poker_data()
  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  strat_labels = utils.get_strat_profile_labels(payoff_tables,
                                                payoffs_are_hpt_format)

  # Run AlphaRank
  rhos, rho_m, pi, _, _ = alpharank.compute(payoff_tables, alpha=1e2)

  # Report & plot results
  alpharank.print_results(
      payoff_tables, payoffs_are_hpt_format, rhos=rhos, rho_m=rho_m, pi=pi)
  utils.print_rankings_table(payoff_tables, pi, strat_labels)
  m_network_plotter = alpharank_visualizer.NetworkPlot(
      payoff_tables, rhos, rho_m, pi, strat_labels, num_top_profiles=8)
  m_network_plotter.compute_and_draw_network()


if __name__ == '__main__':
  app.run(main)
