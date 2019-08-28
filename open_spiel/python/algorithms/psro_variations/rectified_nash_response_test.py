# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for open_spiel.python.algorithms.rnr."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from absl.testing import parameterized

from open_spiel.python.algorithms.psro_variations import optimization_oracle
from open_spiel.python.algorithms.psro_variations import rectified_nash_response
import pyspiel


class RectifiedNashResponseTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "kuhn_poker",
          "rnr_iterations": 2,
          "sims_per_entry": 2,
          "rectify_training": True,
          "restrict_training": True,
          "game_name": "kuhn_poker",
          "meta_strategy_computation_method": "nash",
      }, {
          "testcase_name": "kuhn_poker_prd",
          "rnr_iterations": 2,
          "sims_per_entry": 2,
          "rectify_training": False,
          "restrict_training": True,
          "game_name": "kuhn_poker",
          "meta_strategy_computation_method": "prd",
      }, {
          "testcase_name": "kuhn_poker_uniform",
          "rnr_iterations": 2,
          "sims_per_entry": 2,
          "rectify_training": False,
          "restrict_training": False,
          "game_name": "kuhn_poker",
          "meta_strategy_computation_method": "uniform",
      }, {
          "testcase_name": "leduc_poker",
          "rnr_iterations": 2,
          "sims_per_entry": 2,
          "rectify_training": True,
          "restrict_training": False,
          "game_name": "leduc_poker",
          "meta_strategy_computation_method": "nash",
      })
  def test_rnr(self, game_name, rnr_iterations, sims_per_entry,
               rectify_training, restrict_training,
               meta_strategy_computation_method):
    game = pyspiel.load_game(game_name)
    oracle = optimization_oracle.EvolutionaryStrategyOracle(
        number_policies_sampled=2, number_episodes_sampled=2)
    rnr_solver = rectified_nash_response.RNRSolver(
        game,
        oracle,
        sims_per_entry=sims_per_entry,
        rectify_training=rectify_training,
        restrict_training=restrict_training,
        meta_strategy_computation_method=meta_strategy_computation_method)
    for _ in range(rnr_iterations):
      rnr_solver.iteration()
    meta_game = rnr_solver.get_meta_game
    nash_probabilities = rnr_solver.get_and_update_meta_strategies()

    print(game_name + " Nash probabilities")
    print(nash_probabilities)
    print("")

    print(game_name + " Meta Game Values")
    print(meta_game)
    print("")


if __name__ == "__main__":
  unittest.main()
