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

"""Tests for open_spiel.python.algorithms.generalized_psro."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.algorithms.psro_variations import generalized_psro
from open_spiel.python.algorithms.psro_variations import optimization_oracle
import pyspiel


class DummySolver(object):

  def __init__(self, policies, number_policies_selected, meta_strategies):
    self.policies = policies
    self.number_policies_selected = number_policies_selected
    self.meta_strategies = meta_strategies

  @property
  def get_policies(self):
    return self.policies

  @property
  def get_kwargs(self):
    return {"number_policies_selected": self.number_policies_selected}

  def get_and_update_meta_strategies(self, update=False):
    _ = update
    return self.meta_strategies


class GeneralizedPSROTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "kuhn_poker",
          "rnr_iterations": 2,
          "sims_per_entry": 2,
          "number_players": 2,
          "rectify_training": True,
          "training_strategy_selector": "rectified",
          "game_name": "kuhn_poker",
          "meta_strategy_method": "prd",
      }, {
          "testcase_name": "kuhn_poker_3p",
          "rnr_iterations": 2,
          "sims_per_entry": 2,
          "number_players": 3,
          "rectify_training": True,
          "training_strategy_selector": "probabilistic",
          "game_name": "kuhn_poker",
          "meta_strategy_method": "prd",
      }, {
          "testcase_name": "kuhn_poker_uniform",
          "rnr_iterations": 2,
          "sims_per_entry": 2,
          "number_players": 2,
          "rectify_training": True,
          "training_strategy_selector": "uniform",
          "game_name": "kuhn_poker",
          "meta_strategy_method": "uniform",
      }, {
          "testcase_name": "kuhn_poker_nash",
          "rnr_iterations": 2,
          "sims_per_entry": 2,
          "number_players": 2,
          "rectify_training": True,
          "training_strategy_selector": "exhaustive",
          "game_name": "leduc_poker",
          "meta_strategy_method": "nash",
      }, {
          "testcase_name": "leduc_poker",
          "rnr_iterations": 2,
          "sims_per_entry": 2,
          "number_players": 2,
          "rectify_training": True,
          "training_strategy_selector": "probabilistic_deterministic",
          "game_name": "leduc_poker",
          "meta_strategy_method": "prd",
      })
  def test_gpsro(self, game_name, rnr_iterations, sims_per_entry,
                 number_players, rectify_training, training_strategy_selector,
                 meta_strategy_method):
    game = pyspiel.load_game(game_name,
                             {"players": pyspiel.GameParameter(number_players)})
    oracle = optimization_oracle.EvolutionaryStrategyOracle(
        number_policies_sampled=2, number_episodes_sampled=2)
    g_psro_solver = generalized_psro.GenPSROSolver(
        game,
        oracle,
        sims_per_entry=sims_per_entry,
        rectify_training=rectify_training,
        training_strategy_selector=training_strategy_selector,
        meta_strategy_method=meta_strategy_method)
    for _ in range(rnr_iterations):
      g_psro_solver.iteration()
    meta_game = g_psro_solver.get_meta_game
    meta_probabilities = g_psro_solver.get_and_update_meta_strategies()

    logging.info("%s %sP - %s", game_name, str(number_players),
                 meta_strategy_method)
    logging.info("Meta Strategies")
    logging.info(meta_probabilities)
    logging.info("")

    logging.info("Meta Game Values")
    logging.info(meta_game)
    logging.info("")

  @parameterized.named_parameters(
      {
          "testcase_name": "rectified",
          "selector": generalized_psro.rectified_strategy_selector,
          "policies": [range(5) for _ in range(5)],
          "number_policies_selected": 1,
          "meta_strategies": [[0, 0, 1, 0, 0] for _ in range(5)],
          "results": [[2] for _ in range(5)],
      }, {
          "testcase_name": "prob_strat_selector",
          "selector": generalized_psro.probabilistic_strategy_selector,
          "policies": [range(5) for _ in range(5)],
          "number_policies_selected": 1,
          "meta_strategies": [[0, 0, 1, 0, 0] for _ in range(5)],
          "results": [[2] for _ in range(5)],
      }, {
          "testcase_name":
              "func_prob",
          "selector":
              generalized_psro.functional_probabilistic_strategy_selector,
          "policies": [range(5) for _ in range(5)],
          "number_policies_selected":
              1,
          "meta_strategies": [[0, 0, 1, 0, 0] for _ in range(5)],
          "results": [[2] for _ in range(5)],
      }, {
          "testcase_name":
              "deterministic",
          "selector":
              generalized_psro.probabilistic_deterministic_strategy_selector,
          "policies": [range(5) for _ in range(5)],
          "number_policies_selected":
              3,
          "meta_strategies": [[0, 0, 0.5, 0.3, 0.2] for _ in range(5)],
          "results": [[2, 3, 4] for _ in range(5)],
      })
  def test_selectors(self, selector, policies, number_policies_selected,
                     meta_strategies, results):
    dum_sol = DummySolver(policies, number_policies_selected, meta_strategies)
    used_policies = selector(dum_sol)

    for k in range(len(used_policies)):
      self.assertEqual(used_policies[k], results[k])


if __name__ == "__main__":
  absltest.main()
