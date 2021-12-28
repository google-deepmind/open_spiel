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

"""Tests for open_spiel.python.algorithms.cfr."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import fictitious_play
import pyspiel


class FictitiousPlayTest(absltest.TestCase):

  def test_xfp(self):
    game = pyspiel.load_game("kuhn_poker")
    xfp_solver = fictitious_play.XFPSolver(game)
    for _ in range(100):
      xfp_solver.iteration()
    average_policies = xfp_solver.average_policy_tables()
    tabular_policy = policy.TabularPolicy(game)
    for player_id in range(2):
      for info_state, state_policy in average_policies[player_id].items():
        policy_to_update = tabular_policy.policy_for_key(info_state)
        for action, probability in state_policy.items():
          policy_to_update[action] = probability
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [tabular_policy, tabular_policy])
    print("Kuhn 2P average values after 10 iterations")
    print("P0: {}".format(average_policy_values[0]))
    print("P1: {}".format(average_policy_values[1]))
    self.assertIsNotNone(average_policy_values)
    self.assertTrue(
        np.allclose(average_policy_values, [-1 / 18, 1 / 18], atol=1e-3))

  def test_meta_game_kuhn2p(self):
    print("Kuhn 2p")
    game = pyspiel.load_game("kuhn_poker")
    xfp_solver = fictitious_play.XFPSolver(game, save_oracles=True)
    for _ in range(3):
      xfp_solver.iteration()
    meta_games = xfp_solver.get_empirical_metagame(10, seed=1)
    self.assertIsNotNone(meta_games)
    # Metagame utility matrices for each player
    for i in range(2):
      print("player {}: \n{}".format(i + 1, meta_games[i]))

  def test_meta_game_kuhn3p(self):
    print("Kuhn 3p")
    game = pyspiel.load_game("kuhn_poker", {"players": 3})
    xfp_solver = fictitious_play.XFPSolver(game, save_oracles=True)
    for _ in range(3):
      xfp_solver.iteration()
    meta_games = xfp_solver.get_empirical_metagame(10, seed=3)
    self.assertIsNotNone(meta_games)
    # Metagame utility tensors for each player
    for i in range(3):
      print("player {}: \n{}".format(i + 1, meta_games[i]))

  def test_meta_game_kuhn4p(self):
    print("Kuhn 4p")
    game = pyspiel.load_game("kuhn_poker", {"players": 4})
    xfp_solver = fictitious_play.XFPSolver(game, save_oracles=True)
    for _ in range(3):
      xfp_solver.iteration()
    meta_games = xfp_solver.get_empirical_metagame(10, seed=1)
    self.assertIsNotNone(meta_games)
    # Metagame utility tensors for each player
    for i in range(4):
      print("player {}: \n{}".format(i + 1, meta_games[i]))

  def test_meta_game_leduc2p(self):
    print("Leduc 2p")
    game = pyspiel.load_game("leduc_poker")
    xfp_solver = fictitious_play.XFPSolver(game, save_oracles=True)
    for _ in range(3):
      xfp_solver.iteration()
    meta_games = xfp_solver.get_empirical_metagame(10, seed=86487)
    self.assertIsNotNone(meta_games)
    # Metagame utility matrices for each player
    for i in range(2):
      print("player {}: \n{}".format(i + 1, meta_games[i]))

  def test_matching_pennies_3p(self):
    game = pyspiel.load_game_as_turn_based("matching_pennies_3p")
    xfp_solver = fictitious_play.XFPSolver(game)
    for i in range(1000):
      xfp_solver.iteration()
      if i % 10 == 0:
        conv = exploitability.nash_conv(game, xfp_solver.average_policy())
        print("FP in Matching Pennies 3p. Iter: {}, NashConv: {}".format(
            i, conv))

  def test_shapleys_game(self):
    game = pyspiel.load_game_as_turn_based("matrix_shapleys_game")
    xfp_solver = fictitious_play.XFPSolver(game)
    for i in range(1000):
      xfp_solver.iteration()
      if i % 10 == 0:
        conv = exploitability.nash_conv(game, xfp_solver.average_policy())
        print("FP in Shapley's Game. Iter: {}, NashConv: {}".format(i, conv))


if __name__ == "__main__":
  absltest.main()
