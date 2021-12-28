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

"""Unit test for Information Set MCTS bot.

This test mimics the basic C++ tests in algorithms/is_mcts_test.cc.
"""
# pylint: disable=g-unreachable-test-method

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import evaluate_bots
import pyspiel

SEED = 12983641


class ISMCTSBotTest(absltest.TestCase):

  def ismcts_play_game(self, game):
    evaluator = pyspiel.RandomRolloutEvaluator(1, SEED)
    for final_policy_type in [
        pyspiel.ISMCTSFinalPolicyType.NORMALIZED_VISIT_COUNT,
        pyspiel.ISMCTSFinalPolicyType.MAX_VISIT_COUNT,
        pyspiel.ISMCTSFinalPolicyType.MAX_VALUE
    ]:
      bot = pyspiel.ISMCTSBot(SEED, evaluator, 5.0, 1000, -1, final_policy_type,
                              False, False)
      bots = [bot] * game.num_players()
      evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
      bot = pyspiel.ISMCTSBot(SEED, evaluator, 5.0, 1000, 10, final_policy_type,
                              False, False)
      bots = [bot] * game.num_players()
      evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
      bot = pyspiel.ISMCTSBot(SEED, evaluator, 5.0, 1000, 10, final_policy_type,
                              True, True)
      bots = [bot] * game.num_players()
      evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)

  def test_basic_sim_kuhn(self):
    game = pyspiel.load_game("kuhn_poker")
    self.ismcts_play_game(game)
    game = pyspiel.load_game("kuhn_poker(players=3)")
    self.ismcts_play_game(game)

  def test_basic_sim_leduc(self):
    game = pyspiel.load_game("leduc_poker")
    self.ismcts_play_game(game)
    game = pyspiel.load_game("leduc_poker(players=3)")
    self.ismcts_play_game(game)


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()
