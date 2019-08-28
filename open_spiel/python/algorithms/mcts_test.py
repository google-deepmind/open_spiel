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

"""Tests for google3.third_party.open_spiel.python.algorithms.mcts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import unittest

import numpy as np

from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.algorithms import mcts
import pyspiel


class MctsBotTest(unittest.TestCase):

  def test_can_play_tic_tac_toe(self):
    game = pyspiel.load_game("tic_tac_toe")
    uct_c = math.sqrt(2)
    max_search_nodes = 100
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=20)
    bots = [
        mcts.MCTSBot(game, 0, uct_c, max_search_nodes, evaluator),
        mcts.MCTSBot(game, 1, uct_c, max_search_nodes, evaluator),
    ]
    v = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    self.assertEqual(v[0] + v[1], 0)


if __name__ == "__main__":
  unittest.main()
