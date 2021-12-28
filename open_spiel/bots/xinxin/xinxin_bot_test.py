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

"""Unit test for XinXin MCTS bot."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.algorithms import evaluate_bots
import pyspiel

SEED = 12983641


class ISMCTSBotTest(absltest.TestCase):

  def xinxin_play_game(self, game):
    bots = []
    for _ in range(4):
      bots.append(pyspiel.make_xinxin_bot(game.get_parameters()))
    evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)

  def test_basic_xinxin_selfplay(self):
    game = pyspiel.load_game("hearts")
    self.xinxin_play_game(game)


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()
