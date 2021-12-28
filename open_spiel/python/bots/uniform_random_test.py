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

"""Unit test for uniform random bot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from absl.testing import absltest
from open_spiel.python.bots import uniform_random
import pyspiel


class BotTest(absltest.TestCase):

  def test_policy_is_uniform(self):
    game = pyspiel.load_game("leduc_poker")
    bots = [
        uniform_random.UniformRandomBot(0, random),
        uniform_random.UniformRandomBot(1, random)
    ]

    # deal each player a card
    state = game.new_initial_state()
    state.apply_action(2)
    state.apply_action(4)

    # p0 starts: uniform from [check, bet]
    policy, _ = bots[0].step_with_policy(state)
    self.assertCountEqual(policy, [(1, 0.5), (2, 0.5)])

    # Afte p0 bets, p1 chooses from [fold, call, raise]
    state.apply_action(2)
    policy, _ = bots[1].step_with_policy(state)
    self.assertCountEqual(policy, [(0, 1 / 3), (1, 1 / 3), (2, 1 / 3)])

  def test_no_legal_actions(self):
    game = pyspiel.load_game("kuhn_poker")
    bot = uniform_random.UniformRandomBot(0, random)
    state = game.new_initial_state()
    state.apply_action(2)  # deal
    state.apply_action(1)  # deal
    state.apply_action(1)  # bet
    state.apply_action(0)  # fold
    bot.restart_at(state)
    policy, action = bot.step_with_policy(state)
    self.assertEqual(policy, [])
    self.assertEqual(action, pyspiel.INVALID_ACTION)


if __name__ == "__main__":
  absltest.main()
