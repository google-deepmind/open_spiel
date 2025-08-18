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

"""Tests for the game-specific functions for crazy eights."""


from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import pyspiel

# crazy_eights = pyspiel.crazy_eights

NUM_SIM_GAMES = 10
SEED = 87375711


class GamesCrazyEightsTest(parameterized.TestCase):

  def test_crazy_eights_game_sim(self):
    game = pyspiel.load_game("crazy_eights")
    for _ in range(NUM_SIM_GAMES):
      print("----------------")
      print("New game")
      print("----------------")
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
          action = np.random.choice(action_list, p=prob_list)
          print("Chance samples", state.action_to_string(action))
        else:
          print("Player turn")
          print(
              "Dealer's deck:",
              state.get_dealer_deck(),
          )
          actions = state.legal_actions()
          action = np.random.choice(actions)
          print("Action chosen:", state.action_to_string(action))
        state.apply_action(action)
        print("")
      print("Terminal state: ")
      print(str(state))
      print("Returns:", state.returns())
      print("")


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()
