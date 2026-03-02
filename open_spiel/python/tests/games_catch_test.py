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

"""Tests for the game-specific functions for chess."""


from absl.testing import absltest
import numpy as np

import pyspiel

NUM_SIM_GAMES = 10
SEED = 87375711


class GamesBlackjackTest(absltest.TestCase):

  def test_blackjack_game_sim(self):
    game = pyspiel.load_game("catch")
    for _ in range(NUM_SIM_GAMES):
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
          action = np.random.choice(action_list, p=prob_list)
        else:
          print(
              f"ball row: {state.ball_row()}, ball col: {state.ball_col()},"
              f" paddle col: {state.paddle_col()}"
          )
          actions = state.legal_actions()
          action = np.random.choice(actions)
          print("Action chosen:", state.action_to_string(action))
        state.apply_action(action)
        print(state)
      print("Terminal state: ")
      print(str(state))
      print("Returns:", state.returns())
      print("")


if __name__ == "__main__":
  np.random.seed(SEED)
  absltest.main()
