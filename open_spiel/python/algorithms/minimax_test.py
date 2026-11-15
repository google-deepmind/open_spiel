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

"""Tests for open_spiel.python.algorithms.minimax."""

from absl.testing import absltest

from open_spiel.python.algorithms import minimax
import pyspiel


def first_player_advantage(state):
  # PigState::score is not bound in Python; parse ToString:
  # "Scores: 0 0, Turn total: 0\nCurrent player: 0"
  scores = [int(x) for x in str(state).split(",")[0].split()[1:]]
  return scores[0] - scores[1]


class MinimaxTest(absltest.TestCase):

  def test_compute_game_value(self):
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")

    game_score, best_actions = minimax.alpha_beta_search(tic_tac_toe)
    self.assertEqual(0., game_score)
    self.assertEqual(best_actions, [0, 1, 2, 3, 4, 5, 6, 7, 8])

  def test_compute_game_value_with_evaluation_function(self):
    # We only check it runs
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")

    game_score, best_actions = minimax.alpha_beta_search(
        tic_tac_toe, value_function=lambda x: 0, maximum_depth=1)
    self.assertEqual(0., game_score)
    self.assertEqual(best_actions, [0, 1, 2, 3, 4, 5, 6, 7, 8])

  def test_win(self):
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")
    state = tic_tac_toe.new_initial_state()

    # Construct:
    # .o.
    # .x.
    # ...
    # Optimal actions: 0 (R1C1), 2 (R1C3), 3 (R2C1), 5 (R2C3), 6 (R3C1),
    # 8 (R3C3). Action 7 (R3C2) is legal but only draws.
    state.apply_action(4)
    state.apply_action(1)
    game_score, best_actions = minimax.alpha_beta_search(
        tic_tac_toe, state=state)
    self.assertEqual(1., game_score)
    self.assertEqual(best_actions, [0, 2, 3, 5, 6, 8])

  def test_loss(self):
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")
    state = tic_tac_toe.new_initial_state()

    # Construct:
    # ...
    # xox
    # ..o
    # Optimal actions: 0 (R1C1), 1 (R1C2), 2 (R1C3), 6 (R3C1), 7 (R3C2).
    state.apply_action(5)
    state.apply_action(4)
    state.apply_action(3)
    state.apply_action(8)
    game_score, best_actions = minimax.alpha_beta_search(
        tic_tac_toe, state=state)
    self.assertEqual(-1., game_score)
    self.assertEqual(best_actions, [0, 1, 2, 6, 7])

  def test_single_action(self):
    tic_tac_toe = pyspiel.load_game("tic_tac_toe")
    state = tic_tac_toe.new_initial_state()

    # Construct:
    # xox
    # ...
    # ...
    # Optimal actions: 4 (R2C2) only. Every other legal action loses.
    state.apply_action(0)
    state.apply_action(1)
    state.apply_action(2)
    game_score, best_actions = minimax.alpha_beta_search(
        tic_tac_toe, state=state)
    self.assertEqual(0., game_score)
    self.assertEqual(best_actions, [4])

  def test_expectiminimax(self):
    pig = pyspiel.load_game("pig", {"diceoutcomes": 3})
    state = pig.new_initial_state()
    value, best_actions = minimax.expectiminimax(state, 2,
                                                 first_player_advantage, 0)
    self.assertEqual(1.0 / 3 * 2 + 1.0 / 3 * 3, value)
    self.assertEqual([0], best_actions)


if __name__ == "__main__":
  absltest.main()
