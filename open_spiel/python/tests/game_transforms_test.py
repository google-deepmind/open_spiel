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

"""Test Python bindings for game transforms."""

from absl.testing import absltest

import numpy as np

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import expected_game_score
import pyspiel


SEED = 1098097


class RepeatedGameTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(SEED)

  def test_create_repeated_game(self):
    """Test both create_repeated_game function signatures."""
    repeated_game = pyspiel.create_repeated_game("matrix_rps",
                                                 {"num_repetitions": 10})
    assert repeated_game.utility_sum() == 0
    state = repeated_game.new_initial_state()
    for _ in range(10):
      state.apply_actions([0, 0])
    assert state.is_terminal()

    stage_game = pyspiel.load_game("matrix_mp")
    repeated_game = pyspiel.create_repeated_game(stage_game,
                                                 {"num_repetitions": 5})
    state = repeated_game.new_initial_state()
    for _ in range(5):
      state.apply_actions([0, 0])
    assert state.is_terminal()

    stage_game = pyspiel.load_game("matrix_pd")
    repeated_game = pyspiel.create_repeated_game(stage_game,
                                                 {"num_repetitions": 5})
    assert repeated_game.utility_sum() is None

  def test_cached_tree_sim(self):
    """Test both create_cached_tree function signatures."""
    for game_name in ["kuhn_poker", "python_tic_tac_toe"]:
      cached_tree_game = pyspiel.convert_to_cached_tree(
          pyspiel.load_game(game_name))
      assert cached_tree_game.num_players() == 2
      for _ in range(10):
        state = cached_tree_game.new_initial_state()
        while not state.is_terminal():
          legal_actions = state.legal_actions()
          action = np.random.choice(legal_actions)
          state.apply_action(action)
        self.assertTrue(state.is_terminal())

  def test_cached_tree_cfr_kuhn(self):
    game = pyspiel.load_game("cached_tree(game=kuhn_poker())")
    cfr_solver = cfr.CFRSolver(game)
    for _ in range(300):
      cfr_solver.evaluate_and_update_policy()
    average_policy = cfr_solver.average_policy()
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    np.testing.assert_allclose(
        average_policy_values, [-1 / 18, 1 / 18], atol=1e-3)


if __name__ == "__main__":
  absltest.main()
