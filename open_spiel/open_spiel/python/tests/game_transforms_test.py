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
from open_spiel.python.games import iterated_prisoners_dilemma as ipd  # pylint: disable=unused-import
import pyspiel


SEED = 1098097


class GameTransformsTest(absltest.TestCase):

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

  def test_turn_based_simultaneous_game(self):
    tb_game = pyspiel.load_game(
        "turn_based_simultaneous_game(game="
        "goofspiel(num_cards=13,players=2,points_order=descending))")
    state = tb_game.new_initial_state()
    # Play 11 moves. This gets to the decision right before the game is over.
    # Because the last action is taken automatically there are only 12 moves
    # total.
    for _ in range(11):
      state.apply_action(state.legal_actions()[0])
      state.apply_action(state.legal_actions()[0])
    sim_state = state.simultaneous_game_state()
    assert not state.is_terminal()
    assert not sim_state.is_terminal()
    # For the last joint action, pull out the simultaneous state from inside the
    # wrapper and and apply joint action to it. Both the wrapped state and
    # the simultaneous state should be terminal after this.
    sim_state.apply_actions([sim_state.legal_actions(0)[0],
                             sim_state.legal_actions(1)[0]])
    assert state.is_terminal()
    assert sim_state.is_terminal()

  def test_turn_based_simultaneous_python_game(self):
    tb_game = pyspiel.load_game(
        "turn_based_simultaneous_game(game="
        "python_iterated_prisoners_dilemma())"
    )
    state = tb_game.new_initial_state()
    # Play 10 rounds, then continue.
    for _ in range(10):
      state.apply_action(state.legal_actions()[0])
      state.apply_action(state.legal_actions()[0])
      if state.is_chance_node():
        state.apply_action(ipd.Chance.CONTINUE)
    # Pull out the simultaneous state from inside the wrapper.
    sim_state = state.simultaneous_game_state()
    assert not state.is_terminal()
    assert not sim_state.is_terminal()
    sim_state.apply_actions([sim_state.legal_actions(0)[0],
                             sim_state.legal_actions(1)[0]])
    assert not state.is_terminal()
    assert not sim_state.is_terminal()
    # Cannot properly check is_chance_node() because the wrapper is still in
    # rollout mode, so this would fail: assert state.is_chance_node()
    assert sim_state.is_chance_node()
    sim_state.apply_action(ipd.Chance.STOP)
    assert state.is_terminal()
    assert sim_state.is_terminal()


if __name__ == "__main__":
  absltest.main()
