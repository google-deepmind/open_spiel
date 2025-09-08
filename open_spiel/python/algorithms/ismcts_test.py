# Copyright 2025 DeepMind Technologies Limited
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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from open_spiel.python.algorithms import ismcts
from open_spiel.python.algorithms import mcts
import pyspiel


class IsmctsTest(parameterized.TestCase):

  def test_action_candidates_selection(self):
    ismcts_bot = ismcts.ISMCTSBot(
        game=None,
        uct_c=1.0,
        evaluator=None,
        max_simulations=10,
    )

    # Test that the tie tolerance is respected.
    node = ismcts.ISMCTSNode()
    node.child_info = {
        0: ismcts.ChildInfo(visits=1, return_sum=7.0, prior=1.0),
        1: ismcts.ChildInfo(
            visits=1, return_sum=7.0 - ismcts.TIE_TOLERANCE / 2.0, prior=1.0
        ),
        2: ismcts.ChildInfo(
            visits=1, return_sum=7.0 - ismcts.TIE_TOLERANCE * 2.0, prior=1.0
        ),
    }
    node.total_visits = 4
    self.assertAlmostEqual(
        ismcts_bot._action_value(node, node.child_info[0]).item(),
        8.177410022515474,
    )
    candidates = ismcts_bot._select_candidate_actions(node)
    self.assertLen(candidates, 2)

    # Child 0 and 1 are selected because they are within the tie tolerance.
    self.assertIn(0, candidates)
    self.assertIn(1, candidates)

    # Child 2 is not selected because it is outside the tie tolerance.
    self.assertNotIn(2, candidates)

  def play_game(self, game: pyspiel.Game, ismcts_bot: ismcts.ISMCTSBot):
    state = game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)
      else:
        action = ismcts_bot.step(state)
        state.apply_action(action)

  def test_play_kuhn_poker(self):
    game = pyspiel.load_game("kuhn_poker")
    ismcts_bot = ismcts.ISMCTSBot(
        game=game,
        uct_c=4.0,
        evaluator=mcts.RandomRolloutEvaluator(),
        max_simulations=10,
    )
    self.play_game(game, ismcts_bot)

  def test_invalid_action_at_chance_node(self):
    game = pyspiel.load_game("kuhn_poker")
    state = game.new_initial_state()
    assert state.is_chance_node(), "Kuhn poker should start at a chance node."
    ismcts_bot = ismcts.ISMCTSBot(
        game=game,
        uct_c=4.0,
        evaluator=mcts.RandomRolloutEvaluator(),
        max_simulations=10,
    )
    policy, action = ismcts_bot.step_with_policy(state)
    self.assertLen(policy, 1)
    self.assertEqual(policy[0][0], pyspiel.INVALID_ACTION)
    self.assertEqual(action, pyspiel.INVALID_ACTION)

  @absltest.skip("Skipping. This one does not work.")
  def test_play_universal_poker(self):
    if "universal_poker" in pyspiel.registered_names():
      game = pyspiel.load_game(pyspiel.hunl_game_string("fullgame"))
      ismcts_bot = ismcts.ISMCTSBot(
          game=game,
          uct_c=4.0,
          evaluator=mcts.RandomRolloutEvaluator(),
          max_simulations=10,
      )
      self.play_game(game, ismcts_bot)


if __name__ == "__main__":
  absltest.main()
