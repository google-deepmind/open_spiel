# Copyright 2026 DeepMind Technologies Limited
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

"""Tests for graph pursuit-evasion."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.games import graph_pursuit_evasion
from open_spiel.python.observation import make_observation
import pyspiel


class GraphPursuitEvasionTest(absltest.TestCase):

  def test_registered_game_passes_random_simulation(self):
    game = pyspiel.load_game("python_graph_pursuit_evasion")
    pyspiel.random_sim_test(game, num_sims=20, serialize=False, verbose=False)

  def test_typed_edges_restrict_each_player(self):
    game = graph_pursuit_evasion.GraphPursuitEvasionGame(
        {
            "graph": "0-1:walk,0-2:rail,1-2:walk",
            "edge_types": "walk,rail",
            "pursuer_edge_types": "walk",
            "evader_edge_types": "rail",
            "pursuer_start": 0,
            "evader_start": 2,
        }
    )
    state = game.new_initial_state()
    self.assertCountEqual(
        state.legal_actions(),
        [game.move_action(0, 1, "walk"), game.wait_action],
    )
    state.apply_action(game.wait_action)
    self.assertCountEqual(
        state.legal_actions(),
        [game.move_action(2, 0, "rail"), game.wait_action],
    )

  def test_pursuer_wins_on_capture(self):
    game = graph_pursuit_evasion.GraphPursuitEvasionGame(
        {
            "graph": "0-1:taxi,1-2:taxi",
            "edge_types": "taxi",
            "pursuer_edge_types": "taxi",
            "evader_edge_types": "taxi",
            "pursuer_start": 0,
            "evader_start": 1,
        }
    )
    state = game.new_initial_state()
    state.apply_action(game.move_action(0, 1, "taxi"))
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [1.0, -1.0])

  def test_evader_wins_on_timeout(self):
    game = graph_pursuit_evasion.GraphPursuitEvasionGame(
        {"max_rounds": 1}
    )
    state = game.new_initial_state()
    state.apply_action(game.wait_action)
    state.apply_action(game.wait_action)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [-1.0, 1.0])

  def test_evader_position_hidden_until_reveal_round(self):
    game = graph_pursuit_evasion.GraphPursuitEvasionGame(
        {
            "graph": "0-1:taxi,1-2:bus,2-3:taxi",
            "edge_types": "taxi,bus",
            "pursuer_edge_types": "taxi,bus",
            "evader_edge_types": "taxi,bus",
            "pursuer_start": 0,
            "evader_start": 2,
            "reveal_interval": 2,
        }
    )
    state = game.new_initial_state()
    state.apply_action(game.wait_action)
    state.apply_action(game.move_action(2, 3, "taxi"))

    observation = make_observation(game)
    pursuer = 0
    observation.set_from(state, pursuer)
    self.assertEqual(
        np.argmax(observation.dict["known_opponent_position"]), 2
    )
    perfect_recall = make_observation(
        game, pyspiel.IIGObservationType(perfect_recall=True)
    )
    self.assertNotIn("3", perfect_recall.string_from(state, pursuer))
    self.assertIn("2->3", perfect_recall.string_from(state, 1))

    state.apply_action(game.wait_action)
    state.apply_action(game.wait_action)
    observation.set_from(state, pursuer)
    self.assertEqual(
        np.argmax(observation.dict["known_opponent_position"]), 3
    )
    self.assertIn("reveal=3", perfect_recall.string_from(state, pursuer))

  def test_invalid_graph_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "consecutive"):
      graph_pursuit_evasion.GraphPursuitEvasionGame(
          {"graph": "0-2:taxi"}
      )
    with self.assertRaisesRegex(ValueError, "unknown edge type"):
      graph_pursuit_evasion.GraphPursuitEvasionGame(
          {"graph": "0-1:ferry"}
      )


if __name__ == "__main__":
  absltest.main()
