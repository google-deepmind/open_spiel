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

# Lint as python3
"""Tests for Python Dynamic Routing Game."""

from absl.testing import absltest

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import generate_playthrough
from open_spiel.python.games import dynamic_routing_game  # pylint: disable=unused-import
import pyspiel


class DynamicRoutingGameTest(absltest.TestCase):

    def test_game_from_cc(self):
        """Runs our standard game tests, checking API consistency."""
        game = pyspiel.load_game("python_dynamic_routing_game")
        pyspiel.random_sim_test(game, num_sims=10, serialize=False,
                                verbose=True)

    def test_generate_playthrough(self):
        """Check if playthrough can be generated."""
        generate_playthrough.playthrough("python_dynamic_routing_game", None)

    def test_convert_to_turn_based(self):
        """Check if the game can be converted to turn based game."""
        game = pyspiel.load_game("python_dynamic_routing_game")
        pyspiel.convert_to_turn_based(game)

    def test_action_consistency_convert_to_turn_based(self):
        """Check if the sequential game is consistent with the game."""
        game = pyspiel.load_game("python_dynamic_routing_game")
        seq_game = pyspiel.convert_to_turn_based(game)
        state = game.new_initial_state()
        seq_state = seq_game.new_initial_state()
        self.assertEqual(
            state.legal_actions(seq_state.current_player()),
            seq_state.legal_actions(),
            msg="The sequential actions are not correct.")

    def test_cfr_on_turn_based_game(self):
        """Check if CFR can be applied to the sequential game."""
        game = pyspiel.load_game("python_dynamic_routing_game")
        seq_game = pyspiel.convert_to_turn_based(game)
        cfr.CFRSolver(seq_game)


if __name__ == "__main__":
    absltest.main()
